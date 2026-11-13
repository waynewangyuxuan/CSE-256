"""Transformer encoder and decoder written from scratch (no nn.Transformer* / nn.MultiheadAttention)."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    ff_hidden: int
    causal: bool


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        future = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("blocked", future if config.causal else torch.zeros_like(future), persistent=False)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.head_size).transpose(1, 2)

    def forward(self, x, key_is_pad):
        """x: (B, T, C); key_is_pad: (B, T) bool. Returns (B, T, C) and attention (B, n_head, T, T)."""
        B, T, C = x.shape
        q, k, v = (self.split_heads(f(x)) for f in (self.query, self.key, self.value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        blocked = self.blocked[:T, :T] | key_is_pad[:, None, None, :]
        attention = F.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1)
        mixed = (attention @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(mixed), attention


class FeedForward(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm block: x + attn(ln(x)), then x + ff(ln(x))."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_ff = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config.n_embd, config.ff_hidden, config.n_embd)

    def forward(self, x, key_is_pad):
        attended, attention = self.attn(self.ln_attn(x), key_is_pad)
        x = x + attended
        return x + self.ff(self.ln_ff(x)), attention


class TransformerStack(nn.Module):
    """Token + learned absolute position embeddings, N blocks, final layer norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_final = nn.LayerNorm(config.n_embd)

    def forward(self, ids, key_is_pad):
        """Returns contextual vectors (B, T, C) and one (B, T, T) attention map per layer and head."""
        T = ids.shape[1]
        x = self.token_emb(ids) + self.pos_emb(torch.arange(T, device=ids.device))
        attention_maps = []
        for block in self.blocks:
            x, attention = block(x, key_is_pad)
            attention_maps.extend(attention.unbind(dim=1))
        return self.ln_final(x), attention_maps


class Encoder(nn.Module):
    """Bidirectional encoder. Padding positions are never attended to."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert not config.causal
        self.stack = TransformerStack(config)

    def forward(self, ids):
        return self.stack(ids, ids == PAD_ID)


class SpeakerClassifier(nn.Module):
    """Encoder, mean over the non-padding positions, one-hidden-layer feedforward classifier."""

    def __init__(self, encoder: Encoder, n_input, n_hidden, n_output):
        super().__init__()
        self.encoder = encoder
        self.head = FeedForward(n_input, n_hidden, n_output)

    def forward(self, ids):
        vectors, attention_maps = self.encoder(ids)
        is_word = (ids != PAD_ID).unsqueeze(-1).float()
        mean = (vectors * is_word).sum(dim=1) / is_word.sum(dim=1).clamp(min=1)
        return self.head(mean), attention_maps


class Decoder(nn.Module):
    """GPT-like causal language model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.causal
        self.stack = TransformerStack(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, ids, targets=None):
        """With targets: mean next-word cross-entropy. Without: (logits, attention maps)."""
        no_padding = torch.zeros_like(ids, dtype=torch.bool)
        vectors, attention_maps = self.stack(ids, no_padding)
        logits = self.lm_head(vectors)
        if targets is None:
            return logits, attention_maps
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
