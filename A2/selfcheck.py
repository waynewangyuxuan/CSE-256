"""Fast checks: shapes, attention rows sum to 1, the decoder cannot see the future."""
import torch

from transformer import TransformerConfig, Encoder, SpeakerClassifier, Decoder

torch.manual_seed(0)
base = dict(vocab_size=50, block_size=32, n_embd=64, n_head=2, n_layer=4, ff_hidden=100)
ids = torch.randint(2, 50, (3, 32))
ids[0, 20:] = 0

classifier = SpeakerClassifier(Encoder(TransformerConfig(**base, causal=False)), 64, 100, 3).eval()
logits, maps = classifier(ids)
assert logits.shape == (3, 3) and len(maps) == 8 and maps[0].shape == (3, 32, 32)
assert all(torch.allclose(m.sum(-1), torch.ones(3, 32), atol=1e-5) for m in maps), "encoder rows must sum to 1"
assert maps[0][0, :, 20:].abs().max() == 0, "padding must receive no attention"
longer = torch.cat([ids[:1, :20], torch.zeros(1, 5, dtype=torch.long)], dim=1)
assert torch.allclose(classifier(longer)[0], logits[:1], atol=1e-5), "amount of padding must not change the prediction"

decoder = Decoder(TransformerConfig(**base, causal=True)).eval()
ids = torch.randint(2, 50, (3, 32))
logits, maps = decoder(ids)
assert logits.shape == (3, 32, 50)
assert all(torch.allclose(m.sum(-1), torch.ones(3, 32), atol=1e-5) for m in maps), "decoder rows must sum to 1"
assert all(torch.triu(m, diagonal=1).abs().max() == 0 for m in maps), "decoder must not attend to the future"
future_changed = ids.clone()
future_changed[:, 16:] = torch.randint(2, 50, (3, 16))
assert torch.allclose(decoder(future_changed)[0][:, :16], logits[:, :16], atol=1e-5), "future tokens leaked into the past"
assert decoder(ids, ids).ndim == 0
print("selfcheck ok")
