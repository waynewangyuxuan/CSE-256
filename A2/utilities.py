import os

import matplotlib.pyplot as plt
import torch


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, out_dir, prefix, n_head, plot=((0, 0),)):
        """Checks that every attention row of every map sums to 1, and saves a heatmap
        for each (layer, head) pair in `plot`. Returns True when all maps pass."""
        wordids = self.tokenizer.encode(sentence)[:block_size]
        words = [self.tokenizer.itos[i] for i in wordids]
        padded_sentence = wordids + [0] * (block_size - len(wordids))
        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long, device=device).unsqueeze(0)
        print("Input tensor shape:", tuple(input_tensor.shape))

        self.model.eval()
        with torch.no_grad():
            _, attn_maps = self.model(input_tensor)
        self.model.train()
        print("Number of attention maps:", len(attn_maps))

        all_normalized = True
        for j, attn_map in enumerate(attn_maps):
            total_prob_over_rows = torch.sum(attn_map[0], dim=1)
            if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                all_normalized = False
                print(f"Failed normalization test in map {j}: rows sum to", total_prob_over_rows.cpu().numpy())
        print("All attention rows sum to 1:", all_normalized)

        os.makedirs(out_dir, exist_ok=True)
        n = len(words)
        for layer, head in plot:
            att_map = attn_maps[layer * n_head + head][0].cpu().numpy()
            for name, shown, labels in (("full", att_map, None), ("words", att_map[:n, :n], words)):
                fig, ax = plt.subplots(figsize=(6, 5))
                cax = ax.imshow(shown, cmap="hot", interpolation="nearest", vmin=0)
                ax.xaxis.tick_top()
                if labels:
                    ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
                    ax.set_yticks(range(n), labels, fontsize=7)
                ax.set_xlabel("key (attended to)")
                ax.set_ylabel("query")
                fig.colorbar(cax, ax=ax)
                ax.set_title(f"{prefix}: layer {layer + 1}, head {head + 1}", y=-0.12)
                path = os.path.join(out_dir, f"{prefix}_L{layer + 1}H{head + 1}_{name}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("Saved", path)
        return all_normalized
