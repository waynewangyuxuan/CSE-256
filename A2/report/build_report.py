"""Builds report/partA.html from the training logs, so every number in the report comes from a run."""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDENT_NAME = "Yuxuan (Wayne) Wang"
SPEAKERS = {"obama": "Obama", "wbush": "W. Bush", "hbush": "H. W. Bush"}


def read_log(name):
    with open(os.path.join(ROOT, "logs", name), encoding="utf-8") as f:
        return f.read()


def one(pattern, text):
    return re.search(pattern, text).group(1)


part1, part2, stats = read_log("part1.log"), read_log("part2.log"), read_log("lm_test_stats.log")

vocab_size = int(one(r"Vocabulary size is (\d+)", part1))
encoder_params = int(one(r"Encoder parameters: (\d+)", part1))
head_params = int(one(r"Classifier head parameters: (\d+)", part1))
decoder_params = int(one(r"Decoder parameters: (\d+)", part2))
epochs = [tuple(map(float, m)) for m in re.findall(
    r"epoch\s+(\d+) \| train loss ([\d.]+) \| train acc ([\d.]+) \| test acc ([\d.]+)", part1)]
final_accuracy = float(one(r"Final test accuracy: ([\d.]+)", part1))
ppl_rows = [tuple(map(float, m)) for m in re.findall(
    r"iter\s+(\d+) \| train ppl ([\d.]+) \| obama ppl ([\d.]+) \| wbush ppl ([\d.]+) \| hbush ppl ([\d.]+)", part2)]
during_training, final_ppl = ppl_rows[:-1], ppl_rows[-1]
lm_stats = {m[0]: m[1:] for m in re.findall(
    r"(\w+) \| tokens (\d+) \| unk ([\d.]+)% \| not in train_LM ([\d.]+)%", stats)}
assert len(epochs) == 15 and len(during_training) == 5 and len(lm_stats) == 3
assert part1.count("All attention rows sum to 1: True") == 2 and part2.count("All attention rows sum to 1: True") == 2

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].plot([e[0] for e in epochs], [e[2] for e in epochs], marker="o", label="train", color="#9ecae1")
axes[0].plot([e[0] for e in epochs], [e[3] for e in epochs], marker="o", label="test", color="#08519c")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy (%)"); axes[0].set_title("Part 1: classifier accuracy")
for column, (key, label), color in zip((2, 3, 4), SPEAKERS.items(), ("#08519c", "#d6278a", "#f39c12")):
    axes[1].plot([r[0] for r in during_training], [r[column] for r in during_training], marker="o", label=f"test {label}", color=color)
axes[1].plot([r[0] for r in during_training], [r[1] for r in during_training], marker="o", label="train", color="#9e9e9e")
axes[1].set_xlabel("iteration"); axes[1].set_ylabel("perplexity"); axes[1].set_title("Part 2: decoder perplexity")
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "figures", "report_curves.png"), dpi=170)

best_epoch = max(epochs, key=lambda e: e[3])
epoch_table = "".join(
    f"<tr><td>{int(e)}</td><td>{loss:.4f}</td><td>{train:.2f}</td><td>{test:.2f}</td></tr>" for e, loss, train, test in epochs)
ppl_table = "".join(
    f"<tr><td>{int(i)}</td><td>{tr:.1f}</td><td>{o:.1f}</td><td>{w:.1f}</td><td>{h:.1f}</td></tr>" for i, tr, o, w, h in during_training)
ppl_table += (f"<tr class='final'><td>final</td><td>{final_ppl[1]:.1f}</td><td>{final_ppl[2]:.1f}</td>"
              f"<td>{final_ppl[3]:.1f}</td><td>{final_ppl[4]:.1f}</td></tr>")
final_by_speaker = dict(zip(SPEAKERS, final_ppl[2:]))
stats_table = "".join(
    f"<tr><td>{SPEAKERS[k]}</td><td>{final_by_speaker[k]:.1f}</td><td>{lm_stats[k][0]}</td>"
    f"<td>{lm_stats[k][1]}%</td><td>{lm_stats[k][2]}%</td></tr>" for k in SPEAKERS)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>CSE 256 PA2 report</title>
<style>
@page {{ size: Letter; margin: 0.8in; }}
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 2pt; }} h2 {{ font-size: 14pt; margin: 16pt 0 4pt; border-bottom: 1px solid #999; }}
h3 {{ font-size: 11.5pt; margin: 10pt 0 2pt; }} p {{ margin: 4pt 0 6pt; }}
table {{ border-collapse: collapse; font-size: 9.5pt; margin: 6pt 0; }}
th, td {{ border: 1px solid #bbb; padding: 2pt 7pt; text-align: right; }} th {{ background: #eee; }}
td:first-child, th:first-child {{ text-align: left; }} tr.final td {{ font-weight: bold; background: #f6f6f6; }}
.cols {{ display: flex; gap: 18pt; align-items: flex-start; }} .cols > div {{ flex: 1; }}
figure {{ margin: 6pt 0; text-align: center; break-inside: avoid; }} figcaption {{ font-size: 9pt; color: #333; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6pt; }} .grid2 img {{ width: 100%; }}
code {{ font-size: 9.5pt; }} .meta {{ color: #444; font-size: 10pt; }} .break {{ break-before: page; }}
</style></head><body>
<h1>CSE 256 PA2: Transformer Blocks + Dissecting a Pretrained LLM</h1>
<p class="meta">{STUDENT_NAME} &middot; Fall 2026 &middot; Part A (Parts 1 and 2). Part B, the executed Part 3 notebook, follows this part.</p>

<h2>Implementation shared by Parts 1 and 2</h2>
<p>Everything is in <code>transformer.py</code> and uses only <code>nn.Linear</code>, <code>nn.Embedding</code>, <code>nn.LayerNorm</code> and tensor operations; no library Transformer or attention layers. Both models are the same stack, built from one configuration, with a flag that turns the causal mask on:</p>
<ul>
<li><b>Embeddings.</b> A token embedding table and a learned absolute position embedding table (32 positions), added together before the first block.</li>
<li><b>Multi-head self-attention.</b> Query, key and value projections of size 64, split into 2 heads of size 32; scores are scaled by 1/&radic;32, masked, and passed through a softmax; the head outputs are concatenated and projected.</li>
<li><b>Block.</b> Pre-norm residual block: <code>x + attn(LN(x))</code>, then <code>x + ff(LN(x))</code>, with a ReLU feedforward network. A final layer norm follows the last of the 4 blocks.</li>
<li><b>Masks.</b> The encoder never attends to padding positions. The decoder never attends to future positions.</li>
</ul>
<p>All hyperparameters are the defaults in <code>main.py</code> (batch size 16, block size 32, learning rate 1e-3, <i>n</i><sub>embd</sub> = 64, 2 heads, 4 layers); the optimizer is AdamW, the seed is 42, and the vocabulary has {vocab_size:,} words. <code>selfcheck.py</code> tests the pieces directly: attention rows sum to 1, padding receives zero attention, the amount of padding does not change the classifier's output, and changing future tokens leaves the decoder's earlier logits unchanged.</p>

<h2>Part 1: Encoder trained with a classifier</h2>
<p>The encoder returns one vector per word. The classifier averages them over the real (non-padding) words of the segment and passes the mean through a feedforward network with one hidden layer (64 &rarr; 100 &rarr; 3). Encoder and classifier are trained jointly from scratch with cross-entropy for 15 epochs. The feedforward network inside each encoder block has hidden size 256 (4 &times; <i>n</i><sub>embd</sub>); the handout fixes this size only for the decoder.</p>
<p><b>Parameters.</b> The encoder has <b>{encoder_params:,}</b> parameters: 360,576 in the token embeddings, 2,048 in the position embeddings, 49,792 in each of the 4 blocks, and 128 in the final layer norm. The classifier head adds {head_params:,}.</p>
<div class="cols"><div>
<table><tr><th>Epoch</th><th>Train loss</th><th>Train acc. (%)</th><th>Test acc. (%)</th></tr>{epoch_table}
<tr class="final"><td>final</td><td></td><td></td><td>{final_accuracy:.2f}</td></tr></table>
</div><div>
<p><b>Result.</b> The final test accuracy on <code>test_CLS.tsv</code> is <b>{final_accuracy:.2f}%</b>, in the expected low-to-mid 80s. The test set is balanced (250 segments per speaker), so chance is 33.3%.</p>
<p>The model learns slowly for the first three epochs (test accuracy {epochs[0][3]:.1f}%, {epochs[1][3]:.1f}%, {epochs[2][3]:.1f}%), then improves quickly and flattens after epoch 9. From epoch 9 on, test accuracy stays between {min(e[3] for e in epochs[8:]):.2f}% and {max(e[3] for e in epochs[8:]):.2f}% while training accuracy is above 98%. The best epoch is epoch {int(best_epoch[0])} ({best_epoch[3]:.2f}%); the small moves after that are within about 1.5 points and look like noise from a model that has already fit the 2,092 training segments. I report the last epoch, as the assignment asks, not the best one.</p>
</div></div>
<figure><img src="../figures/report_curves.png" style="width: 100%"><figcaption>Figure 1. Left: classifier accuracy per epoch. Right: decoder perplexity every 100 iterations (train, and the three test sets).</figcaption></figure>

<h3>Sanity checks and attention maps</h3>
<p><code>Utilities.sanity_check</code> was run on two sentences. For both, all 8 attention maps (4 layers &times; 2 heads) passed the test that every row sums to 1. Figure 2 shows two maps for the sentence "That's how progress happens -- in societies and in our own lives."</p>
<figure><div class="grid2"><img src="../figures/encoder_sentence1_L1H1_words.png"><img src="../figures/encoder_sentence1_L4H1_full.png"></div>
<figcaption>Figure 2. Encoder attention. Left: layer 1, head 1, cropped to the 14 real tokens. Right: layer 4, head 1, all 32 positions (14 tokens, then padding). Rows are queries, columns are keys.</figcaption></figure>
<p><b>What the maps show.</b> In layer 1 the attention is sparse and depends on content, not on position: there is no bright diagonal. The queries "how", "progress" and "happens" all put most of their weight on the single word "progress" (about 0.6 to 0.8), the final period attends mostly to "and", and "'s" to "own". Because the encoder is bidirectional, queries attend to words on both sides.</p>
<p>In layer 4 the picture is different. Every row is close to uniform over the 14 real tokens (weights between about 0.04 and 0.15, where exactly uniform would be 0.071), with "progress" slightly brighter. That fits how the output is used: the classifier takes the mean of the output vectors, so by the last layer each position is already close to a summary of the whole segment. The right half of the map is exactly zero: columns 14 to 31 are padding, and the mask stops any query from attending to them. Each row still sums to 1 over the real tokens.</p>

<h2 class="break">Part 2: Pretraining a decoder language model</h2>
<p>The decoder is the same stack with the causal mask on, a feedforward hidden size of 100 with ReLU, and a linear layer from the 64-dimensional output to the vocabulary. It is trained for 500 iterations on <code>train_LM.txt</code> to predict the next word at every position (cross-entropy over the vocabulary). Perplexity is the exponential of the mean loss over up to 200 batches of 16 sequences.</p>
<p><b>Parameters.</b> The decoder has <b>{decoder_params:,}</b> parameters: 360,576 token embeddings, 2,048 position embeddings, 29,668 in each of the 4 blocks, 128 in the final layer norm, and 366,210 in the output layer.</p>
<div class="cols"><div>
<table><tr><th>Iteration</th><th>Train</th><th>Obama</th><th>W. Bush</th><th>H. W. Bush</th></tr>{ppl_table}</table>
<p style="font-size:9pt">Perplexity. The two rows for iteration 500 differ slightly in the train column because the training loader is shuffled, so the 200 evaluation batches are different each time.</p>
</div><div>
<p><b>Result.</b> Final training perplexity is {final_ppl[1]:.1f} (expected: high 100s). Final test perplexities are <b>{final_ppl[2]:.1f}</b> for Obama, <b>{final_ppl[3]:.1f}</b> for W. Bush and <b>{final_ppl[4]:.1f}</b> for H. W. Bush (expected: 300s and 400s). All four fall steadily at every evaluation and are still falling at iteration 500, so the model is undertrained, as the handout says to expect after about 256,000 tokens.</p>
<p>The gap between training and test perplexity widens during training (from {during_training[0][2] - during_training[0][1]:.0f} points for Obama at iteration 100 to {during_training[-1][2] - during_training[-1][1]:.0f} at iteration 500): the training file has only 32,288 tokens and the 500 batches contain 256,000, so each training token is seen about 8 times and the model starts to fit the training text itself.</p>
</div></div>

<h3>Why the three politicians get different perplexities</h3>
<p>The order is the same at every evaluation: Obama lowest, H. W. Bush close behind, W. Bush clearly highest ({final_ppl[3]:.1f}, about {100 * (final_ppl[3] / final_ppl[2] - 1):.0f}% above Obama). One cause can be measured directly (<code>lm_test_stats.py</code>): how many test words the decoder never saw during language-model training.</p>
<table><tr><th>Test set</th><th>Final perplexity</th><th>Tokens</th><th>Mapped to &lt;unk&gt;</th><th>Never seen in <code>train_LM.txt</code></th></tr>{stats_table}</table>
<p>The tokenizer's vocabulary is built from both training files, but the decoder is trained only on <code>train_LM.txt</code>. A test word that appears only in the classification data has an output row that was never the target of a gradient step, and a word in neither file becomes &lt;unk&gt;, which also never occurs as a training target. In both cases the model gives the true word almost no probability, and each such word adds a large loss. The share of these words is lowest for Obama, higher for H. W. Bush and highest for W. Bush, the same order as the perplexities.</p>
<p>This is probably not the whole explanation. Unseen words are a few percent of each test set, and the speakers also differ in topic and phrasing on the words the model does know. How much of each speaker's text is in <code>train_LM.txt</code> must matter as well, but that file is unlabeled, so I could not measure it. What the numbers do support is that perplexity here measures how close each speaker's test text is to the training text, not how "predictable" the speaker is in general.</p>

<h3>Sanity checks and attention maps</h3>
<p>All 8 decoder attention maps passed the rows-sum-to-1 test on both sentences. Figure 3 shows the same sentence as Figure 2.</p>
<figure><div class="grid2"><img src="../figures/decoder_sentence1_L1H1_words.png"><img src="../figures/decoder_sentence1_L4H1_full.png"></div>
<figcaption>Figure 3. Decoder attention. Left: layer 1, head 1, cropped to the 14 real tokens. Right: layer 4, head 1, all 32 positions.</figcaption></figure>
<p><b>What the mask looks like.</b> Both maps are lower-triangular: everything above the diagonal is exactly zero, because query <i>i</i> can only attend to keys <i>j</i> &le; <i>i</i>. The first row has a single cell with weight 1.0, since the first word can only attend to itself, and each later row spreads its weight over one more position, which is why the rows get dimmer further down. In layer 1 the weight tends to sit on the last few words (a band along the diagonal; "our" puts about 0.6 on itself), which makes sense for next-word prediction with so little training. Unlike the encoder, the decoder has no padding mask, because the language-modeling data has no padding: in the full 32-position map the rows below position 13 belong to padding tokens that the sanity-check utility appends, and they attend backwards over the sentence like any other position.</p>
</body></html>"""
with open(os.path.join(HERE, "partA.html"), "w", encoding="utf-8") as f:
    f.write(html)
print("wrote report/partA.html")
