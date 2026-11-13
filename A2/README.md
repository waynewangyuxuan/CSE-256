# CSE 256 PA2: Transformer Blocks + Dissecting a Pretrained LLM

## Setup

Python 3.12 with `torch`, `nltk`, `matplotlib` (Parts 1 and 2) and `transformers`, `numpy`, `jupyter` (Part 3).

```
pip install torch nltk matplotlib numpy transformers jupyter
python -c "import nltk; nltk.download('punkt_tab')"
```

## Parts 1 and 2

Run from this directory. All hyperparameters are the defaults at the top of `main.py`.

```
python main.py part1    # encoder + classifier: accuracy after each of 15 epochs, attention sanity checks
python main.py part2    # decoder LM: perplexity every 100 iterations, attention sanity checks
python main.py          # both
```

Each part takes under a minute on a laptop CPU. Output goes to the terminal; attention maps are saved to `figures/`.

| File | What it holds |
|---|---|
| `transformer.py` | The model, written from scratch: attention, feedforward, block, encoder, classifier, decoder |
| `main.py` | Hyperparameters, data loading, the two training loops, evaluation |
| `utilities.py` | Attention sanity check (rows sum to 1) and attention-map plots |
| `tokenizer.py`, `dataset.py` | Starter code. The only change: the tokenizer sorts its vocabulary so runs are reproducible |
| `selfcheck.py` | Fast tests: `python selfcheck.py` checks shapes, row sums, the padding mask and the causal mask |
| `lm_test_stats.py` | `python lm_test_stats.py` prints how many words of each LM test set were never seen in training (used in the report) |
| `logs/` | Output of the runs the report is based on |

## Part 3

`part3.ipynb` is the executed notebook, with all outputs. To run it again:

```
jupyter nbconvert --to notebook --execute --inplace part3.ipynb
```

It expects `speechesdataset/test_LM_obama.txt` next to the notebook and downloads six checkpoints of
`EleutherAI/pythia-160m-deduped` (about 4 GB) on first use. A full run takes about 2 minutes on a laptop CPU once
the checkpoints are cached.

## Report

`PA2_report.pdf` is Part A (Parts 1 and 2) followed by Part B (a PDF export of `part3.ipynb`).
`report/export_pdf.sh` rebuilds it: Part A is generated from `logs/` by `report/build_report.py`, so its numbers
come from the runs, and Part B is exported from the executed notebook. The script uses Google Chrome to print to PDF.
