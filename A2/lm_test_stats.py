"""Prints, for each language-modeling test set, how many of its words the decoder never saw in training."""
import os
from collections import Counter

from nltk.tokenize import word_tokenize

from main import DATA_DIR, LM_TEST_SETS, build_tokenizer


def main():
    tokenizer = build_tokenizer()
    with open(os.path.join(DATA_DIR, "train_LM.txt"), encoding="utf-8") as f:
        seen_in_lm_training = Counter(word_tokenize(f.read()))
    print(f"train_LM: {sum(seen_in_lm_training.values())} tokens, {len(seen_in_lm_training)} distinct words")
    for name in LM_TEST_SETS:
        with open(os.path.join(DATA_DIR, f"test_LM_{name}.txt"), encoding="utf-8") as f:
            words = word_tokenize(f.read())
        unk = sum(word not in tokenizer.stoi for word in words) / len(words)
        unseen = sum(word not in seen_in_lm_training for word in words) / len(words)
        print(f"{name} | tokens {len(words)} | unk {100 * unk:.2f}% | not in train_LM {100 * unseen:.2f}%")


if __name__ == "__main__":
    main()
