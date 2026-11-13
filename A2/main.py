import argparse
import os

import matplotlib
matplotlib.use("Agg")
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerConfig, Encoder, SpeakerClassifier, Decoder, count_parameters
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

n_ff_encoder = 4 * n_embd  # Feedforward hidden size inside each encoder block
n_ff_decoder = 100  # Feedforward hidden size inside each decoder block (given in the handout)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speechesdataset")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SANITY_SENTENCES = (
    "That's how progress happens -- in societies and in our own lives.",
    "The level of world cooperation and condemnation of Iraq is unprecedented.",
)
LM_TEST_SETS = ("obama", "wbush", "hbush")


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = sorted(os.listdir(directory))
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Perplexity is exp of the mean cross entropy loss over up to eval_iters batches.
    """
    decoderLMmodel.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            losses.append(decoderLMmodel(X, Y).item())
            if len(losses) >= eval_iters: break

    perplexity = torch.exp(torch.tensor(losses).mean()).item()
    decoderLMmodel.train()
    return perplexity


def build_tokenizer():
    print("Loading data and creating tokenizer ...")
    tokenizer = SimpleTokenizer(' '.join(load_texts(DATA_DIR)))
    print("Vocabulary size is", tokenizer.vocab_size)
    return tokenizer


def transformer_config(tokenizer, causal):
    return TransformerConfig(
        vocab_size=tokenizer.vocab_size, block_size=block_size, n_embd=n_embd, n_head=n_head,
        n_layer=n_layer, ff_hidden=n_ff_decoder if causal else n_ff_encoder, causal=causal)


def run_sanity_checks(tokenizer, model, prefix):
    utilities = Utilities(tokenizer, model)
    for i, sentence in enumerate(SANITY_SENTENCES, start=1):
        print(f"Sanity check on: {sentence}")
        utilities.sanity_check(sentence, block_size, FIGURE_DIR, f"{prefix}_sentence{i}", n_head,
                               plot=((0, 0), (n_layer - 1, 0)))


def part1(tokenizer):
    """Train the encoder and the speaker classifier jointly, from scratch."""
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(DATA_DIR, "train_CLS.tsv"))
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, os.path.join(DATA_DIR, "test_CLS.tsv"))
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

    encoder = Encoder(transformer_config(tokenizer, causal=False))
    classifier = SpeakerClassifier(encoder, n_input, n_hidden, n_output).to(device)
    print("Encoder parameters:", count_parameters(encoder))
    print("Classifier head parameters:", count_parameters(classifier.head))

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    for epoch in range(epochs_CLS):
        total_loss = 0.0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = classifier(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_accuracy = compute_classifier_accuracy(classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
        print(f"epoch {epoch + 1:2d} | train loss {total_loss / len(train_CLS_loader):.4f} "
              f"| train acc {train_accuracy:.2f} | test acc {test_accuracy:.2f}")
    print(f"Final test accuracy: {test_accuracy:.2f}")

    run_sanity_checks(tokenizer, classifier, "encoder")


def part2(tokenizer):
    """Pretrain the decoder on next-word prediction and report perplexity per politician."""
    def lm_loader(filename, shuffle):
        with open(os.path.join(DATA_DIR, filename), 'r', encoding='utf-8') as f:
            dataset = LanguageModelingDataset(tokenizer, f.read(), block_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_LM_loader = lm_loader("train_LM.txt", shuffle=True)
    eval_loaders = {"train": train_LM_loader}
    eval_loaders.update({name: lm_loader(f"test_LM_{name}.txt", shuffle=False) for name in LM_TEST_SETS})

    decoder = Decoder(transformer_config(tokenizer, causal=True)).to(device)
    print("Decoder parameters:", count_parameters(decoder))

    def report(iteration):
        perplexities = {name: compute_perplexity(decoder, loader, eval_iters) for name, loader in eval_loaders.items()}
        print(f"iter {iteration:3d} | " + " | ".join(f"{name} ppl {value:.1f}" for name, value in perplexities.items()))

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        loss = decoder(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i + 1) % eval_interval == 0:
            report(i + 1)
    print("Final perplexities:")
    report(max_iters)

    run_sanity_checks(tokenizer, decoder, "decoder")


PARTS = {"part1": part1, "part2": part2}


def main():
    parser = argparse.ArgumentParser(description="CSE 256 PA2, Parts 1 and 2")
    parser.add_argument("part", nargs="?", choices=[*PARTS, "all"], default="all")
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    for name in (PARTS if args.part == "all" else [args.part]):
        torch.manual_seed(seed)
        print(f"===== {name} =====")
        PARTS[name](tokenizer)


if __name__ == "__main__":
    main()
