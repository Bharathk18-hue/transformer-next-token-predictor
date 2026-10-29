import os
import sys
import random
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure project root is in sys.path so both 'python src/dataset.py' and 'python -m src.dataset' work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer import SimpleTokenizer


class NextTokenDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction.
    Given a token sequence, creates shifted input and target pairs:
        input_seq:  tokens[i : i + seq_len]
        target_seq: tokens[i + 1 : i + seq_len + 1]

    Example:
        Tokens: ['machine', 'learning', 'is', 'useful']
        Input:  ['machine', 'learning', 'is']
        Target: ['learning', 'is', 'useful']
    """

    def __init__(self, token_ids: List[int], seq_len: int = 16, stride: int = 1):
        self.seq_len = seq_len
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        if len(token_ids) <= seq_len:
            # Pad sequence if total tokens are shorter than context length
            pad_needed = (seq_len + 1) - len(token_ids)
            padded = token_ids + [0] * pad_needed
            inp = torch.tensor(padded[:seq_len], dtype=torch.long)
            tgt = torch.tensor(padded[1 : seq_len + 1], dtype=torch.long)
            self.samples.append((inp, tgt))
        else:
            # Generate sliding windows
            for i in range(0, len(token_ids) - seq_len, stride):
                inp = torch.tensor(token_ids[i : i + seq_len], dtype=torch.long)
                tgt = torch.tensor(token_ids[i + 1 : i + seq_len + 1], dtype=torch.long)
                self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def load_corpus(filepath: str) -> List[str]:
    """
    Loads raw text corpus and returns clean non-empty sentences.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    return lines


def split_corpus(
    sentences: List[str], train_ratio: float = 0.85, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits sentences into training and validation sets without data leakage.
    """
    shuffled = list(sentences)
    random.seed(seed)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_sentences = shuffled[:split_idx]
    val_sentences = shuffled[split_idx:]

    return train_sentences, val_sentences


def prepare_data(
    corpus_path: str,
    seq_len: int = 16,
    batch_size: int = 16,
    stride: int = 2,
    train_ratio: float = 0.85,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """
    End-to-end data pipeline:
    1. Loads corpus
    2. Splits into train & validation
    3. Builds tokenizer vocabulary from train set
    4. Encodes tokens and constructs PyTorch DataLoaders
    """
    sentences = load_corpus(corpus_path)
    train_sentences, val_sentences = split_corpus(sentences, train_ratio=train_ratio, seed=seed)

    tokenizer = SimpleTokenizer(min_freq=1)
    tokenizer.build_vocab(train_sentences)

    # Encode sentences with EOS delimiter
    train_tokens: List[int] = []
    for s in train_sentences:
        train_tokens.extend(tokenizer.encode(s) + [tokenizer.eos_id])

    val_tokens: List[int] = []
    for s in val_sentences:
        val_tokens.extend(tokenizer.encode(s) + [tokenizer.eos_id])

    train_dataset = NextTokenDataset(train_tokens, seq_len=seq_len, stride=stride)
    val_dataset = NextTokenDataset(val_tokens, seq_len=seq_len, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer


def display_sample_pairs(
    dataset: NextTokenDataset, tokenizer: SimpleTokenizer, num_samples: int = 2
) -> None:
    """
    Displays sample input-target pairs for educational demonstration.
    """
    print("=" * 60)
    print("SAMPLE INPUT-TARGET PAIRS FOR NEXT-TOKEN PREDICTION")
    print("=" * 60)
    for idx in range(min(num_samples, len(dataset))):
        inp, tgt = dataset[idx]
        inp_words = [tokenizer.idx2word.get(i.item(), "<UNK>") for i in inp]
        tgt_words = [tokenizer.idx2word.get(t.item(), "<UNK>") for t in tgt]

        print(f"\n[Sample {idx + 1}]")
        print(f"Input IDs  : {inp.tolist()[:8]} ...")
        print(f"Target IDs : {tgt.tolist()[:8]} ...")
        print(f"Input text : {' '.join(inp_words[:8])} ...")
        print(f"Target text: {' '.join(tgt_words[:8])} ...")
        print("Alignment  : At each step, target is shifted forward by 1 token.")
    print("=" * 60)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_file = os.path.join(current_dir, "..", "data", "corpus.txt")
    train_ldr, val_ldr, tok = prepare_data(corpus_file, seq_len=8, batch_size=4)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Train batches: {len(train_ldr)}, Val batches: {len(val_ldr)}")
    display_sample_pairs(train_ldr.dataset, tok, num_samples=2)
