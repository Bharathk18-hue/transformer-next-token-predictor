import os
import sys
import math
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tokenizer import SimpleTokenizer
from src.dataset import prepare_data
from src.transformer import TransformerLM
from src.generate import load_model_and_tokenizer


def evaluate_model(
    model: TransformerLM,
    val_loader: DataLoader,
    tokenizer: SimpleTokenizer,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluates model on validation data:
      - Validation Cross-Entropy Loss
      - Validation Perplexity = exp(Val Loss)
      - Top-1 Next-Token Accuracy
      - Top-5 Next-Token Accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    total_loss = 0.0
    total_tokens = 0
    correct_top1 = 0
    correct_top5 = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs, causal=True)
            vocab_size = logits.size(-1)

            # Flatten for loss & accuracy
            flat_logits = logits.view(-1, vocab_size)
            flat_targets = targets.view(-1)

            # Mask out padding tokens
            valid_mask = flat_targets != tokenizer.pad_id
            active_logits = flat_logits[valid_mask]
            active_targets = flat_targets[valid_mask]

            if len(active_targets) == 0:
                continue

            loss = criterion(active_logits, active_targets)
            total_loss += loss.item() * len(active_targets)
            total_tokens += len(active_targets)

            # Top-1 accuracy
            top1_preds = torch.argmax(active_logits, dim=-1)
            correct_top1 += (top1_preds == active_targets).sum().item()

            # Top-5 accuracy
            _, top5_preds = torch.topk(active_logits, k=min(5, vocab_size), dim=-1)
            correct_top5 += (
                top5_preds == active_targets.unsqueeze(-1)
            ).any(dim=-1).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 20.0))
    top1_acc = (correct_top1 / max(1, total_tokens)) * 100
    top5_acc = (correct_top5 / max(1, total_tokens)) * 100

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_evaluated_tokens": total_tokens,
    }


def compare_actual_vs_predicted(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    test_phrases: List[str],
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Demonstrates actual next-token predictions on test phrases.
    """
    print("\n" + "=" * 75)
    print("ACTUAL VS. PREDICTED NEXT-TOKEN COMPARISON")
    print("=" * 75)
    print(f"{'Input Context':<30} | {'Predicted Next':<16} | {'Top-5 Candidates'}")
    print("-" * 75)

    model.eval()
    for phrase in test_phrases:
        tokens = tokenizer.encode(phrase)
        if not tokens:
            continue
        input_tensor = torch.tensor([tokens[-model.context_length :]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(input_tensor, causal=True)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(5, len(probs)))

        top1_token = tokenizer.idx2word.get(top_indices[0].item(), "<UNK>")
        top1_pct = f"{top_probs[0].item() * 100:.1f}%"
        cand_str = ", ".join(
            f"{tokenizer.idx2word.get(idx.item(), '<UNK>')}({p.item() * 100:.0f}%)"
            for p, idx in zip(top_probs[1:], top_indices[1:])
        )

        pred_display = f"{top1_token} ({top1_pct})"
        print(f"{phrase:<30} | {pred_display:<16} | {cand_str}")
    print("=" * 75)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(project_root, "models", "best_model.pt")
    vocab_file = os.path.join(project_root, "models", "vocab.json")
    corpus_file = os.path.join(project_root, "data", "corpus.txt")

    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found. Please train first!")
        exit(1)

    model, tokenizer = load_model_and_tokenizer(model_file, vocab_file)

    _, val_loader, _ = prepare_data(
        corpus_path=corpus_file,
        seq_len=model.context_length,
        batch_size=16,
        stride=2,
    )

    metrics = evaluate_model(model, val_loader, tokenizer)
    print("=" * 50)
    print("VALIDATION EVALUATION METRICS")
    print("=" * 50)
    print(f"Validation Loss       : {metrics['val_loss']:.4f}")
    print(f"Validation Perplexity : {metrics['perplexity']:.2f}")
    print(f"Top-1 Accuracy        : {metrics['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy        : {metrics['top5_accuracy']:.2f}%")
    print(f"Tokens Evaluated      : {metrics['total_evaluated_tokens']}")
    print("=" * 50)

    test_examples = [
        "the cat sat on the",
        "machine learning is",
        "deep learning models",
        "natural language",
        "the transformer architecture was",
        "neural networks learn",
    ]
    compare_actual_vs_predicted(model, tokenizer, test_examples)
