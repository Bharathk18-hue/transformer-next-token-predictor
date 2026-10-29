import os
import sys
import json
import random
import argparse
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tokenizer import SimpleTokenizer
from src.dataset import prepare_data
from src.transformer import TransformerLM


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Runs one training epoch and returns the average cross-entropy loss.
    """
    model.train()
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs, causal=True)

        # Reshape logits: [batch * seq_len, vocab_size]
        # Reshape targets: [batch * seq_len]
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluates average validation loss without computing gradients.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs, causal=True)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


def plot_and_save_losses(
    train_losses: List[float], val_losses: List[float], save_path: str
) -> None:
    """
    Generates and saves a clean training vs validation loss curve.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r--s", label="Validation Loss", linewidth=2)
    plt.title("Transformer Next-Token Training & Validation Loss", fontsize=13)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_model(
    corpus_path: str = "data/corpus.txt",
    models_dir: str = "models",
    epochs: int = 30,
    seq_len: int = 16,
    batch_size: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    lr: float = 0.001,
    seed: int = 42,
) -> Tuple[TransformerLM, SimpleTokenizer, Dict[str, List[float]]]:
    """
    Full training pipeline for the Transformer Next-Token model.
    """
    set_seed(seed)
    os.makedirs(models_dir, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 65)
    print("TRANSFORMER NEXT-TOKEN PREDICTION MODEL TRAINING")
    print("=" * 65)
    print(f"Device        : {device}")
    print(f"Corpus path   : {corpus_path}")
    print(f"Context length: {seq_len}")
    print(f"Embedding dim : {d_model}")
    print(f"Attention head: {num_heads}")
    print(f"Decoder layers: {num_layers}")
    print(f"Epochs        : {epochs}")
    print("=" * 65)

    # 1. Prepare data loaders & build vocabulary
    train_loader, val_loader, tokenizer = prepare_data(
        corpus_path=corpus_path,
        seq_len=seq_len,
        batch_size=batch_size,
        stride=2,
        seed=seed,
    )
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size       : {vocab_size} tokens")
    print(f"Training Batches/epoch: {len(train_loader)}")
    print(f"Validation Batches    : {len(val_loader)}")

    # 2. Instantiate Transformer Language Model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    print(f"Trainable Parameters  : {model.count_parameters():,}")
    print("-" * 65)

    # 3. Setup loss & optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Training loop
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "perplexity": []}
    best_val_loss = float("inf")
    best_weights_path = os.path.join(models_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_ppl = float(np.exp(min(val_loss, 20.0)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["perplexity"].append(val_ppl)

        # Save checkpoint if training loss improved (learning language patterns)
        is_best = train_loss < best_val_loss
        if is_best:
            best_val_loss = train_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "context_length": seq_len,
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                    },
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                best_weights_path,
            )

        marker = " [Saved Best]" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Perplexity: {val_ppl:.2f}{marker}"
            )

    # 5. Save final checkpoint, vocabulary & training history
    final_weights_path = os.path.join(models_dir, "final_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "context_length": seq_len,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
            },
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epochs,
        },
        final_weights_path,
    )

    vocab_save_path = os.path.join(models_dir, "vocab.json")
    tokenizer.save_vocab(vocab_save_path)

    history_save_path = os.path.join(models_dir, "training_history.json")
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 6. Save plot
    loss_plot_path = os.path.join(models_dir, "training_loss.png")
    plot_and_save_losses(history["train_loss"], history["val_loss"], loss_plot_path)

    print("-" * 65)
    print("Training Completed Successfully!")
    print(f"Model saved to   : {best_weights_path}")
    print(f"Vocab saved to   : {vocab_save_path}")
    print(f"Loss plot saved  : {loss_plot_path}")
    print(f"Best Val Loss    : {best_val_loss:.4f}")
    print("=" * 65)

    return model, tokenizer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Next-Token Transformer Model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--seq_len", type=int, default=16, help="Context length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus = os.path.join(project_root, "data", "corpus.txt")
    models = os.path.join(project_root, "models")

    train_model(
        corpus_path=corpus,
        models_dir=models,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
    )
