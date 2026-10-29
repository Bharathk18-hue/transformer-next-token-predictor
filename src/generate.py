import os
import sys
import argparse
from typing import List, Tuple, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

from src.tokenizer import SimpleTokenizer
from src.transformer import TransformerLM


def load_model_and_tokenizer(
    model_path: str = "models/best_model.pt",
    vocab_path: str = "models/vocab.json",
    device: str = "cpu",
) -> Tuple[TransformerLM, SimpleTokenizer]:
    """
    Loads saved Transformer model checkpoint and vocabulary mapping.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    # Restore tokenizer vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.load_vocab(vocab_path)

    # Load model checkpoint and rebuild architecture
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer


def predict_next_token_candidates(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    top_k: int = 5,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Computes probability distribution for the next token given an input prompt.
    Returns the top-K candidate tokens along with their percentage probabilities.
    """
    model.eval()
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if not token_ids:
        # Fallback to SOS or default token if empty prompt
        token_ids = [tokenizer.sos_id]

    # Truncate to maximum context length if prompt is too long
    if len(token_ids) > model.context_length:
        token_ids = token_ids[-model.context_length :]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model(input_tensor, causal=True)
        # Select logits for the final token in the prompt sequence
        last_token_logits = logits[0, -1, :]
        probs = F.softmax(last_token_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    candidates = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_word = tokenizer.idx2word.get(idx, tokenizer.UNK_TOKEN)
        candidates.append({
            "token": token_word,
            "token_id": idx,
            "probability": prob,
            "percentage": f"{prob * 100:.2f}%",
        })

    return candidates


def generate_text(
    model: TransformerLM,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 5,
    sample: bool = False,
    repetition_penalty: float = 1.2,
    device: str = "cpu",
) -> str:
    """
    Generates text autoregressively starting from an input prompt.

    Args:
        model: Trained TransformerLM instance
        tokenizer: SimpleTokenizer instance
        prompt: Initial text string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        top_k: Limits sampling to top_k candidate tokens
        sample: If False, uses greedy decoding; if True, uses temperature sampling
        repetition_penalty: Penalizes already generated tokens to prevent repetitive loops (> 1.0)
    """
    model.eval()
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if not token_ids:
        token_ids = [tokenizer.sos_id]

    for _ in range(max_new_tokens):
        # Keep sequence within context window
        context_ids = token_ids[-model.context_length :]
        input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _ = model(input_tensor, causal=True)
            last_logits = logits[0, -1, :].clone()

            # Apply repetition penalty to recently generated tokens
            if repetition_penalty > 1.0:
                for past_id in set(token_ids[-8:]):
                    if past_id not in (tokenizer.pad_id, tokenizer.unk_id):
                        if last_logits[past_id] > 0:
                            last_logits[past_id] /= repetition_penalty
                        else:
                            last_logits[past_id] *= repetition_penalty

            if not sample or temperature <= 0.01:
                # Greedy Decoding: pick argmax
                next_id = int(torch.argmax(last_logits).item())
            else:
                # Temperature scaling
                scaled_logits = last_logits / max(temperature, 1e-4)

                # Optional Top-K filtering
                if top_k > 0:
                    top_vals, top_idx = torch.topk(scaled_logits, min(top_k, len(scaled_logits)))
                    masked_logits = torch.full_like(scaled_logits, float("-inf"))
                    masked_logits.scatter_(0, top_idx, top_vals)
                    probs = F.softmax(masked_logits, dim=-1)
                else:
                    probs = F.softmax(scaled_logits, dim=-1)

                next_id = int(torch.multinomial(probs, num_samples=1).item())

        # If model outputs End-Of-Sequence token, stop generation
        if next_id == tokenizer.eos_id:
            break

        token_ids.append(next_id)

    return tokenizer.decode(token_ids, skip_special=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained Transformer")
    parser.add_argument("--prompt", type=str, default="machine learning is", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=15, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(project_root, "models", "best_model.pt")
    vocab_file = os.path.join(project_root, "models", "vocab.json")

    if not os.path.exists(model_file):
        print(f"Error: Model weights not found at {model_file}. Run train.py first!")
        exit(1)

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_file, vocab_file)

    print("\n" + "=" * 60)
    print(f"Prompt: '{args.prompt}'")
    print("=" * 60)

    # Top-5 Next-Token Candidates
    candidates = predict_next_token_candidates(loaded_model, loaded_tokenizer, args.prompt, top_k=5)
    print("\nTop-5 Next Token Predictions:")
    for rank, c in enumerate(candidates, start=1):
        print(f"  {rank}. '{c['token']:<15}' | Probability: {c['percentage']}")

    # Full text continuation
    sampled_text = generate_text(
        loaded_model,
        loaded_tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        sample=not args.greedy,
    )
    print("\nGenerated Continuation:")
    print(f"'{sampled_text}'")
    print("=" * 60)
