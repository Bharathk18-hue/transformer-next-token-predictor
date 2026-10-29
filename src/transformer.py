import math
import os
import sys
from typing import Tuple, Optional, List
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    Expands dimension by 4x, applies non-linearity, and projects back.
    """

    def __init__(self, d_model: int = 64, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block featuring Pre-LayerNorm architecture:
      x = x + MultiHeadAttention(LayerNorm(x))
      x = x + FeedForward(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dropout=dropout)

    def forward(
        self, x: torch.Tensor, causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sub-layer 1: Causal Multi-Head Self-Attention with residual connection
        normed_x = self.ln1(x)
        attn_out, weights = self.attn(normed_x, causal=causal)
        x = x + attn_out

        # Sub-layer 2: Feed-Forward Network with residual connection
        x = x + self.ffn(self.ln2(x))

        return x, weights


class TransformerLM(nn.Module):
    """
    Complete small Transformer Language Model for Next-Token Prediction:
      1. Token Embedding & Positional Embedding
      2. Stack of Transformer Decoder Blocks
      3. Final Layer Normalization
      4. Linear Projection to Vocabulary Logits
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 32,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Embedding layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer decoder layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm & Unembedding Head
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output projection
        self.lm_head.weight = self.token_embeddings.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights using standard normal distribution for stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            causal: Whether to apply causal mask in attention
            return_attention: If True, returns attention weight tensors from all layers

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            attentions: List of attention tensors per layer (optional)
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum context length {self.context_length}"
            )

        # Generate positional indices [0, 1, ..., seq_len - 1]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        # Combine token embeddings and positional embeddings
        tok_emb = self.token_embeddings(input_ids)      # [batch, seq_len, d_model]
        pos_emb = self.pos_embeddings(positions)        # [seq_len, d_model]
        x = self.dropout(tok_emb + pos_emb)

        # Pass through stacked Transformer Blocks
        attn_list: List[torch.Tensor] = []
        for block in self.blocks:
            x, weights = block(x, causal=causal)
            if return_attention:
                attn_list.append(weights)

        # Apply final LayerNorm and project to vocabulary size
        x = self.ln_final(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, (attn_list if return_attention else None)

    def count_parameters(self) -> int:
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_size = 200
    context_length = 16
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )
    dummy_input = torch.randint(0, vocab_size, (2, 8))
    logits, attns = model(dummy_input, return_attention=True)

    print("TransformerLM Architecture Verification:")
    print(f"Total Trainable Parameters: {model.count_parameters():,}")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {logits.shape} -> [batch, seq_len, vocab_size]")
    print(f"Number of attention layers: {len(attns)}")
    print(f"Layer 0 attention shape   : {attns[0].shape}")
