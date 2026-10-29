import math
from typing import Tuple, Optional
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from scratch:
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + Mask ) * V

    Steps:
      1. Compute similarity scores: Q @ K^T
      2. Scale scores by sqrt(d_k) to prevent vanishing gradients
      3. Apply causal mask (optional) so token t cannot attend to token t+k
      4. Convert scores to probabilities via softmax
      5. Compute weighted combination of values: weights @ V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor  [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor    [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor  [batch_size, num_heads, seq_len, head_dim]
            causal: If True, applies lower-triangular causal mask

        Returns:
            output:       [batch_size, num_heads, seq_len, head_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        d_k = q.size(-1)
        seq_len = q.size(-2)

        # 1 & 2: Dot product similarity scaled by sqrt(d_k)
        # Shape: [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 3: Causal Masking (upper triangle filled with -inf)
        if causal and seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, -1e9)

        # 4: Softmax along last dimension (attention probabilities)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5: Weighted sum of Value vectors
        # Shape: [batch, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention.
    Splits the embedding space into multiple representation subspaces (heads).
    Each head attends independently to different contextual patterns.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            causal: Whether to apply causal masking

        Returns:
            out: Contextualized representations [batch_size, seq_len, d_model]
            weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project input to Q, K, V
        q = self.w_q(x)  # [batch, seq_len, d_model]
        k = self.w_k(x)  # [batch, seq_len, d_model]
        v = self.w_v(x)  # [batch, seq_len, d_model]

        # 2. Reshape into [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Compute scaled dot-product attention
        attn_out, weights = self.attention(q, k, v, causal=causal)

        # 4. Concatenate heads back into [batch, seq_len, d_model]
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # 5. Final linear projection
        out = self.dropout(self.w_o(attn_out))

        return out, weights


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 4

    test_input = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    output, weights = mha(test_input, causal=True)

    print("Multi-Head Attention Test:")
    print(f"Input shape : {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape (per head): {weights.shape}")
    print("\nAttention Matrix for Batch 0, Head 0 (Causal check):")
    print(weights[0, 0].detach().round(decimals=3))
