"""
CharanAI - Single-Head Causal Self-Attention
============================================
Implements scaled dot-product causal self-attention from scratch.

Mathematical formulation:
  1. Projections:   Q = X W_q,  K = X W_k,  V = X W_v
  2. Raw Scores:    S = (Q @ K^T) / sqrt(d_k)
  3. Causal Mask:   S_ij = -inf for all j > i (future tokens)
  4. Probabilities: A = softmax(S, dim=-1)
  5. Weighted Sum:  Output = A @ V
"""
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Single-Head Causal Scaled Dot-Product Self-Attention.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Query, Key, and Value linear projections
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Attention dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            return_attention_weights: If True, returns (output, attention_weights).
        Returns:
            output: Attention-weighted values of shape (batch_size, sequence_length, embedding_dim).
            attention_weights (optional): Softmax attention matrix of shape (batch_size, sequence_length, sequence_length).
        """
        batch_size, sequence_length, _ = x.shape

        # 1. Project input into Query, Key, and Value spaces
        Q = self.query(x)  # (B, T, d)
        K = self.key(x)    # (B, T, d)
        V = self.value(x)  # (B, T, d)

        # 2. Compute Scaled Dot-Product Attention Scores
        # (B, T, d) @ (B, d, T) -> (B, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.embedding_dim)

        # 3. Apply Causal (Autoregressive) Mask
        # Upper triangle (j > i) set to -infinity so future tokens receive 0 attention weight
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 4. Softmax across key dimension to get attention probabilities
        attention_weights = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_weights)

        # 5. Compute weighted sum of values: (B, T, T) @ (B, T, d) -> (B, T, d)
        output = torch.matmul(attention_probs, V)

        if return_attention_weights:
            return output, attention_weights
        return output


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 4
    embedding_dim = 16

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    attn = SelfAttention(embedding_dim)

    out, weights = attn(x, return_attention_weights=True)

    print("Single-Head Self-Attention Test:")
    print(" - Input Shape: ", x.shape)
    print(" - Output Shape:", out.shape)
    print(" - Attention Weights Shape (Batch 0):\n", weights[0].detach().round(decimals=3))
    print(" - Rows sum to 1.0:", weights[0].sum(dim=-1).detach().round(decimals=3))
