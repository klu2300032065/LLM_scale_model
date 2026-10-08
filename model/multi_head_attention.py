"""
CharanAI - Multi-Head Causal Self-Attention
===========================================
Splits the embedding dimension into `num_heads` parallel attention heads,
allowing the model to jointly attend to information from different representation subspaces.

Dimensions:
 - Input:             (Batch, Time, d_model)
 - Q, K, V:           (Batch, Time, num_heads, head_dim) -> (Batch, num_heads, Time, head_dim)
 - Attention Scores:  (Batch, num_heads, Time, Time)
 - Concatenated Out:  (Batch, Time, d_model)
 - Linear Projection: (Batch, Time, d_model)
"""
import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, (
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Output projection combining all heads
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.resid_dropout = nn.Dropout(p=resid_dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim).
            return_attention_weights: If True, also returns attention weight matrix of shape (B, num_heads, T, T).
        """
        batch_size, sequence_length, _ = x.shape

        # 1. Project inputs to Q, K, V: (B, T, d_model)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. Reshape into multi-heads: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = Q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention: (B, h, T, head_dim) @ (B, h, head_dim, T) -> (B, h, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. Causal Mask: Zero out future positions
        causal_mask = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # 5. Attention Probabilities
        weights = torch.softmax(scores, dim=-1)
        weights_dropped = self.attn_dropout(weights)

        # 6. Weighted Sum: (B, h, T, T) @ (B, h, T, head_dim) -> (B, h, T, head_dim)
        attention_out = torch.matmul(weights_dropped, V)

        # 7. Concatenate heads back to (B, T, d_model)
        attention_out = attention_out.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.embedding_dim
        )

        # 8. Output projection + Residual Dropout
        output = self.resid_dropout(self.output_projection(attention_out))

        if return_attention_weights:
            return output, weights
        return output


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 6
    embedding_dim = 64
    num_heads = 4

    x = torch.randn(batch_size, sequence_length, embedding_dim)
    mha = MultiHeadAttention(embedding_dim, num_heads)

    out, weights = mha(x, return_attention_weights=True)

    print("Multi-Head Attention Test:")
    print(" - Input Shape: ", x.shape)
    print(" - Head Dimension:", embedding_dim // num_heads)
    print(" - Output Shape:", out.shape)
    print(" - Multi-Head Attention Weights Shape:", weights.shape)
