"""
CharanAI - Positional Embeddings
================================
Provides both:
 1. LearnedPositionalEmbedding: Standard lookup table optimized via backpropagation (used in GPT-2 / CharanAI).
 2. SinusoidalPositionalEmbedding: Fixed trigonometric frequencies (from "Attention Is All You Need", Vaswani et al.).
"""
import math
from typing import Optional
import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding (GPT-2 style).
    Maps integer positions (0..T-1) to d-dimensional learned vectors.
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_sequence_length, embedding_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: 1D Tensor (T,) or 2D Tensor (B, T) of position indices.
        Returns:
            Positional embedding tensor of shape (B, T, embedding_dim) or (1, T, embedding_dim).
        """
        if positions.dim() == 1:
            # (T,) -> (1, T, d)
            return self.embedding(positions).unsqueeze(0)
        # (B, T) -> (B, T, d)
        return self.embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed Sinusoidal Positional Embedding (Vaswani et al., 2017).

    Formulas:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # Compute the positional encoding matrix once in log space
        pe = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as persistent buffer (saved with state_dict, not a trainable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: 1D Tensor (T,) of sequential position indices [0..T-1].
        Returns:
            Positional embedding tensor of shape (1, T, embedding_dim).
        """
        seq_len = positions.size(-1)
        return self.pe[:, :seq_len, :]


# Backward compatibility alias
PositionalEmbedding = LearnedPositionalEmbedding


if __name__ == "__main__":
    seq_len = 8
    d_model = 64
    max_len = 128

    learned = LearnedPositionalEmbedding(max_len, d_model)
    sinusoidal = SinusoidalPositionalEmbedding(max_len, d_model)

    pos = torch.arange(seq_len)

    learned_out = learned(pos)
    sin_out = sinusoidal(pos)

    print("Positional Embeddings Test:")
    print(" - Input Positions:   ", pos.tolist())
    print(" - Learned Output:    ", learned_out.shape, f"({sum(p.numel() for p in learned.parameters())} params)")
    print(" - Sinusoidal Output: ", sin_out.shape, f"({sum(p.numel() for p in sinusoidal.parameters())} params)")

    # Dot-product similarity across positions in sinusoidal encoding
    sim_pos0_pos1 = torch.cosine_similarity(sin_out[0, 0], sin_out[0, 1], dim=0).item()
    sim_pos0_pos7 = torch.cosine_similarity(sin_out[0, 0], sin_out[0, 7], dim=0).item()
    print(f" - Cosine Similarity (pos 0 vs pos 1 - nearby): {sim_pos0_pos1:.4f}")
    print(f" - Cosine Similarity (pos 0 vs pos 7 - distant): {sim_pos0_pos7:.4f}")
