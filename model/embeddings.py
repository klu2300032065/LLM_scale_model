"""
CharanAI - Combined Embedding Layer
===================================
Combines:
 1. Token Embeddings (Discrete Token ID -> Continuous Representation)
 2. Positional Embeddings (Learned or Sinusoidal Position Vector)
 3. Embedding Dropout (Regularization to prevent token memorization)
"""
from typing import Optional
import torch
import torch.nn as nn

try:
    from model.positional_embedding import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
except ImportError:
    from positional_embedding import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding


class TokenEmbedding(nn.Module):
    """
    Standard Token Embedding lookup table.
    Maps discrete integer token IDs (0..V-1) to vectors of dimension embedding_dim.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, T, embedding_dim)
        return self.embedding(token_ids)


class CharanEmbeddings(nn.Module):
    """
    Unified Embedding module for CharanLM.
    Computes:
        x = Dropout(TokenEmbedding(tokens) + PositionalEmbedding(positions))
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
        position_type: str = "learned",
        dropout: float = 0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim

        # 1. Token Embedding
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)

        # 2. Positional Embedding (Learned or Sinusoidal)
        if position_type == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEmbedding(
                max_sequence_length,
                embedding_dim
            )
        else:
            self.position_embedding = LearnedPositionalEmbedding(
                max_sequence_length,
                embedding_dim
            )

        # 3. Dropout Regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
        Returns:
            Combined embedding tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds maximum ({self.max_sequence_length})"
            )

        # 1. Token Embeddings: (B, T, d)
        tok_emb = self.token_embedding(input_ids)

        # 2. Position IDs: (T,)
        positions = torch.arange(sequence_length, device=input_ids.device)

        # 3. Positional Embeddings: (1, T, d) or (B, T, d)
        pos_emb = self.position_embedding(positions)

        # 4. Element-wise Sum + Dropout
        x = tok_emb + pos_emb
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    vocab_size = 150
    seq_len = 8
    d_model = 64
    max_len = 32

    # Instantiate unified embedding layer
    embedding_layer = CharanEmbeddings(
        vocab_size=vocab_size,
        max_sequence_length=max_len,
        embedding_dim=d_model,
        position_type="learned",
        dropout=0.1
    )

    sample_ids = torch.tensor([[10, 45, 88, 12, 1, 99, 14, 7]])
    out = embedding_layer(sample_ids)

    print("CharanEmbeddings Output:")
    print(" - Input Token IDs shape: ", sample_ids.shape)
    print(" - Output Embedding shape:  ", out.shape)
    print(" - Parameter Count:         ", sum(p.numel() for p in embedding_layer.parameters()))
