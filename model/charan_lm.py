"""
CharanAI - CharanLM Model Architecture
======================================
Custom GPT-style Autoregressive Causal Language Model built from scratch.

Architecture flow:
    Token IDs (B, T)
        ↓
    CharanEmbeddings (Token Embedding + Positional Embedding + Dropout)
        ↓
    Stacked Transformer Blocks (Pre-LN, Multi-Head Attention, MLP)
        ↓
    Final Normalization (LayerNorm or RMSNorm)
        ↓
    Linear LM Head (Linear projection to vocabulary logits)
        ↓
    Logits (B, T, vocab_size)
"""
from typing import Dict, Optional
import torch
import torch.nn as nn

try:
    from model.embeddings import CharanEmbeddings
    from model.transformer_block import TransformerBlock, RMSNorm
except ImportError:
    from embeddings import CharanEmbeddings
    from transformer_block import TransformerBlock, RMSNorm


class CharanLM(nn.Module):
    """
    CharanLM: A small GPT-style Autoregressive Decoder-only Transformer.
    """

    def __init__(
        self,
        vocab_size: int = 150,
        max_sequence_length: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        position_type: str = "learned",
        norm_type: str = "layernorm",
        activation: str = "gelu"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.activation = activation

        # 1. Combined Embeddings (Token + Position + Dropout)
        self.embeddings = CharanEmbeddings(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            position_type=position_type,
            dropout=dropout
        )

        # 2. Stacked Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    norm_type=norm_type,
                    activation=activation,
                    prenorm=True,
                    attn_dropout=dropout,
                    resid_dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final Normalization
        if norm_type.lower() == "rmsnorm":
            self.final_norm = RMSNorm(embedding_dim)
        else:
            self.final_norm = nn.LayerNorm(embedding_dim)

        # 4. Language Model Head (maps hidden representation -> vocab logits)
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length) with token IDs.
        Returns:
            logits: Tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Embeddings: (B, T) -> (B, T, embedding_dim)
        x = self.embeddings(input_ids)

        # Pass through all Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Apply final Normalization
        x = self.final_norm(x)

        # Compute vocabulary logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        return logits

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Custom state_dict loader for backward compatibility with older checkpoints."""
        remapped = {}
        for k, v in state_dict.items():
            if k == "token_embedding.weight":
                remapped["embeddings.token_embedding.embedding.weight"] = v
            elif k == "position_embedding.weight":
                remapped["embeddings.position_embedding.embedding.weight"] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_size = 150
    max_sequence_length = 32
    embedding_dim = 64
    num_heads = 4
    hidden_dim = 256
    num_layers = 2

    model = CharanLM(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )

    sample_input = torch.randint(0, vocab_size, (2, 16))
    logits = model(sample_input)

    print("CharanLM Architecture Summary:")
    print(" - Input IDs shape: ", sample_input.shape)
    print(" - Output Logits shape:", logits.shape)
    print(f" - Total Trainable Parameters: {model.count_parameters():,}")
