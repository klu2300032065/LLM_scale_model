"""
CharanAI - Position-Wise Feed-Forward Network (MLP)
===================================================
Applies a two-layer non-linear transformation to each token vector independently.

Roles in Transformer:
 - Multi-Head Attention: Communicates across tokens in the sequence dimension (T x T).
 - Feed-Forward Network: Enriches factual and semantic knowledge within each token vector (d_model x hidden_dim).

Supported Activation Functions:
 1. GELU (Gaussian Error Linear Unit): Standard in GPT-2, GPT-3, BERT, and CharanAI.
 2. ReLU (Rectified Linear Unit): Standard in original 2017 Transformer.
 3. SwiGLU (Swish Gated Linear Unit): Used in modern state-of-the-art LLMs (LLaMA, Mistral, Gemma).
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU).
    Formula: SwiGLU(x) = (x W_gate * sigmoid(x W_gate)) * (x W_up) * W_down
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(embedding_dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, embedding_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class FeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (MLP).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        # Default hidden_dim is 4 * embedding_dim (standard in GPT architectures)
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation.lower()

        if self.activation_type == "swiglu":
            # SwiGLU replaces standard 2-layer MLP with 3 linear matrices
            # Standard hidden_dim for SwiGLU is (2/3) * 4 * embedding_dim to preserve parameter budget
            swiglu_hidden = int(2 * hidden_dim / 3)
            self.network = SwiGLU(embedding_dim, swiglu_hidden, bias=bias)
            self.dropout = nn.Dropout(p=dropout)
        else:
            if self.activation_type == "relu":
                act_layer = nn.ReLU()
            elif self.activation_type == "gelu":
                act_layer = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}. Choose 'gelu', 'relu', or 'swiglu'.")

            self.network = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim, bias=bias),
                act_layer,
                nn.Linear(hidden_dim, embedding_dim, bias=bias),
                nn.Dropout(p=dropout)
            )
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        if self.activation_type == "swiglu":
            return self.dropout(self.network(x))
        return self.network(x)


if __name__ == "__main__":
    B, T, D, H = 2, 4, 64, 256
    x = torch.randn(B, T, D)

    ffn_gelu = FeedForward(D, H, activation="gelu")
    ffn_relu = FeedForward(D, H, activation="relu")
    ffn_swiglu = FeedForward(D, H, activation="swiglu")

    out_gelu = ffn_gelu(x)
    out_relu = ffn_relu(x)
    out_swiglu = ffn_swiglu(x)

    print("FeedForward Activation Suite:")
    print(" - Input Shape:   ", x.shape)
    print(" - GELU Output:   ", out_gelu.shape, f"({sum(p.numel() for p in ffn_gelu.parameters())} params)")
    print(" - ReLU Output:   ", out_relu.shape, f"({sum(p.numel() for p in ffn_relu.parameters())} params)")
    print(" - SwiGLU Output: ", out_swiglu.shape, f"({sum(p.numel() for p in ffn_swiglu.parameters())} params)")
