"""
CharanAI - Transformer Decoder Block
====================================
Combines:
 1. Normalization (LayerNorm or RMSNorm)
 2. Multi-Head Causal Self-Attention
 3. Residual Connection (Skip connection)
 4. Normalization (LayerNorm or RMSNorm)
 5. Position-wise Feed-Forward Network (GELU / ReLU / SwiGLU)
 6. Residual Connection (Skip connection)

Architecture Designs:
 - Pre-LN (Standard in GPT-2, GPT-3, LLaMA, CharanAI):
     x = x + Attn(Norm(x))
     x = x + MLP(Norm(x))
   Provides an unobstructed identity path for backpropagating gradients without vanishing.

 - Post-LN (Original 2017 "Attention Is All You Need"):
     x = Norm(x + Attn(x))
     x = Norm(x + MLP(x))
"""
from typing import Optional
import torch
import torch.nn as nn

try:
    from model.multi_head_attention import MultiHeadAttention
    from model.feed_forward import FeedForward
except ImportError:
    from multi_head_attention import MultiHeadAttention
    from feed_forward import FeedForward


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Used in modern LLMs (LLaMA, Mistral, Gemma) to save ~7% computation
    by assuming zero-mean activations and normalizing by root-mean-square.

    Formula:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * gamma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean square along the last dimension
        mean_square = torch.mean(x ** 2, dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_square + self.eps)
        return x * rms * self.weight


class TransformerBlock(nn.Module):
    """
    Modular Transformer Decoder Block supporting Pre-LN/Post-LN and LayerNorm/RMSNorm.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        prenorm: bool = True,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0
    ):
        super().__init__()

        self.prenorm = prenorm

        # 1. Multi-Head Attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout
        )

        # 2. Feed-Forward Network
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            dropout=resid_dropout
        )

        # 3. Normalization Layers (LayerNorm vs RMSNorm)
        if norm_type.lower() == "rmsnorm":
            self.norm1 = RMSNorm(embedding_dim)
            self.norm2 = RMSNorm(embedding_dim)
        else:
            self.norm1 = nn.LayerNorm(embedding_dim)
            self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            # Pre-LN: Clean residual gradient highway (x + SubLayer(Norm(x)))
            x = x + self.attention(self.norm1(x))
            x = x + self.feed_forward(self.norm2(x))
        else:
            # Post-LN: Norm applied after addition (Norm(x + SubLayer(x)))
            x = self.norm1(x + self.attention(x))
            x = self.norm2(x + self.feed_forward(x))

        return x


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8
    embedding_dim = 64
    num_heads = 4
    hidden_dim = 256

    x = torch.randn(batch_size, sequence_length, embedding_dim)

    pre_ln_block = TransformerBlock(embedding_dim, num_heads, hidden_dim, norm_type="layernorm", prenorm=True)
    rmsnorm_block = TransformerBlock(embedding_dim, num_heads, hidden_dim, norm_type="rmsnorm", prenorm=True)
    post_ln_block = TransformerBlock(embedding_dim, num_heads, hidden_dim, norm_type="layernorm", prenorm=False)

    out_pre = pre_ln_block(x)
    out_rms = rmsnorm_block(x)
    out_post = post_ln_block(x)

    print("Transformer Block Architecture Tests:")
    print(" - Pre-LN LayerNorm Output: ", out_pre.shape)
    print(" - Pre-LN RMSNorm Output:   ", out_rms.shape)
    print(" - Post-LN LayerNorm Output:", out_post.shape)
