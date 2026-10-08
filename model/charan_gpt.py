"""
CharanAI - CharanGPT Model Architecture & Engine
================================================
A self-contained, production-grade GPT-style Transformer model class featuring:
 1. Clean configuration dataclass (`CharanGPTConfig`)
 2. GPT-2 style weight initialization with residual scaling
 3. Optional Weight Tying (shares TokenEmbedding with Linear LM Head)
 4. Unified model checkpointing API (`save_checkpoint`, `load_checkpoint`)
 5. Built-in autoregressive generation with Temperature & Top-K sampling
"""
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model.embeddings import CharanEmbeddings
    from model.transformer_block import TransformerBlock, RMSNorm
except ImportError:
    from embeddings import CharanEmbeddings
    from transformer_block import TransformerBlock, RMSNorm


@dataclass
class CharanGPTConfig:
    """Hyperparameter configuration for CharanGPT."""
    vocab_size: int = 150
    max_sequence_length: int = 64
    embedding_dim: int = 64
    num_heads: int = 4
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bias: bool = True
    weight_tying: bool = True
    norm_type: str = "layernorm"     # "layernorm" or "rmsnorm"
    activation: str = "gelu"         # "gelu", "relu", or "swiglu"
    position_type: str = "learned"   # "learned" or "sinusoidal"


class CharanGPT(nn.Module):
    """
    CharanGPT: Full-featured custom GPT-style Language Model.
    """

    def __init__(self, config: CharanGPTConfig):
        super().__init__()
        self.config = config

        # 1. Unified Embeddings
        self.embeddings = CharanEmbeddings(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            embedding_dim=config.embedding_dim,
            position_type=config.position_type,
            dropout=config.dropout
        )

        # 2. Transformer Decoder Stack
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                norm_type=config.norm_type,
                activation=config.activation,
                prenorm=True,
                attn_dropout=config.dropout,
                resid_dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # 3. Final Normalization
        if config.norm_type.lower() == "rmsnorm":
            self.final_norm = RMSNorm(config.embedding_dim)
        else:
            self.final_norm = nn.LayerNorm(config.embedding_dim)

        # 4. Language Model Head
        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False
        )

        # 5. Weight Tying (Weight Sharing)
        # Reuses the token embedding matrix as the output classification layer
        if config.weight_tying and config.position_type != "sinusoidal":
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

        # 6. Apply GPT-2 weight initialization
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (GPT-2 paper section 2.3)
        for pn, p in self.named_parameters():
            if pn.endswith("output_projection.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module):
        """Initializes linear and embedding weights with mean=0, std=0.02."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass computing logits and optionally Cross-Entropy loss.
        """
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) exceeds maximum ({self.config.max_sequence_length})"
            )

        # Embeddings
        x = self.embeddings(input_ids)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.final_norm(x)

        # Compute vocabulary logits: (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressively generates next token IDs given a prefix tensor (B, T).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context window to model's maximum context length
            context_ids = input_ids[:, -self.config.max_sequence_length:]

            # Forward pass
            logits, _ = self(context_ids)

            # Pluck logits at the last position
            next_token_logits = logits[:, -1, :]

            if temperature <= 0.0:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def save_checkpoint(
        self,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None
    ):
        """Saves weights, config, and training state to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "epoch": epoch,
            "loss": loss
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"CharanGPT checkpoint successfully saved to: {filepath}")

    @classmethod
    def from_checkpoint(cls, filepath: str, map_location: str = "cpu") -> Tuple["CharanGPT", Dict]:
        """Loads CharanGPT model instance from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location)
        if "config" in checkpoint:
            raw_cfg = checkpoint["config"]
            # Filter config keys matching dataclass
            valid_keys = CharanGPTConfig.__dataclass_fields__.keys()
            filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_keys}
            cfg = CharanGPTConfig(**filtered_cfg)
        else:
            cfg = CharanGPTConfig()

        model = cls(cfg)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        return model, checkpoint

    def count_parameters(self, non_embedding: bool = False) -> int:
        """
        Returns parameter count. If non_embedding=True, excludes token/position embeddings.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            emb_params = sum(p.numel() for p in self.embeddings.parameters() if p.requires_grad)
            return total - emb_params
        return total


if __name__ == "__main__":
    cfg = CharanGPTConfig(
        vocab_size=150,
        max_sequence_length=32,
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        weight_tying=True
    )

    model = CharanGPT(cfg)

    # Test forward pass with loss computation
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    logits, loss = model(x, targets=y)

    print("CharanGPT Engine Smoke Test:")
    print(" - Input Shape:         ", x.shape)
    print(" - Output Logits Shape: ", logits.shape)
    print(" - Calculated Loss:     ", loss.item())
    print(f" - Total Parameters:     {model.count_parameters():,}")
    print(f" - Non-Embedding Params: {model.count_parameters(non_embedding=True):,}")

    # Test built-in generation
    prefix = torch.tensor([[10, 25, 40]], dtype=torch.long)
    gen_out = model.generate(prefix, max_new_tokens=5, temperature=0.0)
    print(" - Autoregressive Output:", gen_out.tolist())
