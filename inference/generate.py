"""
CharanAI - Text Generation & Inference
======================================
Loads the trained CharanLM checkpoint and BPETokenizer to perform
autoregressive text generation from text prompts.
"""
import os
import sys
from typing import Optional
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.charan_lm import CharanLM
from tokenizer.bpe_tokenizer import BPETokenizer

# File paths
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "bpe_tokenizer.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "charan_lm.pth")


def load_model_and_tokenizer(checkpoint_path: str = CHECKPOINT_PATH, tokenizer_path: str = TOKENIZER_PATH):
    """Loads tokenizer and CharanLM model with checkpoint weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    tokenizer = BPETokenizer(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        cfg = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Fallback for raw state_dict checkpoints
        cfg = {
            "vocab_size": tokenizer.vocab_size,
            "max_sequence_length": 32,
            "embedding_dim": 64,
            "num_heads": 4,
            "hidden_dim": 256,
            "num_layers": 2
        }
        state_dict = checkpoint

    model = CharanLM(
        vocab_size=cfg["vocab_size"],
        max_sequence_length=cfg["max_sequence_length"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"]
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, device, cfg["max_sequence_length"]


def generate(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    tokenizer_path: str = TOKENIZER_PATH
) -> str:
    """
    Autoregressively generates continuation text for a given prompt.

    Args:
        prompt: Initial input string.
        max_new_tokens: Number of tokens to predict and append.
        temperature: Temperature scaling for sampling logits (1.0 = standard, 0.0 = greedy).
        top_k: If set, restricts sampling to top-k highest probability tokens.
    """
    model, tokenizer, device, max_seq_len = load_model_and_tokenizer(checkpoint_path, tokenizer_path)

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id.get("<BOS>", 2)]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context to model's maximum context window
        context_ids = input_ids[:, -max_seq_len:]

        # Forward pass to obtain logits for all positions
        with torch.no_grad():
            logits = model(context_ids)  # (1, T, vocab_size)

        # Extract logits corresponding to the very last token
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # Token selection: Greedy (ArgMax) vs Temperature Sampling
        if temperature <= 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        # Append predicted token to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    prompts = [
        "I am",
        "Artificial intelligence",
        "A transformer",
        "Python is",
        "Machine learning"
    ]

    print("=" * 60)
    print("CharanAI — Autoregressive Text Generation Demo")
    print("=" * 60)

    for prompt in prompts:
        output_greedy = generate(prompt, max_new_tokens=15, temperature=0.0)
        print(f"\nPrompt:    '{prompt}'")
        print(f"Generated: '{output_greedy}'")

    print("\n" + "=" * 60)
