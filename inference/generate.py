"""
CharanAI - Text Generation & Inference
======================================
Loads the trained CharanGPT checkpoint and BPETokenizer to perform
autoregressive text generation from text prompts.
"""
import os
import sys
from typing import Optional
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.charan_gpt import CharanGPT, CharanGPTConfig
from tokenizer.bpe_tokenizer import BPETokenizer

TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "bpe_tokenizer.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "charan_lm.pth")


def load_model_and_tokenizer(checkpoint_path: str = CHECKPOINT_PATH, tokenizer_path: str = TOKENIZER_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    tokenizer = BPETokenizer(tokenizer_path)
    model, _ = CharanGPT.from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    checkpoint_path: str = CHECKPOINT_PATH,
    tokenizer_path: str = TOKENIZER_PATH
) -> str:
    """
    Autoregressively generates continuation text for a given prompt string.
    """
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, tokenizer_path)

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.token_to_id.get("<BOS>", 2)]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Use CharanGPT's built-in generation method
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )[0].tolist()

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
    print("CharanAI — CharanGPT Text Generation Demo")
    print("=" * 60)

    for prompt in prompts:
        output_greedy = generate(prompt, max_new_tokens=15, temperature=0.0)
        print(f"\nPrompt:    '{prompt}'")
        print(f"Generated: '{output_greedy}'")

    print("\n" + "=" * 60)
