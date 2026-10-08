"""
CharanAI - Attention Weights Visualization Tool
===============================================
Inspects and visualizes the attention weight matrices learned by CharanLM.
Renders ASCII heatmaps showing which tokens each attention head focuses on.
"""
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.charan_lm import CharanLM
from tokenizer.bpe_tokenizer import BPETokenizer

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "charan_lm.pth")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "bpe_tokenizer.json")


def visualize_attention(prompt: str = "I am building my own language model."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BPETokenizer(TOKENIZER_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    cfg = checkpoint.get("config", {
        "vocab_size": 150,
        "max_sequence_length": 32,
        "embedding_dim": 64,
        "num_heads": 4,
        "hidden_dim": 256,
        "num_layers": 2
    })
    state_dict = checkpoint.get("model_state_dict", checkpoint)

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

    token_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.id_to_token.get(t, "<UNK>") for t in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    print("=" * 80)
    print("CharanAI — Multi-Head Attention Inspection")
    print("=" * 80)
    print(f"Prompt: \"{prompt}\"")
    print(f"Tokens ({len(tokens)}): {tokens}\n")

    # Forward pass up to Layer 0 attention
    with torch.no_grad():
        x = model.embeddings(input_ids)
        # Layer 0 Attention weights
        block0 = model.transformer_blocks[0]
        norm_x = block0.norm1(x)
        _, attn_weights_l0 = block0.attention(norm_x, return_attention_weights=True)

    # Render ASCII attention heatmaps for all 4 heads
    # attn_weights_l0 shape: (1, num_heads, T, T)
    weights = attn_weights_l0[0].cpu()  # (num_heads, T, T)
    col_label = "Query \\ Key"

    for head in range(cfg["num_heads"]):
        print(f"\n--- Layer 0 | Head {head + 1} Attention Matrix ---")
        head_w = weights[head]

        # Column headers (Keys)
        col_headers = " ".join(f"{t[:5]:>5}" for t in tokens)
        header = f"{col_label:<12} | {col_headers}"
        print(header)
        print("-" * len(header))

        for i, q_tok in enumerate(tokens):
            row_str = f"{q_tok[:10]:<12} | "
            for j in range(len(tokens)):
                w = head_w[i, j].item()
                if w < 0.01:
                    row_str += "    . "
                else:
                    row_str += f"{w:5.2f} "
            print(row_str)

    print("\n" + "=" * 80)
    print("Attention Interpretation:")
    print(" - Each row represents a Query token deciding how much to attend to past Key tokens.")
    print(" - Values sum to 1.0 horizontally (due to Softmax across Keys).")
    print(" - Values above the main diagonal are '.' (0.00) because of the Causal Mask.")
    print("=" * 80)


if __name__ == "__main__":
    visualize_attention()
