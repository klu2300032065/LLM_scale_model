"""
CharanAI - Dataset Builder
==========================
Processes raw text from data/raw/training.txt, fits the BPETokenizer,
encodes the corpus into integer token IDs, and saves data/processed/token_ids.pt.
"""
import os
import sys
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer.bpe_tokenizer import BPETokenizer

RAW_TEXT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "training.txt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "bpe_tokenizer.json")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "token_ids.pt")
VOCAB_SIZE = 150


def build_dataset():
    print("=" * 50)
    print("CharanAI — Building Processed Dataset")
    print("=" * 50)

    # 1. Load raw text
    if not os.path.exists(RAW_TEXT_PATH):
        raise FileNotFoundError(f"Raw text file not found at: {RAW_TEXT_PATH}")

    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    print(f"Loaded raw training corpus: {len(raw_text)} characters, {len(raw_text.split())} words.")

    # 2. Train Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(raw_text, vocab_size=VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)

    # 3. Encode corpus
    token_ids = tokenizer.encode(raw_text)
    unk_id = tokenizer.token_to_id.get("<UNK>", 1)
    unk_count = token_ids.count(unk_id)

    print(f"\nEncoding Statistics:")
    print(f" - Total Token IDs generated: {len(token_ids)}")
    print(f" - Vocabulary Size: {tokenizer.vocab_size}")
    print(f" - <UNK> count: {unk_count} ({(unk_count / len(token_ids) * 100):.2f}%)")
    print(f" - Compression Ratio: {len(raw_text) / len(token_ids):.2f} chars/token")

    # 4. Save processed tensor
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    tensor = torch.tensor(token_ids, dtype=torch.long)
    torch.save(tensor, PROCESSED_DATA_PATH)

    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"First 20 token IDs: {token_ids[:20]}")
    print("=" * 50)


if __name__ == "__main__":
    build_dataset()
