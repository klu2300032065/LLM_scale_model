"""
CharanAI - BPETokenizer Loader Compatibility Wrapper
===================================================
Exports BPETokenizer from tokenizer.bpe_tokenizer to maintain
a single source of truth across the entire codebase.
"""
from tokenizer.bpe_tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]


if __name__ == "__main__":
    tokenizer = BPETokenizer("tokenizer/bpe_tokenizer.json")
    text = "Artificial intelligence can automate tasks."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Original text:", text)
    print("Encoded IDs:", encoded)
    print("Decoded text:", decoded)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Match?:", text == decoded)
