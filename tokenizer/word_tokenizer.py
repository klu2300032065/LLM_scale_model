"""
CharanAI - Word-Level Tokenizer
===============================
A word-level tokenizer that tokenizes text into whole words and punctuation marks.
Used to demonstrate the trade-offs of Word-Level vs Character-Level vs Subword (BPE) tokenization.
"""
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Union


class WordTokenizer:
    """
    Word-level Tokenizer with configurable vocabulary size and unknown token handling.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self, tokenizer_path: Optional[str] = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.vocab_size: int = 0

        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load(tokenizer_path)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            self.vocab_size = len(self.token_to_id)
            return idx
        return self.token_to_id[token]

    def _tokenize_raw(self, text: str) -> List[str]:
        """Splits text into words and punctuation tokens, ignoring pure whitespace."""
        # Matches word sequences or individual non-word characters (punctuation)
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def train(self, text: str, max_vocab_size: int = 1000, min_frequency: int = 1):
        """
        Builds vocabulary from text keeping the top most frequent words.
        """
        self.token_to_id = {}
        self.id_to_token = {}

        # 1. Add Special Tokens
        for st in self.SPECIAL_TOKENS:
            self._add_token(st)

        # 2. Count Word Frequencies
        tokens = self._tokenize_raw(text)
        counts = Counter(tokens)

        # 3. Add most frequent words
        for word, freq in counts.most_common():
            if freq < min_frequency:
                break
            if len(self.token_to_id) >= max_vocab_size:
                break
            self._add_token(word)

        print(f"WordTokenizer Training Complete:")
        print(f" - Total Words Scanned: {len(tokens)}")
        print(f" - Unique Words Found: {len(counts)}")
        print(f" - Final Vocabulary Size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Converts text string into word token IDs."""
        tokens = self._tokenize_raw(text)
        unk_id = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: List[Union[int, str]]) -> str:
        """Converts token IDs back into human readable text."""
        words = []
        for tid in token_ids:
            tid_int = int(tid)
            token = self.id_to_token.get(tid_int, "<UNK>")
            if token not in ["<PAD>", "<BOS>", "<EOS>"]:
                words.append(token)

        # Reconstruct sentence with basic punctuation spacing rules
        text = " ".join(words)
        # Clean up spaces before punctuation marks (e.g. "word ." -> "word.")
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"WordTokenizer saved to: {filepath}")

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(idx): token for token, idx in self.token_to_id.items()
        }
        self.vocab_size = len(self.token_to_id)
        return self


if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming software engineering.
    Building a language model from scratch requires understanding tokenization.
    """

    tok = WordTokenizer()
    tok.train(sample_text)

    test = "Building a language model from scratch."
    enc = tok.encode(test)
    dec = tok.decode(enc)

    print("\nTest:", test)
    print("Encoded IDs:", enc)
    print("Decoded Text:", dec)
