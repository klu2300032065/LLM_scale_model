"""
CharanAI - Byte-Pair Encoding (BPE) Tokenizer
=============================================
A clean, educational implementation of Subword BPE.
Converts text into subword token IDs and back.
"""
import json
import os
import string
from collections import Counter
from typing import Dict, List, Optional, Union


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Subword Tokenizer.
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    END_OF_WORD = "</w>"

    def __init__(self, tokenizer_path: Optional[str] = None):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[List[str]] = []
        self.vocab_size: int = 0

        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load(tokenizer_path)

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.vocab_size = len(self.token_to_id)
            return token_id
        return self.token_to_id[token]

    def train(self, text: str, vocab_size: int = 150):
        self.token_to_id = {}
        self.id_to_token = {}
        self.merges = []

        # 1. Special tokens
        for token in self.SPECIAL_TOKENS:
            self._add_token(token)

        # 2. End-of-word marker
        self._add_token(self.END_OF_WORD)

        # 3. Printable ASCII characters + any characters in training text
        base_chars = set(string.printable) | set(text)
        for ch in sorted(base_chars):
            if not ch.isspace():
                self._add_token(ch)

        # 4. Words split into character sequences + end of word
        words = text.strip().split()
        token_sequences: List[List[str]] = [
            list(word) + [self.END_OF_WORD] for word in words
        ]

        # 5. Iteratively merge most frequent pairs
        while len(self.token_to_id) < vocab_size:
            pair_counts: Counter = Counter()
            for seq in token_sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]
            if count < 1:
                break

            merged_token = best_pair[0] + best_pair[1]
            if merged_token in self.token_to_id:
                break

            self._add_token(merged_token)
            self.merges.append(list(best_pair))

            # Apply merge to all sequences
            for idx, seq in enumerate(token_sequences):
                new_seq = []
                i = 0
                while i < len(seq):
                    if (
                        i < len(seq) - 1
                        and seq[i] == best_pair[0]
                        and seq[i + 1] == best_pair[1]
                    ):
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                token_sequences[idx] = new_seq

        print(f"BPE Training Complete: Vocab size = {len(self.token_to_id)}, Merges = {len(self.merges)}")

    def encode(self, text: str) -> List[int]:
        words = text.strip().split()
        all_tokens = []

        for word in words:
            tokens = list(word) + [self.END_OF_WORD]

            for merge in self.merges:
                first, second = merge[0], merge[1]
                merged = first + second
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == first
                        and tokens[i + 1] == second
                    ):
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            all_tokens.extend(tokens)

        unk_id = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(token, unk_id) for token in all_tokens]

    def decode(self, token_ids: List[Union[int, str]]) -> str:
        tokens = []
        for tid in token_ids:
            tid_int = int(tid)
            token = self.id_to_token.get(tid_int, "<UNK>")
            if token not in ["<PAD>", "<BOS>", "<EOS>"]:
                tokens.append(token)

        text = "".join(tokens)
        text = text.replace(self.END_OF_WORD, " ")
        return text.strip()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "vocab_size": len(self.token_to_id),
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer saved to: {filepath}")

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(idx): token for token, idx in self.token_to_id.items()
        }
        self.merges = data.get("merges", [])
        self.vocab_size = len(self.token_to_id)
        return self

    @classmethod
    def from_file(cls, filepath: str):
        tokenizer = cls()
        tokenizer.load(filepath)
        return tokenizer


if __name__ == "__main__":
    sample_corpus = """
Hello my name is Charan.
I am building my own language model.
Artificial intelligence can automate many tasks.
Language models learn patterns from text.
A transformer uses attention to understand relationships between tokens.
Machine learning learns useful patterns from data.
Python is a powerful programming language.
AI agents can plan and execute tasks.
A computer can process information very quickly.
Software systems can automate repetitive work.
Deep learning uses neural networks to learn representations.
Natural language processing allows computers to work with human language.
The future of artificial intelligence depends on useful and reliable systems.
I am learning how to build an AI system from scratch.
My goal is to create my own intelligent assistant.
"""
    tok = BPETokenizer()
    tok.train(sample_corpus, vocab_size=150)
    tok.save("tokenizer/bpe_tokenizer.json")

    test_sentence = "I am learning how to build an AI system from scratch."
    enc = tok.encode(test_sentence)
    dec = tok.decode(enc)

    print("\nTest Sentence:", test_sentence)
    print("Encoded:", enc)
    print("Decoded:", dec)
    print("Match?:", test_sentence == dec)
