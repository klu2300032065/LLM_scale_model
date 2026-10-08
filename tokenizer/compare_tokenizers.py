"""
CharanAI - Tokenizer Comparison Tool
====================================
Educational benchmark comparing:
 1. Character-Level Tokenizer (CharTokenizer)
 2. Word-Level Tokenizer (WordTokenizer)
 3. Subword Byte-Pair Encoding Tokenizer (BPETokenizer)

Analyzes vocabulary sizes, sequence lengths, compression ratios, and out-of-vocabulary handling.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer.tokenizer import CharTokenizer
from tokenizer.word_tokenizer import WordTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

RAW_TEXT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "training.txt")


def compare_tokenizers():
    with open(RAW_TEXT_PATH, "r", encoding="utf-8") as f:
        corpus = f.read()

    print("=" * 75)
    print("CharanAI — Tokenization Strategy Comparison")
    print("=" * 75)
    print(f"Training Corpus Size: {len(corpus)} characters | {len(corpus.split())} words\n")

    # 1. Train all three tokenizers
    char_tok = CharTokenizer()
    char_tok.train(corpus)

    word_tok = WordTokenizer()
    word_tok.train(corpus, max_vocab_size=500)

    bpe_tok = BPETokenizer()
    bpe_tok.train(corpus, vocab_size=150)

    # 2. Benchmark on Test Sentences
    test_sentences = [
        "I am learning how to build an AI system from scratch.",
        "Deep learning uses neural networks to learn representations.",
        "Unseen words like supercomputer and microarchitecture cause out-of-vocabulary challenges."
    ]

    print("\n" + "=" * 75)
    print(f"{'Tokenizer':<18} | {'Vocab Size':<10} | {'Tokens':<8} | {'Chars/Token':<12} | {'UNK Count'}")
    print("-" * 75)

    for sentence in test_sentences:
        print(f"\nEvaluating Sentence: \"{sentence}\" ({len(sentence)} chars, {len(sentence.split())} words)")
        print("-" * 75)

        for name, tok in [("Character", char_tok), ("Word", word_tok), ("Subword (BPE)", bpe_tok)]:
            encoded = tok.encode(sentence)
            unk_id = tok.token_to_id.get("<UNK>", 1)
            unk_count = encoded.count(unk_id)
            ratio = len(sentence) / max(len(encoded), 1)

            print(f"{name:<18} | {len(tok.token_to_id):<10} | {len(encoded):<8} | {ratio:<12.2f} | {unk_count}")

    print("=" * 75)
    print("\nKey Insights for LLM Engineering:")
    print("1. Character-level has tiny vocab, but sequence length is huge (high transformer self-attention cost O(T^2)).")
    print("2. Word-level has short sequence length, but fails on new/unseen words (produces <UNK>).")
    print("3. Subword (BPE) balances sequence length and vocabulary, splitting unknown words into known subwords with 0 <UNK>.")
    print("=" * 75)


if __name__ == "__main__":
    compare_tokenizers()
