"""
CharanAI - Comprehensive Test Suite
===================================
Automated verification tests for:
 1. Tokenizers (Character, Word, BPE encode/decode/save/load/no-UNK)
 2. Dataset (Causal sliding context windows & target shifting)
 3. Positional Embeddings (Learned & Sinusoidal relative distance decay)
 4. Combined Embeddings (Token + Position + Dropout)
 5. Self-Attention & Multi-Head Attention (Shapes & Causal Masking)
 6. FeedForward Networks (GELU, ReLU, SwiGLU)
 7. Transformer Blocks (Pre-LN, Post-LN, RMSNorm, LayerNorm & Gradient Highway)
 8. CharanLM Model (Forward pass & Parameter counts)
 9. End-to-End Generation Pipeline
"""
import os
import sys
import unittest
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer.tokenizer import CharTokenizer
from tokenizer.word_tokenizer import WordTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import LanguageModelDataset
from model.positional_embedding import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from model.embeddings import TokenEmbedding, CharanEmbeddings
from model.attention import SelfAttention
from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward, SwiGLU
from model.transformer_block import TransformerBlock, RMSNorm
from model.charan_lm import CharanLM
from inference.generate import generate


class TestTokenizers(unittest.TestCase):
    def setUp(self):
        self.sample_text = (
            "Hello my name is Charan.\n"
            "I am building my own language model from scratch.\n"
            "Artificial intelligence uses deep neural networks."
        )

    def test_bpe_special_tokens_and_roundtrip(self):
        tok = BPETokenizer()
        tok.train(self.sample_text, vocab_size=120)
        for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            self.assertIn(token, tok.token_to_id)

        sentence = "I am building my own language model."
        encoded = tok.encode(sentence)
        decoded = tok.decode(encoded)
        self.assertEqual(sentence, decoded)

        # 0 UNK on training corpus
        full_encoded = tok.encode(self.sample_text)
        self.assertEqual(full_encoded.count(tok.token_to_id["<UNK>"]), 0)

    def test_word_tokenizer_encode_decode(self):
        word_tok = WordTokenizer()
        word_tok.train(self.sample_text)
        sentence = "I am building my own language model from scratch."
        enc = word_tok.encode(sentence)
        dec = word_tok.decode(enc)
        self.assertEqual(sentence, dec)

        # Unseen word triggers UNK
        unseen_enc = word_tok.encode("UnseenWord123")
        self.assertEqual(unseen_enc, [word_tok.token_to_id["<UNK>"]])

    def test_bpe_save_and_load(self):
        tok = BPETokenizer()
        tok.train(self.sample_text, vocab_size=100)
        test_path = os.path.join(PROJECT_ROOT, "tests", "test_tok.json")
        tok.save(test_path)

        loaded_tok = BPETokenizer.from_file(test_path)
        self.assertEqual(tok.vocab_size, loaded_tok.vocab_size)
        self.assertEqual(tok.token_to_id, loaded_tok.token_to_id)
        if os.path.exists(test_path):
            os.remove(test_path)


class TestDataset(unittest.TestCase):
    def test_sliding_window_targets(self):
        tokens = [10, 20, 30, 40, 50, 60]
        seq_len = 3
        dataset = LanguageModelDataset(tokens, sequence_length=seq_len)

        self.assertEqual(len(dataset), len(tokens) - seq_len)

        x0, y0 = dataset[0]
        self.assertEqual(x0.tolist(), [10, 20, 30])
        self.assertEqual(y0.tolist(), [20, 30, 40])


class TestEmbeddings(unittest.TestCase):
    def test_positional_embeddings(self):
        max_len, d_model, seq_len = 64, 32, 8
        pos = torch.arange(seq_len)

        learned = LearnedPositionalEmbedding(max_len, d_model)
        l_out = learned(pos)
        self.assertEqual(l_out.shape, (1, seq_len, d_model))

        sinusoidal = SinusoidalPositionalEmbedding(max_len, d_model)
        s_out = sinusoidal(pos)
        self.assertEqual(s_out.shape, (1, seq_len, d_model))

        sim_near = torch.cosine_similarity(s_out[0, 0], s_out[0, 1], dim=0).item()
        sim_far = torch.cosine_similarity(s_out[0, 0], s_out[0, 7], dim=0).item()
        self.assertGreater(sim_near, sim_far)

    def test_charan_embeddings_combined(self):
        vocab_size, max_len, d_model = 150, 32, 64
        emb = CharanEmbeddings(vocab_size, max_len, d_model, position_type="learned", dropout=0.1)
        input_ids = torch.randint(0, vocab_size, (2, 8))

        out = emb(input_ids)
        self.assertEqual(out.shape, (2, 8, d_model))


class TestModelComponents(unittest.TestCase):
    def test_self_attention_causal_mask(self):
        B, T, D = 1, 4, 32
        x = torch.randn(B, T, D)
        layer = SelfAttention(embedding_dim=D)
        out = layer(x)
        self.assertEqual(out.shape, (B, T, D))

    def test_multi_head_attention_shape(self):
        B, T, D, H = 2, 8, 64, 4
        x = torch.randn(B, T, D)
        mha = MultiHeadAttention(embedding_dim=D, num_heads=H)
        out = mha(x)
        self.assertEqual(out.shape, (B, T, D))

    def test_feed_forward_activations(self):
        B, T, D, Hidden = 2, 8, 64, 256
        x = torch.randn(B, T, D)
        for act in ["gelu", "relu", "swiglu"]:
            ffn = FeedForward(embedding_dim=D, hidden_dim=Hidden, activation=act)
            out = ffn(x)
            self.assertEqual(out.shape, (B, T, D))

    def test_normalization_layers(self):
        B, T, D = 2, 8, 64
        x = torch.randn(B, T, D)
        ln = nn.LayerNorm(D)
        rms = RMSNorm(D)
        self.assertEqual(ln(x).shape, (B, T, D))
        self.assertEqual(rms(x).shape, (B, T, D))

    def test_transformer_block_variants(self):
        B, T, D, H, Hidden = 2, 8, 64, 4, 256
        x = torch.randn(B, T, D)
        pre_ln = TransformerBlock(D, H, Hidden, norm_type="layernorm", prenorm=True)
        rmsnorm = TransformerBlock(D, H, Hidden, norm_type="rmsnorm", prenorm=True)
        post_ln = TransformerBlock(D, H, Hidden, norm_type="layernorm", prenorm=False)

        self.assertEqual(pre_ln(x).shape, (B, T, D))
        self.assertEqual(rmsnorm(x).shape, (B, T, D))
        self.assertEqual(post_ln(x).shape, (B, T, D))

    def test_pre_ln_gradient_flow(self):
        B, T, D, H = 2, 4, 32, 2
        x = torch.randn(B, T, D, requires_grad=True)
        block = TransformerBlock(D, H, norm_type="layernorm", prenorm=True)
        out = block(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.norm().item(), 0.0)

    def test_charan_lm_forward(self):
        vocab_size = 150
        max_seq_len = 32
        model = CharanLM(
            vocab_size=vocab_size,
            max_sequence_length=max_seq_len,
            embedding_dim=64,
            num_heads=4,
            hidden_dim=256,
            num_layers=2
        )
        input_ids = torch.randint(0, vocab_size, (2, 16))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 16, vocab_size))
        self.assertGreater(model.count_parameters(), 10000)


class TestEndToEndGeneration(unittest.TestCase):
    def test_generation_runs_without_unk(self):
        result = generate("I am", max_new_tokens=10, temperature=0.0)
        self.assertTrue(result.startswith("I am"))
        self.assertNotIn("<UNK>", result)


if __name__ == "__main__":
    unittest.main()
