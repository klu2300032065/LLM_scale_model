# CharanAI — Small GPT-style Language Model from Scratch

**CharanAI** is an educational, custom-built GPT-style causal autoregressive language model designed and implemented from the ground up in PyTorch without using any pretrained LLM weights.

---

## Architecture Overview

```text
Raw Text
   ↓
BPETokenizer (Subword Merges)
   ↓
Token IDs
   ↓
Token Embeddings + Learned Positional Embeddings
   ↓
Transformer Blocks (x2)
   │
   ├── LayerNorm (Pre-LN)
   ├── Multi-Head Causal Self-Attention (4 heads)
   ├── Residual Connection
   ├── LayerNorm (Pre-LN)
   ├── Feed-Forward Network (GELU, d_model -> 4*d_model -> d_model)
   └── Residual Connection
   ↓
Final LayerNorm
   ↓
Linear LM Head
   ↓
Logits (CrossEntropyLoss during training / Softmax + Sampling during generation)
   ↓
Generated Text
```

---

## Project Structure

```text
C:\CharanAI
│
├── data/
│   ├── raw/
│   │   └── training.txt          # Raw text corpus
│   └── processed/
│       └── token_ids.pt          # Tokenized PyTorch tensor
│
├── tokenizer/
│   ├── bpe_tokenizer.py          # Custom Byte-Pair Encoding implementation
│   ├── bpe_tokenizer.json        # Learned vocab & merge rules
│   ├── tokenizer.py              # Character-level tokenizer
│   └── loader.py                 # Module compatibility loader
│
├── model/
│   ├── embeddings.py             # Token embeddings
│   ├── positional_embedding.py   # Positional embeddings
│   ├── attention.py              # Single-head self-attention
│   ├── multi_head_attention.py   # Multi-head causal attention
│   ├── feed_forward.py           # Position-wise MLP
│   ├── transformer_block.py      # Pre-LN Transformer decoder block
│   └── charan_lm.py              # Complete CharanLM architecture
│
├── training/
│   ├── dataset.py                # Sliding-window causal dataset
│   ├── build_dataset.py          # Data preparation pipeline
│   └── train.py                  # PyTorch training loop
│
├── inference/
│   └── generate.py               # Autoregressive text generation
│
├── tests/
│   └── test_pipeline.py          # End-to-end automated test suite
│
├── checkpoints/
│   └── charan_lm.pth             # Model weights checkpoint
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quickstart

### 1. Build Tokenizer and Dataset
```bash
python training/build_dataset.py
```

### 2. Train CharanLM
```bash
python training/train.py
```

### 3. Generate Text
```bash
python inference/generate.py
```

### 4. Run Test Suite
```bash
python tests/test_pipeline.py
```
