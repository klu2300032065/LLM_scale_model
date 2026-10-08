"""
CharanAI - Model Training Pipeline
==================================
Trains the CharanGPT model on processed token sequences using PyTorch.
Includes:
 - Device auto-detection (CUDA / CPU)
 - Dataset loading with sliding causal context window
 - Cross-Entropy Loss computation with integrated forward pass
 - AdamW optimizer step with weight decay
 - Loss progression tracking and model checkpointing
"""
import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.charan_gpt import CharanGPT, CharanGPTConfig
from training.dataset import LanguageModelDataset
from training.build_dataset import build_dataset, PROCESSED_DATA_PATH, TOKENIZER_PATH
from tokenizer.bpe_tokenizer import BPETokenizer

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = CharanGPTConfig(
    vocab_size=150,
    max_sequence_length=32,
    embedding_dim=64,
    num_heads=4,
    hidden_dim=256,
    num_layers=2,
    dropout=0.05,
    weight_tying=True,
    norm_type="layernorm",
    activation="gelu"
)

LEARNING_RATE = 0.003
EPOCHS = 60
BATCH_SIZE = 16
SEED = 42

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "charan_lm.pth")


def train():
    torch.manual_seed(SEED)

    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("CharanAI — CharanGPT Training Pipeline")
    print("=" * 60)
    print(f"Device: {device}")

    # 2. Check/Build Processed Dataset
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Processed data or tokenizer missing. Building dataset first...")
        build_dataset()

    token_ids = torch.load(PROCESSED_DATA_PATH)
    print(f"Loaded token dataset: {len(token_ids)} tokens.")

    # 3. Create Dataset & DataLoader
    dataset = LanguageModelDataset(token_ids, sequence_length=CONFIG.max_sequence_length)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset samples: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    # 4. Instantiate Model
    model = CharanGPT(CONFIG).to(device)
    print(f"Total Trainable Parameters: {model.count_parameters():,}")
    print(f"Non-Embedding Parameters:  {model.count_parameters(non_embedding=True):,}")

    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 6. Training Loop
    print("\nStarting Training Loop...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            # Forward pass with built-in loss computation
            logits, loss = model(input_ids, targets=target_ids)

            # Backward pass & weight update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # 7. Save Checkpoint using CharanGPT API
    model.save_checkpoint(
        CHECKPOINT_PATH,
        optimizer=optimizer,
        epoch=EPOCHS,
        loss=avg_loss
    )
    print("=" * 60)


if __name__ == "__main__":
    train()
