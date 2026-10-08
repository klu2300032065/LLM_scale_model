"""
CharanAI - Model Training Pipeline
==================================
Trains the CharanLM model on processed token sequences using PyTorch.
Includes:
 - Device auto-detection (CUDA / CPU)
 - Dataset loading with sliding causal context window
 - Cross-Entropy Loss computation
 - AdamW optimizer step
 - Loss progression tracking and model checkpointing
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.charan_lm import CharanLM
from training.dataset import LanguageModelDataset
from training.build_dataset import build_dataset, PROCESSED_DATA_PATH, TOKENIZER_PATH
from tokenizer.bpe_tokenizer import BPETokenizer

# ==============================================================================
# Hyperparameters & Configuration
# ==============================================================================
VOCAB_SIZE = 150
MAX_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 64
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 2

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
    print("=" * 55)
    print("CharanAI — Training Pipeline")
    print("=" * 55)
    print(f"Device: {device}")

    # 2. Check/Build Processed Dataset
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Processed data or tokenizer missing. Building dataset first...")
        build_dataset()

    token_ids = torch.load(PROCESSED_DATA_PATH)
    print(f"Loaded token dataset: {len(token_ids)} tokens.")

    # 3. Create Dataset & DataLoader
    dataset = LanguageModelDataset(token_ids, sequence_length=MAX_SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset samples: {len(dataset)} | Batches per epoch: {len(dataloader)}")

    # 4. Instantiate Model
    model = CharanLM(
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # 5. Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
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

            # Forward pass: (B, T) -> (B, T, vocab_size)
            logits = model(input_ids)

            # Reshape for Cross-Entropy Loss: (B*T, vocab_size) vs (B*T)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

            # Backward pass & weight update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # 7. Save Checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
            },
            "final_loss": avg_loss,
        },
        CHECKPOINT_PATH
    )
    print(f"Checkpoint successfully saved to: {CHECKPOINT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    train()
