"""
CharanAI - Production-Grade Training Pipeline & Engine
=====================================================
A complete LLM pretraining system featuring:
 1. Train / Validation Data Splitting (90/10)
 2. Cosine Annealing Learning Rate Schedule with Linear Warmup
 3. Global Gradient Norm Clipping (max_norm=1.0)
 4. Validation Loss & Perplexity (PPL = exp(loss)) tracking
 5. Best & Final Checkpoint Saving & Metrics Logging (JSON)
 6. Training throughput benchmarking (Tokens/sec)
"""
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple
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
# Model & Training Configuration
# ==============================================================================
MODEL_CONFIG = CharanGPTConfig(
    vocab_size=150,
    max_sequence_length=32,
    embedding_dim=64,
    num_heads=4,
    hidden_dim=256,
    num_layers=2,
    dropout=0.0,
    weight_tying=True,
    norm_type="layernorm",
    activation="gelu"
)

MAX_LR = 0.003
MIN_LR = 0.0003
WARMUP_EPOCHS = 5
TOTAL_EPOCHS = 60
BATCH_SIZE = 16
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 0.01
SEED = 42

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
FINAL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "charan_lm.pth")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_charan_lm.pth")
METRICS_PATH = os.path.join(CHECKPOINT_DIR, "training_metrics.json")


def get_learning_rate(epoch: int, total_epochs: int, warmup_epochs: int, max_lr: float, min_lr: float) -> float:
    """Computes learning rate using Linear Warmup followed by Cosine Annealing Decay."""
    if epoch <= warmup_epochs:
        return max_lr * (epoch / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: CharanGPT, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evaluates model on validation data, computing average loss and perplexity."""
    model.eval()
    total_loss = 0.0
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        _, loss = model(input_ids, targets=target_ids)
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(dataloader))
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity


def train():
    torch.manual_seed(SEED)

    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("CharanAI — Production-Grade CharanGPT Training Pipeline")
    print("=" * 70)
    print(f"Hardware Device: {device}")

    # 2. Check/Build Processed Dataset
    if not os.path.exists(PROCESSED_DATA_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Processed dataset missing. Building dataset first...")
        build_dataset()

    token_ids = torch.load(PROCESSED_DATA_PATH)
    total_tokens = len(token_ids)

    # 3. Train / Validation Split (90/10)
    split_idx = int(total_tokens * 0.9)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    train_dataset = LanguageModelDataset(token_ids, sequence_length=MODEL_CONFIG.max_sequence_length)
    val_dataset = LanguageModelDataset(val_tokens, sequence_length=MODEL_CONFIG.max_sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataset Statistics:")
    print(f" - Total Tokens:     {total_tokens}")
    print(f" - Train Samples:    {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # 4. Instantiate Model
    model = CharanGPT(MODEL_CONFIG).to(device)
    print(f"\nModel Architecture:")
    print(f" - Total Parameters:         {model.count_parameters():,}")
    print(f" - Non-Embedding Parameters: {model.count_parameters(non_embedding=True):,}")
    print(f" - Weight Tying:             {MODEL_CONFIG.weight_tying}")

    # 5. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=MIN_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    # 6. Training Loop
    metrics_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "learning_rate": []
    }

    best_val_loss = float("inf")
    print("\nStarting Pretraining Loop...")
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val PPL':<10} | {'LR':<10} | {'Tokens/sec':<10}")
    print("-" * 70)

    start_time = time.time()
    total_processed_tokens = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        current_lr = get_learning_rate(epoch, TOTAL_EPOCHS, WARMUP_EPOCHS, MAX_LR, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        total_train_loss = 0.0
        epoch_tokens = 0

        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()
            _, loss = model(input_ids, targets=target_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            optimizer.step()

            total_train_loss += loss.item()
            epoch_tokens += input_ids.numel()

        total_processed_tokens += epoch_tokens
        epoch_time = time.time() - epoch_start
        tokens_per_sec = epoch_tokens / max(epoch_time, 1e-5)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_ppl = evaluate(model, val_loader, device)

        metrics_history["epochs"].append(epoch)
        metrics_history["train_loss"].append(round(avg_train_loss, 4))
        metrics_history["val_loss"].append(round(avg_val_loss, 4))
        metrics_history["val_perplexity"].append(round(val_ppl, 2))
        metrics_history["learning_rate"].append(round(current_lr, 6))

        # Checkpoint Best Validation Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_checkpoint(
                BEST_CHECKPOINT_PATH,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_val_loss
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == TOTAL_EPOCHS:
            print(f"{epoch:<6} | {avg_train_loss:<11.4f} | {avg_val_loss:<10.4f} | {val_ppl:<10.2f} | {current_lr:<10.6f} | {tokens_per_sec:<10.0f}")

    total_time = time.time() - start_time

    # Save Final Checkpoint
    model.save_checkpoint(
        FINAL_CHECKPOINT_PATH,
        optimizer=optimizer,
        epoch=TOTAL_EPOCHS,
        loss=avg_train_loss
    )

    print("-" * 70)
    print(f"Training completed in {total_time:.2f}s (Average {(total_processed_tokens / total_time):.0f} tokens/sec).")
    print(f"Final Train Loss: {avg_train_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")

    # Save Metrics JSON
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Training metrics saved to: {METRICS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    train()
