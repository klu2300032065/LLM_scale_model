"""
CharanAI - Transformer Block Diagnostic & Gradient Highway Inspector
=====================================================================
Demonstrates the critical difference between Pre-LN and Post-LN architectures
by measuring gradient flow across a deep 10-layer stack of Transformer Blocks.
"""
import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.transformer_block import TransformerBlock


def inspect_gradient_flow():
    torch.manual_seed(42)
    B, T, D, H, num_layers = 2, 8, 64, 4, 10

    print("=" * 75)
    print(f"CharanAI — Deep Transformer Gradient Highway Analysis ({num_layers} Layers)")
    print("=" * 75)

    # 1. Build Pre-LN Stack
    pre_ln_blocks = nn.ModuleList([
        TransformerBlock(D, H, norm_type="layernorm", prenorm=True) for _ in range(num_layers)
    ])

    # 2. Build Post-LN Stack
    post_ln_blocks = nn.ModuleList([
        TransformerBlock(D, H, norm_type="layernorm", prenorm=False) for _ in range(num_layers)
    ])

    x_pre = torch.randn(B, T, D, requires_grad=True)
    x_post = torch.randn(B, T, D, requires_grad=True)

    # Forward through Pre-LN
    h_pre = x_pre
    for block in pre_ln_blocks:
        h_pre = block(h_pre)
    loss_pre = h_pre.sum()
    loss_pre.backward()

    # Forward through Post-LN
    h_post = x_post
    for block in post_ln_blocks:
        h_post = block(h_post)
    loss_post = h_post.sum()
    loss_post.backward()

    grad_norm_pre = x_pre.grad.norm().item()
    grad_norm_post = x_post.grad.norm().item()

    print(f"\nGradient Magnitude at Input Layer (x0):")
    print(f" - Pre-LN Input Gradient Norm:  {grad_norm_pre:.4f}  (Healthy, direct gradient highway)")
    print(f" - Post-LN Input Gradient Norm: {grad_norm_post:.4f}  (Subject to normalization distortion)")

    print("\nLayer-by-Layer Weight Gradient Magnitudes:")
    print(f"{'Layer':<8} | {'Pre-LN Query Grad Norm':<25} | {'Post-LN Query Grad Norm':<25}")
    print("-" * 65)

    for i in range(num_layers):
        pre_g = pre_ln_blocks[i].attention.query.weight.grad.norm().item()
        post_g = post_ln_blocks[i].attention.query.weight.grad.norm().item()
        print(f"Layer {i+1:<2} | {pre_g:<25.4f} | {post_g:<25.4f}")

    print("=" * 75)
    print("Engineering Conclusion:")
    print("Pre-LN preserves a clean identity path: d(x_L)/d(x_0) = I + sum(d(F_l)/d(x_0)).")
    print("This allows training 100+ layer Transformers without learning rate warmup hacks!")
    print("=" * 75)


if __name__ == "__main__":
    inspect_gradient_flow()
