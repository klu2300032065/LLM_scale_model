"""
CharanAI - Language Model Dataset
=================================
Implements a PyTorch Dataset for causal autoregressive next-token prediction.
Given a sequence of token IDs, slices sliding windows of length `sequence_length`:
 - Input  x = tokens[i : i + sequence_length]
 - Target y = tokens[i + 1 : i + sequence_length + 1]
"""
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """
    Sliding window dataset for training autoregressive language models.

    Args:
        token_ids: List of integers or 1D torch.Tensor of token IDs.
        sequence_length: The context window size (T / block_size).
    """

    def __init__(
        self,
        token_ids: Union[List[int], torch.Tensor],
        sequence_length: int
    ):
        if isinstance(token_ids, list):
            self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        elif isinstance(token_ids, torch.Tensor):
            self.token_ids = token_ids.to(dtype=torch.long)
        else:
            raise TypeError("token_ids must be a List[int] or torch.Tensor")

        self.sequence_length = sequence_length

        if len(self.token_ids) <= self.sequence_length:
            raise ValueError(
                f"Token sequence length ({len(self.token_ids)}) must be greater "
                f"than context window sequence_length ({self.sequence_length})"
            )

    def __len__(self) -> int:
        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence: tokens from index to index + sequence_length
        input_ids = self.token_ids[index : index + self.sequence_length]

        # Target sequence: next tokens, shifted by 1 position
        target_ids = self.token_ids[index + 1 : index + self.sequence_length + 1]

        return input_ids, target_ids


if __name__ == "__main__":
    # Educational demonstration of causal language modeling targets
    sample_tokens = list(range(10, 22))  # [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    context_len = 4

    dataset = LanguageModelDataset(sample_tokens, sequence_length=context_len)

    print(f"Sample Tokens ({len(sample_tokens)}):", sample_tokens)
    print(f"Context Window Length: {context_len}")
    print(f"Number of training samples: {len(dataset)}")

    x0, y0 = dataset[0]
    print("\nSample 0:")
    print("  Input  (x):", x0.tolist())
    print("  Target (y):", y0.tolist())
    print("  Mathematical alignment: target at position t is the true next token for input up to t.")
