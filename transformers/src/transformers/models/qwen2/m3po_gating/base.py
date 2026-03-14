"""
Base class for M3PO gating functions.

All gating implementations inherit from BaseM3POGating and implement the
compute_similarity_matrix() method. The base class provides shared functionality
for converting similarity to attention weights and tracking statistics.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseM3POGating(ABC, nn.Module):
    """
    Abstract base class for M3PO gating functions.

    All concrete gating implementations must implement compute_similarity_matrix().
    The base class provides:
    - Attention weight computation from similarity matrices
    - Statistics tracking (entropy, max weight, variance)
    - Debug logging utilities
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the gating function.

        Args:
            config: Optional configuration dict with parameters like:
                - temperature: Temperature for softmax (default: 0.1)
                - debug: Enable debug logging (default: False)
        """
        super().__init__()
        self.config = config or {}
        self.temperature = self.config.get("temperature", 0.1)
        self.debug = self.config.get("debug", False) or os.environ.get("M3PO_DEBUG", "-1") == "1"

        # Statistics tracking
        self.stats = {
            "attention_entropy": [],
            "attention_max": [],
            "attention_variance": [],
            "similarity_mean": [],
            "similarity_std": [],
        }

    @property
    def has_learnable_parameters(self) -> bool:
        """
        Returns True if this gating function has learnable parameters.

        Override this in learnable subclasses to return True.
        """
        return False

    @abstractmethod
    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pairwise similarity between paths.

        Args:
            output_distributions: Softmax probabilities over vocabulary [num_paths, vocab_size]
            hidden_states: Optional hidden states [num_paths, hidden_dim] (for future use)

        Returns:
            similarity_matrix: Pairwise similarity [num_paths, num_paths]
                - Values should be in a reasonable range (e.g., [-1, 1] or [0, 1])
                - Diagonal will be masked out by compute_attention_weights()
        """
        pass

    def compute_attention_weights(
        self,
        similarity_matrix: torch.Tensor,
        thinking_mask: torch.Tensor,
        mask_diagonal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert similarity matrix to attention weights using temperature-scaled softmax.

        This implements the core M3PO attention mechanism (Eq. 5 in the paper):
            A_ij = softmax(S_ij / T)
        where S_ij is the similarity matrix and T is temperature.

        Args:
            similarity_matrix: [num_paths, num_paths] pairwise similarities
            thinking_mask: [num_paths] boolean mask, True for paths still generating reasoning
            mask_diagonal: If True, prevent self-attention (default: True)

        Returns:
            attention_weights: [num_paths, num_paths] softmax attention weights
            valid_mask: [num_paths, num_paths] mask for valid attention targets
        """
        num_paths = similarity_matrix.size(0)
        device = similarity_matrix.device

        # Create mask for valid attention targets
        # valid_mask[i, j] = True if path i can attend to path j
        valid_mask = thinking_mask.unsqueeze(0) & thinking_mask.unsqueeze(1)  # [num_paths, num_paths]

        if mask_diagonal:
            # Prevent self-attention
            diagonal_mask = ~torch.eye(num_paths, dtype=torch.bool, device=device)
            valid_mask = valid_mask & diagonal_mask

        # Apply temperature scaling
        scaled_similarity = similarity_matrix / self.temperature

        # Mask out invalid positions with large negative value
        masked_similarity = scaled_similarity.masked_fill(~valid_mask, float("-inf"))

        # Compute softmax attention weights
        attention_weights = F.softmax(masked_similarity, dim=1)  # Softmax over attention targets (dim 1)

        # Handle edge case: if a path has no valid attention targets, set attention to 0
        # This happens when only 1 path is thinking or a path is isolated
        # NaN can occur when all values in a row are -inf (softmax of all -inf = NaN)
        has_valid_targets = valid_mask.any(dim=1)  # [num_paths]

        # Replace NaN rows with zeros (using where instead of multiply to avoid NaN * 0 = NaN)
        attention_weights = torch.where(
            has_valid_targets.unsqueeze(1),
            attention_weights,
            torch.zeros_like(attention_weights)
        )

        # Track statistics
        self._update_stats(similarity_matrix, attention_weights, valid_mask)

        if self.debug:
            self._log_attention_stats(similarity_matrix, attention_weights, thinking_mask)

        return attention_weights, valid_mask

    def _update_stats(
        self,
        similarity_matrix: torch.Tensor,
        attention_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        """Update statistics for later analysis."""
        with torch.no_grad():
            # Only compute stats for valid attention weights
            valid_attention = attention_weights[valid_mask]

            if valid_attention.numel() > 0:
                # Attention entropy (per path)
                # Entropy = -sum(p * log(p)), higher = more uniform attention
                eps = 1e-10
                entropy_per_path = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=1)
                self.stats["attention_entropy"].append(entropy_per_path.mean().item())

                # Attention max (per path) - measures focus
                max_per_path = attention_weights.max(dim=1)[0]
                self.stats["attention_max"].append(max_per_path.mean().item())

                # Attention variance (per path)
                var_per_path = attention_weights.var(dim=1)
                self.stats["attention_variance"].append(var_per_path.mean().item())

            # Similarity statistics
            valid_similarity = similarity_matrix[valid_mask]
            if valid_similarity.numel() > 0:
                self.stats["similarity_mean"].append(valid_similarity.mean().item())
                self.stats["similarity_std"].append(valid_similarity.std().item())

    def _log_attention_stats(
        self,
        similarity_matrix: torch.Tensor,
        attention_weights: torch.Tensor,
        thinking_mask: torch.Tensor,
    ):
        """Log detailed attention statistics for debugging."""
        num_thinking = thinking_mask.sum().item()
        print(f"\n[{self.__class__.__name__}] Attention Statistics:")
        print(f"  Num thinking paths: {num_thinking}/{len(thinking_mask)}")

        if num_thinking > 0 and num_thinking <= 8:  # Only print for small batches
            print(f"  Similarity matrix:\n{similarity_matrix}")
            print(f"  Attention weights:\n{attention_weights}")

        # Summary statistics
        if attention_weights.sum() > 0:
            print(f"  Attention entropy: {self.stats['attention_entropy'][-1]:.4f}")
            print(f"  Attention max: {self.stats['attention_max'][-1]:.4f}")
            print(f"  Similarity mean: {self.stats['similarity_mean'][-1]:.4f}")

    def get_stats_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all steps.

        Returns:
            Dict with mean values for each tracked statistic.
        """
        summary = {}
        for key, values in self.stats.items():
            if len(values) > 0:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_std"] = (
                    (sum((x - summary[f"{key}_mean"]) ** 2 for x in values) / len(values)) ** 0.5
                )
        return summary

    def reset_stats(self):
        """Reset statistics tracking."""
        for key in self.stats:
            self.stats[key] = []

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"temperature={self.temperature}, learnable={self.has_learnable_parameters}"
