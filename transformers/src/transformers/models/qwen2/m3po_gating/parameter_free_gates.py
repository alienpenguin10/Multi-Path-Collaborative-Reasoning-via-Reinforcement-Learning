"""
Parameter-free gating functions for M3PO.

These gating functions compute similarity between output distributions
without any learnable parameters. They provide baseline comparisons to
understand the effect of different similarity measures.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .base import BaseM3POGating


class RawDotProductGating(BaseM3POGating):
    """
    Raw dot-product similarity without normalization.

    Computes: S_ij = p_i · p_j (unnormalized)

    This is confidence-weighted - paths with higher probability mass will have
    larger similarity scores. Unlike cosine similarity, this does not normalize
    by magnitude, so high-confidence predictions have stronger influence.

    Characteristics:
    - No normalization (confidence-weighted)
    - Symmetric: S_ij = S_ji
    - Range: [0, 1] since p_i and p_j are probability distributions
    - Emphasizes high-confidence paths
    """

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute raw dot-product similarity.

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used (for API compatibility)

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        # Raw dot product: p_i · p_j
        # Since both are probability distributions (sum to 1), result is in [0, 1]
        similarity_matrix = torch.mm(output_distributions, output_distributions.t())

        return similarity_matrix


class ScaledDotProductGating(BaseM3POGating):
    """
    Scaled dot-product attention (Transformer-style).

    Computes: S_ij = (p_i · p_j) / sqrt(d)
    where d is the vocabulary size.

    This is the standard attention mechanism from "Attention is All You Need".
    Scaling by sqrt(d) prevents the dot products from becoming too large,
    which would push the softmax into regions with extremely small gradients.

    Characteristics:
    - Scaled for numerical stability
    - Symmetric: S_ij = S_ji
    - Range: approximately [0, 1/sqrt(vocab_size)]
    - More stable than raw dot-product for large vocabularies
    """

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product similarity.

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used (for API compatibility)

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        vocab_size = output_distributions.size(1)
        scaling_factor = vocab_size ** 0.5

        # Scaled dot product: (p_i · p_j) / sqrt(vocab_size)
        similarity_matrix = torch.mm(output_distributions, output_distributions.t()) / scaling_factor

        return similarity_matrix


class KLDivergenceGating(BaseM3POGating):
    """
    Jensen-Shannon Divergence (JSD) similarity.

    Computes: S_ij = 1 - sqrt(JSD(p_i || p_j))
    where JSD(p||q) = 0.5 * [KL(p||m) + KL(q||m)] and m = 0.5 * (p + q)

    JSD is a symmetric, smoothed version of KL divergence. It measures
    distributional similarity - paths with similar output distributions
    (even if low confidence) will have high similarity.

    Characteristics:
    - Distribution-theoretic (most principled measure)
    - Symmetric: JSD(p||q) = JSD(q||p)
    - Bounded: JSD in [0, log(2)]
    - Range after transform: [0, 1]
    - Computationally expensive: O(num_paths^2 * vocab_size)
    """

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute JSD-based similarity.

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used (for API compatibility)

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        num_paths = output_distributions.size(0)
        device = output_distributions.device

        # Add small epsilon for numerical stability
        eps = 1e-8
        output_distributions = output_distributions + eps
        output_distributions = output_distributions / output_distributions.sum(dim=1, keepdim=True)

        # Compute pairwise JSD
        similarity_matrix = torch.zeros(num_paths, num_paths, device=device)

        for i in range(num_paths):
            for j in range(i, num_paths):  # Only compute upper triangle (symmetric)
                p = output_distributions[i]
                q = output_distributions[j]

                # Jensen-Shannon Divergence
                m = 0.5 * (p + q)

                # KL(p || m)
                kl_pm = (p * (torch.log(p) - torch.log(m))).sum()

                # KL(q || m)
                kl_qm = (q * (torch.log(q) - torch.log(m))).sum()

                # JSD = 0.5 * [KL(p||m) + KL(q||m)]
                jsd = 0.5 * (kl_pm + kl_qm)

                # Convert to similarity: 1 - sqrt(JSD)
                # sqrt brings it from [0, log(2)] to [0, sqrt(log(2))] ≈ [0, 0.83]
                similarity = 1.0 - torch.sqrt(jsd.clamp(min=0))

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity  # Symmetric

        return similarity_matrix


class BhattacharyyaGating(BaseM3POGating):
    """
    Bhattacharyya coefficient similarity.

    Computes: S_ij = sum(sqrt(p_i * p_j))

    The Bhattacharyya coefficient measures the overlap between two probability
    distributions. It's related to the Bhattacharyya distance and Hellinger distance.

    Characteristics:
    - Probability-theoretic measure
    - Symmetric: BC(p,q) = BC(q,p)
    - Range: [0, 1] (1 = identical distributions)
    - Efficient to compute: O(vocab_size) per pair
    - More interpretable than JSD
    """

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Bhattacharyya coefficient similarity.

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used (for API compatibility)

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        # Take square root of distributions
        sqrt_dists = torch.sqrt(output_distributions + 1e-8)

        # Bhattacharyya coefficient: BC = sum(sqrt(p_i * p_j)) = sum(sqrt(p_i) * sqrt(p_j))
        similarity_matrix = torch.mm(sqrt_dists, sqrt_dists.t())

        return similarity_matrix


# Register all parameter-free gates
from .factory import register_gating_function

register_gating_function("raw_dot", RawDotProductGating)
register_gating_function("scaled_dot", ScaledDotProductGating)
register_gating_function("kl_divergence", KLDivergenceGating)
register_gating_function("bhattacharyya", BhattacharyyaGating)
