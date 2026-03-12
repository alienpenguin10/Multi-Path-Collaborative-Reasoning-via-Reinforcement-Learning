"""
Learnable gating functions for M3PO.

These gating functions compute similarity between output distributions
using learnable parameters that receive gradients during GRPO loss computation
(via the Phase 3 differentiable path through apply_m3po_to_logits).

Luong Attention:
    S_ij = p_i^T W p_j  (bilinear)
    Low-rank factorization: W = U @ V^T to reduce from O(V^2) to O(V*r) parameters
    where V = vocab_size and r = rank.

Bahdanau Attention:
    S_ij = v^T tanh(W1 * p_i + W2 * p_j)  (additive / MLP-style)
    Projects distributions to a lower-dimensional space before computing similarity.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseM3POGating


class LuongAttentionGating(BaseM3POGating):
    """
    Luong-style bilinear attention with low-rank factorization.

    Computes: S_ij = p_i^T W p_j
    where W = U @ V^T is a low-rank factorization.

    Full W would be (vocab_size, vocab_size) = ~23B parameters for Qwen2.5.
    Low-rank W = U @ V^T with rank r:
        U: (vocab_size, rank) and V: (vocab_size, rank)
        Total: 2 * vocab_size * rank parameters

    For Qwen2.5 (vocab_size=151936) with rank=128:
        2 * 151936 * 128 = ~38.9M parameters

    The bilinear form captures asymmetric interactions between vocabulary
    positions, allowing the model to learn which token predictions should
    influence each other across paths.

    Characteristics:
    - Learnable: parameters receive gradients via Phase 3 infrastructure
    - Asymmetric: S_ij != S_ji in general (captures directional influence)
    - Low-rank: tractable parameter count via factorization
    - Range: unbounded (temperature scaling handles normalization)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.rank = self.config.get("rank", 512)
        self.vocab_size = self.config.get("vocab_size", 151936)

        # Low-rank factorization: W = U @ V^T
        init_strategy = self.config.get("init_strategy", "identity")
        if init_strategy == "identity":
            # QR decomposition gives orthonormal columns; setting U=V=Q makes
            # W = Q @ Q^T ≈ partial identity, so initial similarities are
            # meaningful (~0.01-0.1) instead of ~0 from tiny Xavier init.
            random_matrix = torch.randn(self.vocab_size, self.rank)
            Q, _ = torch.linalg.qr(random_matrix)
            self.U = nn.Parameter(Q.clone().contiguous())
            self.V = nn.Parameter(Q.clone().contiguous())
        else:  # xavier
            std = math.sqrt(2.0 / (self.vocab_size + self.rank))
            self.U = nn.Parameter(torch.randn(self.vocab_size, self.rank) * std)
            self.V = nn.Parameter(torch.randn(self.vocab_size, self.rank) * std)

    @property
    def has_learnable_parameters(self) -> bool:
        return True

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute bilinear similarity: S_ij = p_i^T (U @ V^T) p_j

        Efficient computation order (avoids materializing the full W):
            1. a = p @ U        → (N, rank)    [O(N * V * r)]
            2. b = p @ V        → (N, rank)    [O(N * V * r)]
            3. S = a @ b^T      → (N, N)       [O(N^2 * r)]

        Total: O(N * V * r) instead of O(N * V^2) if W were materialized.

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        # Cast small input to float32 to match parameters (avoids creating large bfloat16 copies of U/V)
        # output_distributions is (N, vocab_size) ~ 2.4MB vs U/V at (vocab_size, rank) ~ 38MB each
        p = output_distributions.float()

        # Project distributions through low-rank factors
        a = torch.mm(p, self.U)  # (N, rank)
        b = torch.mm(p, self.V)  # (N, rank)

        # Bilinear similarity
        similarity_matrix = torch.mm(a, b.t())  # (N, N)

        return similarity_matrix

    def extra_repr(self) -> str:
        num_params = 2 * self.vocab_size * self.rank
        return (
            f"temperature={self.temperature}, rank={self.rank}, "
            f"vocab_size={self.vocab_size}, params={num_params:,}"
        )


class BahdanauAttentionGating(BaseM3POGating):
    """
    Bahdanau-style additive (MLP) attention.

    Computes: S_ij = v^T tanh(W1 * p_i + W2 * p_j)

    This projects each distribution into a lower-dimensional attention space,
    combines them additively, applies a nonlinearity, then scores with a
    learned vector.

    Parameters:
        W1: (attn_dim, vocab_size) - projects "query" distribution
        W2: (attn_dim, vocab_size) - projects "key" distribution
        v:  (attn_dim,)            - scoring vector

    For Qwen2.5 (vocab_size=151936) with attn_dim=256:
        2 * 256 * 151936 + 256 = ~77.8M parameters

    The additive form captures nonlinear relationships between distributions
    that the bilinear Luong form cannot. The tanh nonlinearity allows the
    model to learn complex interaction patterns.

    Characteristics:
    - Learnable: parameters receive gradients via Phase 3 infrastructure
    - Symmetric by default: W1=W2 makes S_ij = S_ji (but learned W1!=W2 breaks symmetry)
    - MLP-style: can capture nonlinear distribution relationships
    - Range: [-1, 1] per element before sum (bounded by tanh and v)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.attn_dim = self.config.get("attn_dim", 256)
        self.vocab_size = self.config.get("vocab_size", 151936)

        # Projection matrices and scoring vector
        # Xavier initialization for layers feeding into tanh
        std_w = math.sqrt(2.0 / (self.vocab_size + self.attn_dim))
        self.W1 = nn.Parameter(torch.randn(self.attn_dim, self.vocab_size) * std_w)
        self.W2 = nn.Parameter(torch.randn(self.attn_dim, self.vocab_size) * std_w)

        # Scoring vector - smaller init to keep initial similarities near 0
        std_v = math.sqrt(1.0 / self.attn_dim)
        self.v = nn.Parameter(torch.randn(self.attn_dim) * std_v)

    @property
    def has_learnable_parameters(self) -> bool:
        return True

    def compute_similarity_matrix(
        self,
        output_distributions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute additive attention similarity: S_ij = v^T tanh(W1*p_i + W2*p_j)

        Efficient computation:
            1. h1 = W1 @ p^T    → (attn_dim, N)  [project queries]
            2. h2 = W2 @ p^T    → (attn_dim, N)  [project keys]
            3. For each (i,j): score = v^T tanh(h1[:,i] + h2[:,j])
               Vectorized: expand h1 and h2, add, tanh, dot with v

        Args:
            output_distributions: [num_paths, vocab_size] probability distributions
            hidden_states: Not used

        Returns:
            similarity_matrix: [num_paths, num_paths]
        """
        N = output_distributions.size(0)

        # Cast small input to float32 to match parameters (avoids creating large bfloat16 copies of W1/W2)
        p = output_distributions.float()

        # Project distributions: (attn_dim, N)
        h1 = torch.mm(self.W1, p.t())  # (attn_dim, N)
        h2 = torch.mm(self.W2, p.t())  # (attn_dim, N)

        # Expand for pairwise combination:
        # h1[:, i] + h2[:, j] for all (i, j)
        # h1: (attn_dim, N, 1) + h2: (attn_dim, 1, N) → (attn_dim, N, N)
        combined = h1.unsqueeze(2) + h2.unsqueeze(1)  # (attn_dim, N, N)

        # Apply nonlinearity
        activated = torch.tanh(combined)  # (attn_dim, N, N)

        # Score with v: v^T @ activated for each (i,j)
        # v: (attn_dim,) → (attn_dim, 1, 1) for broadcasting
        similarity_matrix = (self.v.unsqueeze(1).unsqueeze(2) * activated).sum(dim=0)  # (N, N)

        return similarity_matrix

    def extra_repr(self) -> str:
        num_params = 2 * self.attn_dim * self.vocab_size + self.attn_dim
        return (
            f"temperature={self.temperature}, attn_dim={self.attn_dim}, "
            f"vocab_size={self.vocab_size}, params={num_params:,}"
        )


# Register learnable gates
from .factory import register_gating_function

register_gating_function("luong", LuongAttentionGating)
register_gating_function("bahdanau", BahdanauAttentionGating)
