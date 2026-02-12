"""
Factory for creating M3PO gating functions.

Provides a registry-based factory pattern for selecting gating implementations.
"""

from typing import Dict, Optional

from .base import BaseM3POGating


# Registry mapping gating type names to their implementation classes
# 'baseline' maps to None, which means use original cosine similarity in m3po_utils.py
GATING_REGISTRY = {
    "baseline": None,  # Original cosine similarity (no gating object)
}


def create_gating_function(
    gating_type: str,
    config: Optional[Dict] = None,
) -> Optional[BaseM3POGating]:
    """
    Factory function to create a gating function instance.

    Args:
        gating_type: Name of the gating function (must be in GATING_REGISTRY)
        config: Optional configuration dict with parameters like:
            - temperature: Temperature for softmax (default: 0.1)
            - rank: Rank for low-rank factorization (for learnable gates)
            - attn_dim: Attention dimension (for Bahdanau)
            - debug: Enable debug logging

    Returns:
        Instance of BaseM3POGating subclass, or None for baseline

    Raises:
        ValueError: If gating_type is not in GATING_REGISTRY

    Example:
        >>> gating_fn = create_gating_function('raw_dot', {'temperature': 0.1})
        >>> gating_fn = create_gating_function('baseline')  # Returns None
    """
    if gating_type not in GATING_REGISTRY:
        available = ", ".join(GATING_REGISTRY.keys())
        raise ValueError(
            f"Unknown gating type: '{gating_type}'. Available types: {available}"
        )

    gating_class = GATING_REGISTRY[gating_type]

    if gating_class is None:
        # Baseline - use original cosine similarity
        return None

    # Create instance with config
    return gating_class(config=config)


def register_gating_function(name: str, gating_class: type):
    """
    Register a new gating function in the registry.

    Args:
        name: Name to register under (e.g., 'raw_dot')
        gating_class: Class that inherits from BaseM3POGating

    Example:
        >>> register_gating_function('my_gate', MyGatingClass)
    """
    if not (gating_class is None or issubclass(gating_class, BaseM3POGating)):
        raise TypeError(
            f"Gating class must inherit from BaseM3POGating, got {gating_class}"
        )

    GATING_REGISTRY[name] = gating_class


def list_gating_functions() -> list:
    """
    List all available gating functions.

    Returns:
        List of gating function names
    """
    return list(GATING_REGISTRY.keys())
