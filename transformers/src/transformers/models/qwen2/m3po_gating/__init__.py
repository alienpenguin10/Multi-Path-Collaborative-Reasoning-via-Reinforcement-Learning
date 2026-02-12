"""
M3PO Gating Functions Package

This package provides alternative gating mechanisms for M3PO cross-path attention.
"""

from .base import BaseM3POGating
from .factory import create_gating_function, GATING_REGISTRY, register_gating_function

# Import gating implementations to trigger registration
from . import parameter_free_gates

__all__ = [
    "BaseM3POGating",
    "create_gating_function",
    "GATING_REGISTRY",
    "register_gating_function",
]
