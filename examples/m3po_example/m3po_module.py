"""
M3PO (Multi-Path Collaborative Reasoning) Module

This module implements cross-path interaction for collaborative reasoning
across multiple generation paths during training.

Based on the M3PO paper: Multi-Path Collaborative Reasoning via Reinforcement Learning
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class M3POConfig:
    """Configuration for M3PO cross-path interaction."""
    num_generations: int = 8
    lambda_blend: float = 0.1
    temperature: float = 0.1
    thinking_mask: Optional[List[bool]] = None


def apply_cross_path_interaction_batch(
    current_embeds: torch.Tensor,  # (batch_size * N, hidden_dim)
    num_generations: int,
    thinking_mask: Optional[List[bool]] = None,
    lambda_blend: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Apply cross-path interaction to batched embeddings.
    
    Groups sequences by question and applies cross-path blending within each group.
    
    Args:
        current_embeds: Hidden states (batch_size * N, hidden_dim)
        num_generations: Number of paths per question (N)
        thinking_mask: Optional mask for active paths (batch_size * N length)
        lambda_blend: Blending coefficient (0-1)
        temperature: Temperature for attention softmax
    
    Returns:
        Blended embeddings with same shape as input
    """
    batch_size = current_embeds.shape[0] // num_generations
    hidden_dim = current_embeds.shape[1]
    
    # Reshape to group by question: (batch_size * N, hidden_dim) -> (batch_size, N, hidden_dim)
    embeds = current_embeds.view(batch_size, num_generations, hidden_dim)
    
    # Default thinking mask: all paths active
    if thinking_mask is None:
        thinking_mask = [True] * (batch_size * num_generations)
    
    # Process each question independently
    blended_list = []
    for b in range(batch_size):
        batch_embeds = embeds[b]  # (N, hidden_dim)
        batch_thinking = thinking_mask[b * num_generations : (b + 1) * num_generations]
        
        blended = apply_cross_path_interaction(
            current_embeds=batch_embeds,
            is_thinking=batch_thinking,
            lambda_blend=lambda_blend,
            temperature=temperature,
        )
        blended_list.append(blended)
    
    # Reshape back to (batch_size * N, hidden_dim)
    return torch.cat(blended_list, dim=0)


def apply_cross_path_interaction(
    current_embeds: torch.Tensor,  # (N, hidden_dim)
    is_thinking: List[bool],
    lambda_blend: float,
    temperature: float,
) -> torch.Tensor:
    """
    Apply cross-path interaction to embeddings from multiple paths.
    
    Computes similarity between paths and blends each path's embedding with
    contextual information from similar paths.
    
    Args:
        current_embeds: Hidden states for N paths (N, hidden_dim)
        is_thinking: Boolean mask indicating which paths are still active
        lambda_blend: Blending coefficient (0 = no blending, 1 = full contextual)
        temperature: Temperature for attention softmax (lower = sharper)
    
    Returns:
        Blended embeddings (N, hidden_dim)
    """
    N = current_embeds.shape[0]
    hidden_dim = current_embeds.shape[1]
    device = current_embeds.device
    
    # Compute pairwise cosine similarity between embeddings
    # Normalize embeddings: (N, hidden_dim)
    norm_embeds = current_embeds / (current_embeds.norm(dim=1, keepdim=True) + 1e-8)
    
    # Similarity matrix: (N, N)
    similarity_matrix = torch.mm(norm_embeds, norm_embeds.t())
    
    # Mask diagonal (no self-interaction)
    mask = torch.eye(N, dtype=torch.bool, device=device)
    similarity_matrix.masked_fill_(mask, float('-inf'))
    
    # Mask inactive paths (those that exited thinking mode)
    thinking_mask = torch.tensor(is_thinking, device=device, dtype=torch.bool)
    # If path i is not thinking, it shouldn't contribute or receive
    active_mask = thinking_mask.unsqueeze(0) & thinking_mask.unsqueeze(1)  # (N, N)
    active_mask = active_mask & ~mask  # Exclude diagonal
    similarity_matrix = similarity_matrix.masked_fill(~active_mask, float('-inf'))
    
    # Temperature-scaled softmax to get attention weights
    # Handle rows where all values are -inf (no valid attention targets)
    scaled_similarity = similarity_matrix / temperature
    all_inf_mask = (similarity_matrix == float('-inf')).all(dim=1)  # (N,)
    
    # Compute softmax
    attention_weights = F.softmax(scaled_similarity, dim=1)  # (N, N)
    
    # Replace NaN rows (from all -inf) with zeros (no attention)
    if all_inf_mask.any():
        attention_weights[all_inf_mask] = 0.0
    
    # Compute contextual embeddings: c_i = Σ_j A_ij * e_j
    contextual_embeds = torch.mm(attention_weights, current_embeds)  # (N, hidden_dim)
    
    # Blend: h_i = (1 - λ) * e_i + λ * c_i
    blended_embeds = (1 - lambda_blend) * current_embeds + lambda_blend * contextual_embeds
    
    # Only apply blending to paths that are still in thinking mode
    # Keep original for non-thinking paths
    result = torch.where(thinking_mask.unsqueeze(1), blended_embeds, current_embeds)
    
    return result
