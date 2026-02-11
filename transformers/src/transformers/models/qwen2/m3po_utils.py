"""
M3PO (Multi-Path Perception Policy Optimization) Utilities for Qwen2

Implements the collaborative learning mechanism from the M3PO paper:
"Multi-Path Collaborative Reasoning via Reinforcement Learning"

Key equations from the paper:
- Equation 2: h̄_i = (1 - λ) * e_i + λ * c_i  (hybrid thinking embedding)
- Equation 3: S_ij = (p_i · p_j) / (||p_i|| ||p_j||)  (similarity from OUTPUT DISTRIBUTIONS)
- Equation 5: A_ij = softmax(S_ij / T)  (attention weights)
- Equation 6: c_i = Σ A_ij * e_j  (contextual embedding from TOKEN EMBEDDINGS)

CRITICAL: The paper explicitly states (Section 4.3, Figure 4):
"The hidden states approach remains zero reward, primarily due to the
distributional discrepancy between hidden states and the pretrained
embedding space, leading to incompatibility and performance degradation."

Therefore, this implementation:
1. Computes similarity from OUTPUT PROBABILITY DISTRIBUTIONS (not hidden states)
2. Blends TOKEN EMBEDDINGS (not hidden states)
3. Operates BETWEEN generation steps (not inside transformer layers)
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class M3POConfig:
    """Configuration for M3PO cross-path interaction."""
    num_generations: int = 4          # N paths per question
    lambda_blend: float = 0.1         # Blending coefficient (paper default: 0.1)
    temperature: float = 0.1          # Attention temperature (paper default: 0.1)
    batch_size: int = 1               # Number of questions in batch


def compute_cross_path_attention(
    output_distributions: torch.Tensor,  # (N, vocab_size) - probability distributions
    thinking_mask: List[bool],           # Which paths are still in thinking mode
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute cross-path attention weights from output probability distributions.

    This follows Equations 3-5 from the M3PO paper:
    - S_ij = cosine_similarity(p_i, p_j)  [Eq. 3]
    - Mask diagonal (no self-interaction)  [Eq. 4]
    - A_ij = softmax(S_ij / T)  [Eq. 5]

    Args:
        output_distributions: Softmax probabilities for each path (N, vocab_size)
        thinking_mask: Boolean mask, True = path is still in thinking mode
        temperature: Temperature for softmax (lower = sharper attention)

    Returns:
        attention_weights: (N, N) attention matrix for cross-path blending
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    N = output_distributions.shape[0]
    device = output_distributions.device

    # Equation 3: Compute pairwise cosine similarity from OUTPUT DISTRIBUTIONS
    # Normalize distributions
    norm_dists = output_distributions / (output_distributions.norm(dim=1, keepdim=True) + 1e-8)

    # Similarity matrix: S_ij = (p_i · p_j) / (||p_i|| ||p_j||)
    similarity_matrix = torch.mm(norm_dists, norm_dists.t())  # (N, N)

    # Equation 4: Mask diagonal (no self-reinforcement)
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill(diag_mask, float('-inf'))

    # Mask inactive paths (those that exited thinking mode)
    thinking_tensor = torch.tensor(thinking_mask, device=device, dtype=torch.bool)

    # Path i shouldn't receive from inactive paths, inactive paths shouldn't contribute
    active_mask = thinking_tensor.unsqueeze(0) & thinking_tensor.unsqueeze(1)
    active_mask = active_mask & ~diag_mask  # Also exclude diagonal
    similarity_matrix = similarity_matrix.masked_fill(~active_mask, float('-inf'))

    # Equation 5: Temperature-scaled softmax for attention weights
    scaled_sim = similarity_matrix / temperature

    # Handle rows where all values are -inf (no valid attention targets)
    all_inf_mask = (similarity_matrix == float('-inf')).all(dim=1)

    attention_weights = F.softmax(scaled_sim, dim=1)  # (N, N)

    # Replace NaN rows (from all -inf) with zeros
    if all_inf_mask.any():
        attention_weights[all_inf_mask] = 0.0

    if debug:
        active_count = thinking_tensor.sum().item()
        print(f"[M3PO] Computing attention: {active_count}/{N} active paths, temp={temperature}")
        if N <= 8:
            print(f"[M3PO] Similarity matrix:\n{similarity_matrix}")
            print(f"[M3PO] Attention weights:\n{attention_weights}")

    return attention_weights


def blend_token_embeddings(
    token_embeddings: torch.Tensor,      # (N, hidden_dim) - embeddings of sampled tokens
    attention_weights: torch.Tensor,     # (N, N) - cross-path attention
    thinking_mask: List[bool],
    lambda_blend: float = 0.1,
) -> torch.Tensor:
    """
    Blend token embeddings using cross-path attention.

    This follows Equations 2 and 6 from the M3PO paper:
    - c_i = Σ_j A_ij * e_j  [Eq. 6] (contextual embedding)
    - h̄_i = (1 - λ) * e_i + λ * c_i  [Eq. 2] (hybrid embedding)

    Args:
        token_embeddings: Embeddings of sampled tokens (N, hidden_dim)
        attention_weights: Cross-path attention matrix (N, N)
        thinking_mask: Which paths are still in thinking mode
        lambda_blend: Blending coefficient (0 = no blend, 1 = full contextual)

    Returns:
        blended_embeddings: (N, hidden_dim) hybrid embeddings for next step
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    N = token_embeddings.shape[0]
    device = token_embeddings.device

    # Equation 6: Compute contextual embeddings c_i = Σ_j A_ij * e_j
    contextual_embeddings = torch.mm(attention_weights, token_embeddings)  # (N, hidden_dim)

    # Equation 2: Blend h̄_i = (1 - λ) * e_i + λ * c_i
    blended = (1 - lambda_blend) * token_embeddings + lambda_blend * contextual_embeddings

    # Only apply blending to paths still in thinking mode
    thinking_tensor = torch.tensor(thinking_mask, device=device, dtype=torch.bool)
    result = torch.where(thinking_tensor.unsqueeze(1), blended, token_embeddings)

    if debug:
        diff = (result - token_embeddings).abs().mean().item()
        print(f"[M3PO] Blending with lambda={lambda_blend}, mean change={diff:.6f}")

    return result


def apply_m3po_step(
    logits: torch.Tensor,                # (batch_size * N, vocab_size) - model output logits
    embed_tokens: torch.nn.Embedding,    # Token embedding layer
    sampled_tokens: torch.Tensor,        # (batch_size * N,) - sampled token IDs
    config: M3POConfig,
    thinking_mask: Optional[List[bool]] = None,
) -> torch.Tensor:
    """
    Apply one step of M3PO cross-path interaction.

    This is the main entry point called during generation. It:
    1. Computes output probability distributions from logits
    2. Computes cross-path attention from distributions (Equations 3-5)
    3. Gets token embeddings for sampled tokens
    4. Blends embeddings using cross-path attention (Equations 2, 6)

    Args:
        logits: Raw logits from model (batch_size * N, vocab_size)
        embed_tokens: Token embedding layer to get embeddings
        sampled_tokens: Token IDs that were sampled (batch_size * N,)
        config: M3PO configuration
        thinking_mask: Which paths are still in thinking mode

    Returns:
        blended_embeddings: (batch_size * N, 1, hidden_dim) ready for next step
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    total_paths = logits.shape[0]
    batch_size = config.batch_size
    N = config.num_generations

    if total_paths != batch_size * N:
        raise ValueError(f"[M3PO] logits batch size {total_paths} != batch_size*N = {batch_size}*{N}")

    # Default: all paths in thinking mode
    if thinking_mask is None:
        thinking_mask = [True] * total_paths

    # Compute output probability distributions (softmax of logits)
    output_distributions = F.softmax(logits, dim=-1)  # (batch_size * N, vocab_size)

    # Get token embeddings for sampled tokens
    token_embeddings = embed_tokens(sampled_tokens)  # (batch_size * N, hidden_dim)

    if debug:
        print(f"[M3PO STEP] batch_size={batch_size}, N={N}, total_paths={total_paths}")
        print(f"[M3PO STEP] logits shape: {logits.shape}, embeddings shape: {token_embeddings.shape}")

    # Process each question's paths independently
    blended_list = []
    for b in range(batch_size):
        start_idx = b * N
        end_idx = (b + 1) * N

        # Get this question's distributions and embeddings
        batch_dists = output_distributions[start_idx:end_idx]  # (N, vocab_size)
        batch_embeds = token_embeddings[start_idx:end_idx]     # (N, hidden_dim)
        batch_thinking = thinking_mask[start_idx:end_idx]

        # Compute cross-path attention from output distributions
        attention = compute_cross_path_attention(
            output_distributions=batch_dists,
            thinking_mask=batch_thinking,
            temperature=config.temperature,
        )

        # Blend token embeddings
        blended = blend_token_embeddings(
            token_embeddings=batch_embeds,
            attention_weights=attention,
            thinking_mask=batch_thinking,
            lambda_blend=config.lambda_blend,
        )

        blended_list.append(blended)

    # Concatenate all batches: (batch_size * N, hidden_dim)
    result = torch.cat(blended_list, dim=0)

    # Add sequence dimension for transformer input: (batch_size * N, 1, hidden_dim)
    return result.unsqueeze(1)


@torch.no_grad()
def generate_with_m3po(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 512,
    num_generations: int = 4,
    lambda_blend: float = 0.1,
    temperature_m3po: float = 0.1,
    temperature_sampling: float = 1.0,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    thinking_end_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Generate with M3PO - cleaner implementation.

    Uses inputs_embeds for the entire generation to allow M3PO blending.
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    device = input_ids.device
    batch_size = input_ids.shape[0]
    total_paths = batch_size * num_generations

    embed_tokens = model.model.embed_tokens

    config = M3POConfig(
        num_generations=num_generations,
        lambda_blend=lambda_blend,
        temperature=temperature_m3po,
        batch_size=batch_size,
    )

    # Expand for parallel paths
    expanded_input_ids = input_ids.repeat_interleave(num_generations, dim=0)
    expanded_attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)

    # Convert to embeddings for the initial prompt
    inputs_embeds = embed_tokens(expanded_input_ids)  # (total_paths, seq_len, hidden_dim)

    # Track state
    generated_ids = expanded_input_ids.clone()
    thinking_mask = [True] * total_paths
    finished = torch.zeros(total_paths, dtype=torch.bool, device=device)

    # No KV cache when using inputs_embeds (simpler but slower)
    # For production, you'd want to optimize this

    if debug:
        print(f"[M3PO GEN v2] Starting: batch={batch_size}, N={num_generations}, total={total_paths}")

    for step in range(max_new_tokens):
        # Forward pass with embeddings
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=expanded_attention_mask,
            use_cache=False,  # Disable cache for simplicity with embeddings
        )

        # Get logits for last position
        logits = outputs.logits[:, -1, :]  # (total_paths, vocab_size)

        # Temperature for sampling
        if temperature_sampling != 1.0:
            sample_logits = logits / temperature_sampling
        else:
            sample_logits = logits

        # Sample tokens
        probs = F.softmax(sample_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Check EOS
        if eos_token_id is not None:
            finished = finished | (next_tokens == eos_token_id)

        # Update thinking mask
        if thinking_end_tokens:
            for i, tok in enumerate(next_tokens.tolist()):
                if tok in thinking_end_tokens:
                    thinking_mask[i] = False

        # Get token embeddings
        next_embeds = embed_tokens(next_tokens)  # (total_paths, hidden_dim)

        # Apply M3PO if any paths still thinking
        if any(thinking_mask) and lambda_blend > 0:
            # Compute attention from output distributions
            blended_embeds = apply_m3po_step(
                logits=logits,  # Original logits for similarity
                embed_tokens=embed_tokens,
                sampled_tokens=next_tokens,
                config=config,
                thinking_mask=thinking_mask,
            )
            # Shape: (total_paths, 1, hidden_dim)
        else:
            blended_embeds = next_embeds.unsqueeze(1)

        # Append to sequence embeddings
        inputs_embeds = torch.cat([inputs_embeds, blended_embeds], dim=1)

        # Update attention mask
        expanded_attention_mask = torch.cat([
            expanded_attention_mask,
            torch.ones(total_paths, 1, device=device, dtype=expanded_attention_mask.dtype)
        ], dim=1)

        # Track generated tokens
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=1)

        if finished.all():
            if debug:
                print(f"[M3PO GEN v2] All done at step {step}")
            break

        if debug and step < 3:
            active = sum(thinking_mask)
            print(f"[M3PO GEN v2] Step {step}: {active}/{total_paths} thinking, seq_len={inputs_embeds.shape[1]}")

    return generated_ids


__all__ = [
    "M3POConfig",
    "compute_cross_path_attention",
    "blend_token_embeddings",
    "apply_m3po_step",
    "generate_with_m3po",
]
