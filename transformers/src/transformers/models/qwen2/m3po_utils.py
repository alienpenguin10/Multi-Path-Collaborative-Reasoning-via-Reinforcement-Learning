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
from torch.utils.checkpoint import checkpoint as grad_checkpoint
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
    gating_function=None,                # Optional: Alternative gating function (BaseM3POGating)
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
        gating_function: Optional BaseM3POGating instance for alternative similarity computation

    Returns:
        attention_weights: (N, N) attention matrix for cross-path blending
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    N = output_distributions.shape[0]
    device = output_distributions.device

    # Convert thinking_mask to tensor
    thinking_tensor = torch.tensor(thinking_mask, device=device, dtype=torch.bool)

    # Compute attention weights
    if gating_function is not None:
        # Use alternative gating function (computes similarity + attention in one call)
        similarity_matrix = gating_function.compute_similarity_matrix(
            output_distributions=output_distributions,
            hidden_states=None,
        )
        attention_weights, _ = gating_function.compute_attention_weights(
            similarity_matrix=similarity_matrix,
            thinking_mask=thinking_tensor,
            mask_diagonal=True,
        )
    else:
        # Baseline: Equation 3 - Compute pairwise cosine similarity from OUTPUT DISTRIBUTIONS
        # Normalize distributions
        norm_dists = output_distributions / (output_distributions.norm(dim=1, keepdim=True) + 1e-8)

        # Similarity matrix: S_ij = (p_i · p_j) / (||p_i|| ||p_j||)
        similarity_matrix = torch.mm(norm_dists, norm_dists.t())  # (N, N)

        # Equation 4: Mask diagonal (no self-reinforcement)
        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(diag_mask, float('-inf'))

        # Mask inactive paths (those that exited thinking mode)
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
    contextual_embeddings = torch.mm(attention_weights.to(token_embeddings.dtype), token_embeddings)  # (N, hidden_dim)

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
    gating_function=None,                # Optional: Alternative gating function (BaseM3POGating)
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
            gating_function=gating_function,
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


def apply_m3po_to_logits(
    logits: torch.Tensor,          # (N, seq_len, vocab_size) — one question group
    num_generations: int,          # N paths
    lambda_blend: float = 0.1,
    temperature: float = 0.1,
    gating_function=None,          # Optional BaseM3POGating
    completion_mask=None,          # Optional (N, seq_len) binary mask
) -> torch.Tensor:
    """
    Apply M3PO cross-path logit blending for a single question group during loss computation.

    This creates a differentiable path from loss → log_probs → blended_logits → attention_weights
    → similarity_matrix → gating_parameters, enabling gradient flow to learnable gating params.

    Blending formula (per position t):
        1. p_i = softmax(logits_i[t])                     # output distributions
        2. S_ij = gating_fn.compute_similarity_matrix(p)   # differentiable similarity
        3. A_ij = softmax(S_ij / T)                        # attention weights
        4. blended_i[t] = (1-λ)*logits_i[t] + λ*Σ_j A_ij*logits_j[t]  # logit blending

    Args:
        logits: Raw logits from model (N, seq_len, vocab_size) for one question group
        num_generations: Number of paths N (should equal logits.shape[0])
        lambda_blend: Blending coefficient (0 = no blend, 1 = full contextual)
        temperature: Temperature for attention softmax
        gating_function: Optional BaseM3POGating instance for custom similarity computation
        completion_mask: Optional (N, seq_len) binary mask, 1 for valid positions

    Returns:
        blended_logits: (N, seq_len, vocab_size) with cross-path blending applied
    """
    debug = os.environ.get('M3PO_DEBUG', '0') == '1'

    N, seq_len, vocab_size = logits.shape
    device = logits.device

    if N != num_generations:
        raise ValueError(f"[M3PO] logits batch dim {N} != num_generations {num_generations}")

    if gating_function is not None:
        # Custom gating with gradient checkpointing to avoid storing 512 autograd nodes.
        # Process positions in chunks; each chunk is recomputed during backward.
        thinking_mask = torch.ones(N, dtype=torch.bool, device=device)
        CHUNK_SIZE = 32  # positions per checkpoint segment
        gating_params = list(gating_function.parameters())

        def _blend_chunk(logits_chunk, thinking_mask, lambda_blend_t, *extra_params):
            """Blend a chunk of positions. Called inside grad_checkpoint so intermediates
            are recomputed during backward instead of stored."""
            chunk_len = logits_chunk.shape[1]
            blended_chunk = logits_chunk.clone()
            for t in range(chunk_len):
                p = F.softmax(logits_chunk[:, t, :], dim=-1)
                similarity_matrix = gating_function.compute_similarity_matrix(
                    output_distributions=p, hidden_states=None,
                )
                attention_weights, _ = gating_function.compute_attention_weights(
                    similarity_matrix=similarity_matrix,
                    thinking_mask=thinking_mask,
                    mask_diagonal=True,
                )
                contextual = torch.mm(
                    attention_weights.to(logits_chunk.dtype),
                    logits_chunk[:, t, :]
                )
                blended_chunk[:, t, :] = (1 - lambda_blend_t) * logits_chunk[:, t, :] + lambda_blend_t * contextual
            return blended_chunk

        # lambda_blend as a tensor so grad_checkpoint can track it
        lambda_blend_t = torch.tensor(lambda_blend, device=device, dtype=logits.dtype)

        blended_chunks = []
        for start in range(0, seq_len, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, seq_len)

            # Skip fully-masked chunks
            if completion_mask is not None and completion_mask[:, start:end].sum() == 0:
                blended_chunks.append(logits[:, start:end, :])
                continue

            chunk = logits[:, start:end, :].contiguous()
            blended_chunk = grad_checkpoint(
                _blend_chunk, chunk, thinking_mask, lambda_blend_t, *gating_params,
                use_reentrant=False,
            )
            blended_chunks.append(blended_chunk)

        blended = torch.cat(blended_chunks, dim=1)

        if debug:
            diff = (blended - logits).abs().mean().item()
            print(f"[M3PO LOGITS] Custom gating, N={N}, seq_len={seq_len}, mean_change={diff:.6f}")

        return blended

    else:
        # Baseline cosine similarity: vectorize across positions using torch.bmm
        # Reshape to (seq_len, N, vocab_size) for batched computation
        logits_t = logits.permute(1, 0, 2)  # (seq_len, N, vocab_size)

        # 1. Output distributions
        p_t = F.softmax(logits_t, dim=-1)  # (seq_len, N, vocab_size)

        # 2. Cosine similarity: normalize then batched matmul
        norm_p = p_t / (p_t.norm(dim=-1, keepdim=True) + 1e-8)  # (seq_len, N, vocab_size)
        sim = torch.bmm(norm_p, norm_p.transpose(1, 2))  # (seq_len, N, N)

        # 3. Mask diagonal and apply temperature-scaled softmax
        diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # (1, N, N)
        sim = sim.masked_fill(diag_mask, float('-inf'))
        scaled_sim = sim / temperature
        attention = F.softmax(scaled_sim, dim=-1)  # (seq_len, N, N)

        # Handle all-inf rows (single path case)
        all_inf = (sim == float('-inf')).all(dim=-1, keepdim=True)  # (seq_len, N, 1)
        attention = attention.masked_fill(all_inf, 0.0)

        # 4. Blend logits
        contextual_t = torch.bmm(
            attention.to(logits_t.dtype),
            logits_t
        )  # (seq_len, N, vocab_size)
        blended_t = (1 - lambda_blend) * logits_t + lambda_blend * contextual_t

        # Reshape back to (N, seq_len, vocab_size)
        blended = blended_t.permute(1, 0, 2)

        if debug:
            diff = (blended - logits).abs().mean().item()
            print(f"[M3PO LOGITS] Baseline cosine, N={N}, seq_len={seq_len}, mean_change={diff:.6f}")

        return blended


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
    gating_function=None,                # Optional: Alternative gating function (BaseM3POGating)
) -> torch.Tensor:
    """
    Generate with M3PO using KV-cached decoding.

    Prefills the KV cache with prompt embeddings, then decodes one token at a time
    with M3PO-blended embeddings. Past KV states are unaffected by current-step
    blending, so caching is mathematically equivalent to full recomputation.
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

    if debug:
        print(f"[M3PO GEN v2] Starting: batch={batch_size}, N={num_generations}, total={total_paths}")

    # Disable gradient checkpointing for generation — it forces use_cache=False
    was_training = model.training
    model.eval()

    # Prefill: process entire prompt, initialize KV cache
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=expanded_attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    current_logits = outputs.logits[:, -1, :]  # (total_paths, vocab_size)
    del inputs_embeds, outputs  # free prompt embeddings, now encoded in KV cache

    for step in range(max_new_tokens):
        # Temperature for sampling
        if temperature_sampling != 1.0:
            sample_logits = current_logits / temperature_sampling
        else:
            sample_logits = current_logits

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

        # Apply M3PO if any paths still thinking
        if any(thinking_mask) and lambda_blend > 0:
            # Compute attention from output distributions
            blended_embeds = apply_m3po_step(
                logits=current_logits,  # Original logits for similarity
                embed_tokens=embed_tokens,
                sampled_tokens=next_tokens,
                config=config,
                thinking_mask=thinking_mask,
                gating_function=gating_function,
            )
            # Shape: (total_paths, 1, hidden_dim)
        else:
            next_embeds = embed_tokens(next_tokens)  # (total_paths, hidden_dim)
            blended_embeds = next_embeds.unsqueeze(1)

        # Update attention mask (must include all cached + new positions)
        expanded_attention_mask = torch.cat([
            expanded_attention_mask,
            torch.ones(total_paths, 1, device=device, dtype=expanded_attention_mask.dtype)
        ], dim=1)

        # Single-token decode with KV cache
        outputs = model(
            inputs_embeds=blended_embeds,           # (total_paths, 1, hidden_dim)
            attention_mask=expanded_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        current_logits = outputs.logits[:, -1, :]  # (total_paths, vocab_size)

        # Track generated tokens
        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=1)

        if finished.all():
            if debug:
                print(f"[M3PO GEN v2] All done at step {step}")
            break

        if debug and step < 3:
            active = sum(thinking_mask)
            print(f"[M3PO GEN v2] Step {step}: {active}/{total_paths} thinking")

    # Restore training mode for subsequent loss computation
    if was_training:
        model.train()

    return generated_ids


__all__ = [
    "M3POConfig",
    "compute_cross_path_attention",
    "blend_token_embeddings",
    "apply_m3po_step",
    "apply_m3po_to_logits",
    "generate_with_m3po",
]
