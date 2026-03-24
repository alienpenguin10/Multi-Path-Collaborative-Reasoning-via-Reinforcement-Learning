"""
Part 1: Basic Setup and Imports
"""
# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import os
import json
import math
import numpy as np
import wandb
import sys

# Must be set before any CUDA operations to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Must be set before CUDA init for cuBLAS determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# Force NCCL to use deterministic ring algorithm (tree/default can vary reduction order)
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_PROTO"] = "Simple"

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
import torch.distributed as dist
torch.backends.cuda.enable_flash_sdp(False)  # Flash SDP is non-deterministic in backward pass
torch.backends.cuda.enable_mem_efficient_sdp(False)  # mem_efficient backend requires stride % 4 == 0
torch.backends.cuda.enable_math_sdp(True)  # Math SDP is deterministic
# Disable reduced-precision reductions — bf16 accumulation order can vary
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torch.nn.utils.rnn import pad_sequence
import datetime
import bitsandbytes as bnb

# Hugging Face libraries for transformer models
sys.path.insert(0, os.path.abspath("transformers/src"))
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["M3PO_DEBUG"] = "-1"  # Disable M3PO debug prints (set to "1" to enable)

from dotenv import load_dotenv
load_dotenv()
# Call the function to set random seed for reproducibility
from utils import set_random_seed, get_next_trial_number
BASE_SEED = 42
set_random_seed(BASE_SEED)


# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")

"""
Part 2: Data Formatting and Answer Extraction
"""
from utils import extract_answer_from_model_output

"""
Part 3: Dataset Preparation
"""
from utils import prepare_dataset


"""
Part 4: Evaluation Functions
"""
from utils import evaluate_model

"""
Part 5: Reward Functions
"""
from utils import combined_reward

"""
Part 6: DDP Helpers
"""
def setup_ddp():
    """Initialize distributed training. Called by each process spawned by torchrun."""
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    """Tear down distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Returns True if this is rank 0 (or if DDP is not active)."""
    return not dist.is_initialized() or dist.get_rank() == 0

"""
Part 7: DDP GRPO From Scratch
In this section, we implement all the building blocks of the GRPO algorithm from scratch. 
The implementation assumes that the machine running the code has at least 2 GPUs. 
We use PyTorch's DistributedDataParallel (DDP) to distribute the policy model across GPUs, one process per GPU. Each process trains on different data and gradients are synchronized via ring-allreduce.
"""
def selective_log_softmax(logits, input_ids, chunk_size=2):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Uses chunked processing to avoid OOM errors with large batches.

    Args:
        logits (torch.Tensor): The raw logits output from the model. Shape: (batch, seq_len, vocab)
        input_ids (torch.Tensor): The token IDs for which we want the log probabilities. Shape: (batch, seq_len)
        chunk_size (int): Number of sequences to process at once to avoid OOM.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens. Shape: (batch, seq_len)

    Explanation:
        1. Processes in chunks to manage GPU memory.
        2. Applies log softmax to convert logits to log probabilities over the vocabulary.
        3. Uses gather to extract only the log probabilities corresponding to the input_ids.
        4. Removes the extra dimension to match the original shape of input_ids.
    """
    batch_size = logits.shape[0]

    # If batch is small enough, process all at once
    if batch_size <= chunk_size:
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    # Process in chunks to avoid OOM
    results = []
    for i in range(0, batch_size, chunk_size):
        chunk_logits = logits[i:i+chunk_size]
        chunk_ids = input_ids[i:i+chunk_size]
        chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
        chunk_result = chunk_log_probs.gather(dim=-1, index=chunk_ids.unsqueeze(-1)).squeeze(-1)
        results.append(chunk_result)
        # Free memory
        del chunk_log_probs

    return torch.cat(results, dim=0)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep, chunk_size=2):
    """
    Computes the log probabilities for a batch of tokens.

    Uses chunked processing to avoid OOM errors with large batches.

    Args:
        model: The language model.
        input_ids (torch.Tensor): Token IDs for input sequences.
        attention_mask (torch.Tensor): Attention mask for input sequences.
        logits_to_keep (int): Number of tokens to keep from the end of the sequence.
        chunk_size (int): Number of sequences to process at once to avoid OOM.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Gets logits from the model for the input sequence (in chunks to save memory).
        2. Selects logits for all tokens except the last one (as we predict next tokens).
        3. Selects only the last 'logits_to_keep' tokens from both logits and input_ids.
        4. Computes log probabilities for these tokens using selective_log_softmax.
    """
    batch_size = input_ids.shape[0]

    # If batch is small enough, process all at once
    if batch_size <= chunk_size:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits[:, :-1, :]
        ids_to_keep = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        return selective_log_softmax(logits, ids_to_keep)

    # Process in chunks to avoid OOM
    results = []
    for i in range(0, batch_size, chunk_size):
        chunk_ids = input_ids[i:i+chunk_size]
        chunk_mask = attention_mask[i:i+chunk_size]

        chunk_logits = model(input_ids=chunk_ids, attention_mask=chunk_mask, logits_to_keep=logits_to_keep + 1).logits[:, :-1, :]
        chunk_ids_to_keep = chunk_ids[:, -logits_to_keep:]
        chunk_logits = chunk_logits[:, -logits_to_keep:, :]

        # Compute log probs for this chunk (no need for extra chunking since already small)
        log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
        chunk_result = log_probs.gather(dim=-1, index=chunk_ids_to_keep.unsqueeze(-1)).squeeze(-1)
        results.append(chunk_result)

        # Free memory
        del chunk_logits, log_probs

    return torch.cat(results, dim=0)

def compute_log_probs_with_m3po(model, input_ids, attention_mask, logits_to_keep,
                                batch_size, num_generations,
                                lambda_blend=0.1, temperature_m3po=0.1,
                                gating_function=None, completion_mask=None):
    """
    Computes log probabilities with M3PO cross-path logit blending.

    Unlike compute_log_probs() which chunks by chunk_size=2, this function chunks
    by question group (N paths together) because M3PO needs all paths for cross-path
    interaction within each group.

    This creates a differentiable path:
        loss → log_probs → blended_logits → attention_weights → similarity_matrix → gating_parameters

    Args:
        model: The language model.
        input_ids: Token IDs for input sequences (batch_size * N, seq_len).
        attention_mask: Attention mask (batch_size * N, seq_len).
        logits_to_keep: Number of tokens to keep from the end.
        batch_size: Number of questions in the batch.
        num_generations: Number of paths per question (N).
        lambda_blend: M3PO blending coefficient.
        temperature_m3po: M3PO attention temperature.
        gating_function: Optional BaseM3POGating instance.
        completion_mask: Optional (batch_size * N, logits_to_keep) binary mask.

    Returns:
        torch.Tensor: Log probabilities with M3PO blending applied.
    """
    from transformers.models.qwen2.m3po_utils import apply_m3po_to_logits

    total_sequences = input_ids.shape[0]
    N = num_generations

    results = []
    for b in range(batch_size):
        start_idx = b * N
        end_idx = (b + 1) * N

        # Get this question group's data
        group_ids = input_ids[start_idx:end_idx]
        group_mask = attention_mask[start_idx:end_idx]

        # Forward pass for this group — only project last logits_to_keep+1 positions through lm_head
        group_logits = model(
            input_ids=group_ids, attention_mask=group_mask,
            logits_to_keep=logits_to_keep + 1,
        ).logits[:, :-1, :]  # (N, logits_to_keep, vocab_size)
        group_token_ids = group_ids[:, -logits_to_keep:]  # (N, logits_to_keep)

        # Get completion mask for this group if provided
        group_completion_mask = None
        if completion_mask is not None:
            group_completion_mask = completion_mask[start_idx:end_idx]  # (N, logits_to_keep)

        # Apply M3PO logit blending (differentiable)
        blended_logits = apply_m3po_to_logits(
            logits=group_logits,
            num_generations=N,
            lambda_blend=lambda_blend,
            temperature=temperature_m3po,
            gating_function=gating_function,
            completion_mask=group_completion_mask,
        )

        # Compute log probs from blended logits
        log_probs = nn.functional.log_softmax(blended_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=group_token_ids.unsqueeze(-1)).squeeze(-1)
        results.append(token_log_probs)

        # Free intermediates between groups (stays in PyTorch's cache, not released to CUDA)
        del group_logits, blended_logits, log_probs

    return torch.cat(results, dim=0)


def detect_thinking_phase_end(tokenizer, completion_ids):
    """
    Check if any path has exited the thinking phase.

    M3PO should only be applied during the "thinking" phase, not during answer generation.
    This function detects when paths have generated </reasoning> or <answer> tokens.

    Args:
        tokenizer: The tokenizer for encoding marker tokens
        completion_ids: Generated token IDs (batch_size * N, seq_len)

    Returns:
        List of booleans: True = still thinking, False = in answer phase
    """
    total_sequences = completion_ids.shape[0]
    thinking_mask = []

    # Get token IDs for end-of-thinking markers
    # Note: These may be multi-token sequences depending on tokenizer
    try:
        end_reasoning_tokens = tokenizer.encode("</reasoning>", add_special_tokens=False)
        answer_start_tokens = tokenizer.encode("<answer>", add_special_tokens=False)
    except Exception:
        # If encoding fails, assume all paths are still thinking
        return [True] * total_sequences

    for i in range(total_sequences):
        seq = completion_ids[i].tolist()
        still_thinking = True

        # Check for </reasoning> marker
        if len(end_reasoning_tokens) > 0:
            for j in range(len(seq) - len(end_reasoning_tokens) + 1):
                if seq[j:j + len(end_reasoning_tokens)] == end_reasoning_tokens:
                    still_thinking = False
                    break

        # Check for <answer> marker if still thinking
        if still_thinking and len(answer_start_tokens) > 0:
            for j in range(len(seq) - len(answer_start_tokens) + 1):
                if seq[j:j + len(answer_start_tokens)] == answer_start_tokens:
                    still_thinking = False
                    break

        thinking_mask.append(still_thinking)

    return thinking_mask

def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.

    Args:
        completion_ids (torch.Tensor): Token IDs of the generated completions.
        eos_token_id (int): The ID of the end-of-sequence token.

    Returns:
        torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

    Explanation:
        1. Identifies positions where EOS tokens occur in each sequence.
        2. Finds the index of the first EOS token in each sequence.
        3. Creates a mask where positions before and including the first EOS are 1, others are 0.
        4. If no EOS token is found in a sequence, all positions are set to 1.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32,
                         lambda_blend=0.1, temperature_m3po=0.1, use_m3po=True, gating_function=None):
    """
    Generates multiple completions for each prompt with optional M3PO cross-path interaction.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (list): List of text prompts.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate.
        lambda_blend (float): M3PO blending coefficient (0 = no blending, 1 = full contextual).
        temperature_m3po (float): M3PO attention temperature (lower = sharper).
        use_m3po (bool): Whether to enable M3PO cross-path interaction.

    Returns:
        tuple: Containing prompt IDs, prompt mask, completion IDs, and completion mask.

    Explanation:
        1. Encodes the prompts and moves them to the appropriate device.
        2. If M3PO is enabled, uses custom generation that applies cross-path blending
           in the embedding space (following the M3PO paper).
        3. Extracts the completion IDs (excluding the prompt tokens).
        4. Creates a mask for the completions using create_completion_mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    # print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)

    if use_m3po:
        # Use the new M3PO generation that correctly applies cross-path interaction
        # in the embedding space (as per the paper)
        from transformers.models.qwen2.m3po_utils import generate_with_m3po

        # Get token IDs that signal end of thinking phase
        # The paper says M3PO should only apply during "thinking", not during answer generation
        thinking_end_tokens = []
        try:
            # </reasoning> and <answer> tokens signal end of thinking
            thinking_end_tokens.extend(tokenizer.encode("</reasoning>", add_special_tokens=False))
            thinking_end_tokens.extend(tokenizer.encode("<answer>", add_special_tokens=False))
        except Exception:
            pass

        # print(f"[M3PO] Enabled with lambda={lambda_blend}, temp={temperature_m3po}, paths={prompt_ids.size(0) * num_generations}")
        # print(f"[M3PO] Thinking end tokens: {thinking_end_tokens}")

        # generate_with_m3po handles the expansion internally
        outputs = generate_with_m3po(
            model=model,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            num_generations=num_generations,
            lambda_blend=lambda_blend,
            temperature_m3po=temperature_m3po,
            temperature_sampling=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            thinking_end_tokens=thinking_end_tokens if thinking_end_tokens else None,
            gating_function=gating_function,
        )

        # print("[M3PO] Generation complete")

        # The output already has expanded batch size (batch_size * num_generations)
        # Update prompt_ids and prompt_mask to match
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    else:
        # Standard generation without M3PO
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

        model.eval()  # Prevent check_model_inputs from forcing use_cache=False during generation
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model.train()  # Restore training mode for subsequent compute_log_probs

    # print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length,
                          lambda_blend=0.1, temperature_m3po=0.1, use_m3po=True, gating_function=None,
                          compiled_model=None):
    """
    Generates data for GRPO rollouts including completions and log probabilities.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        tokenizer: The tokenizer for encoding and decoding text.
        batch_samples (list): Batch of training samples.
        num_generations (int): Number of completions to generate per sample.
        max_completion_length (int): Maximum completion length.
        lambda_blend (float): M3PO blending coefficient.
        temperature_m3po (float): M3PO attention temperature.
        use_m3po (bool): Whether to enable M3PO cross-path interaction.

    Returns:
        dict: Dictionary containing all data needed for GRPO updates.

    Explanation:
        1. Extracts prompts and expected answers from the batch samples.
        2. Generates completions using the current policy model with M3PO if enabled.
        3. Combines prompt and completion tokens.
        4. Computes log probabilities from both the policy model and reference model.
        5. Formats completions for reward calculation.
        6. Repeats prompts and answers to match the number of generated completions.
        7. Returns all data needed for GRPO loss calculation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length,
            lambda_blend=lambda_blend, temperature_m3po=temperature_m3po, use_m3po=use_m3po,
            gating_function=gating_function
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        fwd_model = compiled_model if compiled_model is not None else model
        old_log_probs = compute_log_probs(fwd_model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2, verbose=False,
              use_m3po=False, lambda_blend=0.1, temperature_m3po=0.1, gating_function=None,
              compiled_model=None):
    """
    Computes the GRPO loss for updating the policy model.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
        reward_function: Function that calculates rewards for completions.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter for PPO.
        verbose (bool): Whether to print detailed information about each example.

    Returns:
        torch.Tensor: The GRPO loss to be minimized.

    Explanation:
        1. Computes current token log probabilities using the policy model.
        2. Calculates the probability ratio between current and old policies.
        3. Computes rewards using the provided reward_function.
        4. Calculates advantages by standardizing rewards within each prompt.
        5. Computes the PPO surrogate objective with clipping.
        6. Calculates the KL divergence between reference and policy models.
        7. Combines surrogate loss and KL penalty.
        8. Averages the loss across all tokens and batches.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    fwd_model = compiled_model if compiled_model is not None else model
    if use_m3po and gating_function is not None and gating_function.has_learnable_parameters:
        token_log_probs = compute_log_probs_with_m3po(
            fwd_model, input_ids, attention_mask, logits_to_keep,
            batch_size=rollout_data["batch_size"],
            num_generations=rollout_data["num_generations"],
            lambda_blend=lambda_blend, temperature_m3po=temperature_m3po,
            gating_function=gating_function, completion_mask=completion_mask,
        )
    else:
        token_log_probs = compute_log_probs(fwd_model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards_list = reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"])
    rewards = torch.tensor(
        rewards_list,
        dtype=torch.float32,
        device=device
    )

    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]

    # Print detailed information if verbose mode is enabled
    if verbose:
        print("\n" + "="*80)
        print("DETAILED GRPO TRAINING OUTPUT")
        print("="*80)

        # Get unique prompts (questions)
        unique_prompts = rollout_data["repeated_prompts"][::num_generations]
        unique_answers = rollout_data["repeated_answers"][::num_generations]

        for i, (prompt, expected_answer) in enumerate(zip(unique_prompts, unique_answers)):
            print(f"\n{'-'*80}")
            print(f"EXAMPLE {i+1}/{batch_size}")
            print(f"{'-'*80}")

            # Extract just the question (user content) from the prompt
            question_lines = prompt.split('\n')
            # The question is typically after the system prompt
            question = '\n'.join(question_lines[6:]) if len(question_lines) > 6 else prompt
            print(f"\n[QUESTION]:")
            print(f"{question}")

            print(f"\n[EXPECTED ANSWER]: {expected_answer}")

            print(f"\n[GENERATED RESPONSES] ({num_generations} generations):")
            for j in range(num_generations):
                idx = i * num_generations + j
                completion = rollout_data["formatted_completions"][idx][0]['content']
                reward = rewards_list[idx]

                # Extract the model's answer
                extracted_answer = extract_answer_from_model_output(completion)

                print(f"\n  --- Generation {j+1} ---")
                print(f"  Reward: {reward:.2f}")
                print(f"  Extracted Answer: {extracted_answer}")
                print(f"  Full Response:")
                # Indent the completion for readability
                for line in completion.split('\n'):
                    print(f"    {line}")

        print(f"\n{'='*80}")

    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    # print("Average Reward:", avg_reward)
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Print loss breakdown if verbose
    if verbose:
        print(f"\n[LOSS FUNCTION BREAKDOWN]:")
        print(f"  Surrogate Loss (PPO objective): {surrogate_loss.mean().item():.6f}")
        print(f"  KL Divergence (regularization): {kl.mean().item():.6f}")
        print(f"  Beta (KL penalty coefficient): {beta}")
        print(f"  Per-token Loss (surrogate - beta*KL): {per_token_loss.mean().item():.6f}")
        print(f"  Final Loss (negated, masked avg): {loss.item():.6f}")
        print(f"\n  Loss Formula: L = -mean((surrogate_loss - beta*KL) * mask / mask_sum)")
        print(f"  where surrogate_loss = min(ratio*A, clip(ratio,1-eps,1+eps)*A)")
        print(f"  and KL = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1")
        print(f"{'='*80}\n")

    return loss, avg_reward

def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500, batch_size=4,
                              num_generations=4, max_completion_length=128, beta=0.1,
                              learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None,
                              local_rank=0,
                              lambda_blend=0.1, temperature_m3po=0.1, use_m3po=True,
                              gating_type='baseline', gating_config=None,
                              gating_warmup_steps=0, gating_lr=None, gating_grad_clip=None,
                              gradient_accumulation_steps=1, warmup_ratio=0.1,
                              seed=None):
    """
    Train with GRPO + M3PO (Multi-Path Perception Policy Optimization).

    This function implements GRPO training with optional M3PO cross-path collaborative
    reasoning, following the M3PO paper.

    Args:
        model: The language model to train.
        tokenizer: The tokenizer for encoding and decoding text.
        train_data (list): Training dataset.
        num_iterations (int): Number of outer iterations (reference model updates).
        num_steps (int): Number of batch updates per iteration.
        batch_size (int): Number of prompts per batch.
        num_generations (int): Number of completions per prompt (N in M3PO paper).
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL penalty coefficient (0.005 in M3PO paper).
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of policy updates per batch.
        epsilon (float): PPO clipping parameter.
        reward_function: Function that calculates rewards for completions.
        local_rank (int): Local GPU rank for DDP (set by torchrun).
        lambda_blend (float): M3PO blending coefficient λ (0.1 in paper).
        temperature_m3po (float): M3PO attention temperature T (0.1 in paper).
        use_m3po (bool): Whether to enable M3PO cross-path interaction.

    Returns:
        The trained model.

    Explanation:
        1. For each outer iteration:
           - Creates a reference model as a deep copy of the current policy model.
           - Reinitializes the optimizer for the policy model with paper parameters.
           - For each training step:
             a. Samples a batch of examples from the training data.
             b. Generates rollout data with M3PO-enabled completions.
             c. For mu iterations:
                i. Computes the GRPO loss.
                ii. Updates the policy model using gradient descent.
           - Monitors GPU memory usage and prints progress information.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_m3po and is_main_process():
        print(f"[M3PO] Training with cross-path interaction: lambda={lambda_blend}, temp={temperature_m3po}")

    # Create gating function if specified
    gating_function = None
    if use_m3po and gating_type != 'baseline':
        from transformers.models.qwen2.m3po_gating import create_gating_function
        gating_config = gating_config or {}
        # Use temperature_m3po if not specified in gating_config
        if 'temperature' not in gating_config:
            gating_config['temperature'] = temperature_m3po
        try:
            gating_function = create_gating_function(gating_type, gating_config)
            if is_main_process():
                print(f"[M3PO] Using {gating_type} gating function with config: {gating_config}")
                if gating_function and gating_function.has_learnable_parameters:
                    print(f"[M3PO] Warning: Learnable gating parameters detected. Gradient flow will be enabled during loss computation.")
        except ValueError as e:
            if is_main_process():
                print(f"[M3PO] Error creating gating function: {e}")
                print(f"[M3PO] Falling back to baseline (cosine similarity)")
            gating_function = None

    # Move model to device and sync parameters across ranks.
    model.to(device)
    if dist.is_initialized():
        # Broadcast model parameters from rank 0 so all ranks start with same weights
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        if is_main_process():
            print(f"Manual gradient sync across {dist.get_world_size()} GPUs")
    else:
        print(f"Running on single GPU: {device}")

    raw_model = model

    compiled_model = raw_model  # torch.compile disabled — causes sdpa stride/dynamo issues

    # Outer loop: iterative GRPO updates.
    try:
      for iteration in range(num_iterations):
        if is_main_process():
            print(f"\nIteration {iteration+1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.
        ref_model = copy.deepcopy(raw_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        # print("Reference model created.")

        # Reinitialize the optimizer for this iteration with M3PO paper parameters.
        # Paper Table 3: weight_decay=0.1, betas=(0.9, 0.99)
        has_learnable_gating = gating_function is not None and gating_function.has_learnable_parameters
        if has_learnable_gating:
            gating_function = gating_function.to(device)
            effective_gating_lr = gating_lr if gating_lr is not None else learning_rate * 100
            effective_gating_grad_clip = gating_grad_clip if gating_grad_clip is not None else 1.0
            if is_main_process():
                gating_params = list(gating_function.parameters())
                print(f"[M3PO] Gating function has {len(gating_params)} parameter tensors, "
                      f"shapes: {[p.shape for p in gating_params]}")

        if has_learnable_gating and gating_warmup_steps > 0:
            # Phase 1: freeze model, train only gating params for fixed N steps
            for param in model.parameters():
                param.requires_grad = False
            # Keep gradient checkpointing enabled — disabling causes OOM on longer sequences
            gating_params_list = list(gating_function.parameters())
            if not gating_params_list:
                raise RuntimeError(
                    f"[M3PO] Gating function {type(gating_function).__name__} has no parameters! "
                    f"Module children: {list(gating_function.named_parameters())}"
                )
            optimizer = bnb.optim.AdamW8bit(
                gating_params_list,
                lr=effective_gating_lr,
                weight_decay=0.0,
                betas=(0.9, 0.99),
            )
            # Temperature stays at config value (0.1) — identity init produces similarities in the right range
            if is_main_process():
                print(f"[M3PO] Phase 1 (warmup): training only gating params for {gating_warmup_steps} steps at lr={effective_gating_lr}")
        else:
            # Standard optimizer (no warmup or no learnable gating)
            if has_learnable_gating:
                optimizer = bnb.optim.AdamW8bit(
                    [
                        {"params": list(model.parameters()), "lr": learning_rate},
                        {"params": list(gating_function.parameters()), "lr": effective_gating_lr, "weight_decay": 0.0},
                    ],
                    weight_decay=0.1,
                    betas=(0.9, 0.99),
                )
                phase2_optimizer_step_count = 0
            else:
                params_to_optimize = list(model.parameters())
                optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize,
                    lr=learning_rate,
                    weight_decay=0.1,
                    betas=(0.9, 0.99),
                )
        model.train()

        in_warmup_phase = has_learnable_gating and gating_warmup_steps > 0
        gating_grad_verified = False
        total_steps = num_steps + (gating_warmup_steps if in_warmup_phase else 0)

        if in_warmup_phase:
            # Phase 1: gating LR decays via single cosine over ALL optimizer steps (warmup + joint)
            all_optimizer_steps = total_steps * mu // gradient_accumulation_steps
            def lr_lambda_gating_phase1(current_step):
                progress = current_step / max(1, all_optimizer_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_gating_phase1)
        else:
            # No gating warmup — cosine decay only (no warmup) for the full run
            # scheduler.step() is called per optimizer step, not per outer step.
            # Each outer step does `mu` backwards; optimizer steps every `gradient_accumulation_steps`.
            phase2_total = num_steps * mu // gradient_accumulation_steps
            def lr_lambda_full(current_step):
                progress = current_step / max(1, phase2_total)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_full)

        # Inner loop: training steps with gradient accumulation.
        optimizer.zero_grad()
        accum_count = 0
        step = 0
        while step < total_steps:
            # Phase transition: fixed warmup complete
            if in_warmup_phase and step == gating_warmup_steps:
                if is_main_process():
                    print(f"[M3PO] Phase 2 at step {step + 1}: unfreezing model for joint training")
                in_warmup_phase = False
                for param in model.parameters():
                    param.requires_grad = True
                # Save gating optimizer state before replacing optimizer
                old_gating_state = {}
                for p in gating_function.parameters():
                    if p in optimizer.state:
                        old_gating_state[p] = optimizer.state[p]
                optimizer = bnb.optim.AdamW8bit(
                    [
                        {"params": list(model.parameters()), "lr": learning_rate},
                        {"params": list(gating_function.parameters()), "lr": effective_gating_lr, "weight_decay": 0.0},
                    ],
                    weight_decay=0.1,
                    betas=(0.9, 0.99),
                )
                # Restore gating Adam state (momentum + variance) from Phase 1
                for p in gating_function.parameters():
                    if p in old_gating_state:
                        optimizer.state[p] = old_gating_state[p]
                phase2_optimizer_step_count = 0
                # Cosine LR schedules for Phase 2 (no warmup)
                # Model params: cosine decay from 1.0 over phase 2 steps
                # Gating params: continue single cosine decay over ALL steps (warmup + joint)
                phase2_total = num_steps * mu // gradient_accumulation_steps
                all_optimizer_steps = total_steps * mu // gradient_accumulation_steps
                # gating_steps_so_far = optimizer steps already taken in Phase 1
                gating_steps_so_far = gating_warmup_steps * mu // gradient_accumulation_steps
                def lr_lambda_model(current_step):
                    progress = current_step / max(1, phase2_total)
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                def lr_lambda_gating(current_step):
                    # Continue the single cosine curve from where Phase 1 left off
                    effective_step = gating_steps_so_far + current_step
                    progress = effective_step / max(1, all_optimizer_steps)
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda_model, lr_lambda_gating])
                optimizer.zero_grad()
                accum_count = 0

            batch_samples = random.sample(train_data, batch_size)
            rollout_data = generate_rollout_data(
                raw_model,
                ref_model,
                tokenizer,
                batch_samples,
                num_generations,
                max_completion_length,
                lambda_blend=lambda_blend,
                temperature_m3po=temperature_m3po,
                use_m3po=use_m3po,
                gating_function=gating_function,
                compiled_model=compiled_model,
            )
            for grpo_iter in range(mu):
                verbose_output = False
                loss, avg_reward = grpo_loss(
                    raw_model,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon,
                    verbose=verbose_output,
                    use_m3po=use_m3po,
                    lambda_blend=lambda_blend,
                    temperature_m3po=temperature_m3po,
                    gating_function=gating_function,
                    compiled_model=compiled_model,
                )
                # Scale loss for gradient accumulation
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()

                # Sync model gradients across ranks (replaces DDP's automatic all_reduce)
                if dist.is_initialized():
                    for p in raw_model.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                # One-time verification that gating params receive gradients in Phase 1
                if is_main_process() and in_warmup_phase and not gating_grad_verified:
                    for name, p in gating_function.named_parameters():
                        if p.grad is not None:
                            print(f"[M3PO] Verified: {name} grad norm = {p.grad.norm().item():.6e}")
                        else:
                            print(f"[M3PO] WARNING: {name} has no gradient!")
                    gating_grad_verified = True

                # Sync learnable gating gradients across ranks
                if has_learnable_gating and dist.is_initialized():
                    for p in gating_function.parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

                accum_count += 1

                if accum_count % gradient_accumulation_steps == 0:
                    # Separate gradient clipping for model and gating params
                    if has_learnable_gating:
                        if not in_warmup_phase:
                            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=0.1)
                        torch.nn.utils.clip_grad_norm_(list(gating_function.parameters()), max_norm=effective_gating_grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=0.1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if has_learnable_gating and not in_warmup_phase:
                        phase2_optimizer_step_count += 1

                # Log to wandb (rank 0 only)
                if is_main_process():
                    step_loss = loss.item()
                    log_dict = {
                        "loss": step_loss,
                        "average_reward": avg_reward,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                    }
                    if has_learnable_gating:
                        gating_grad_norm = sum(
                            p.grad.norm().item() ** 2 for p in gating_function.parameters() if p.grad is not None
                        ) ** 0.5
                        log_dict["m3po/gating_grad_norm"] = gating_grad_norm
                        log_dict["m3po/temperature"] = gating_function.temperature
                        log_dict["m3po/gating_lr"] = scheduler.get_last_lr()[-1]
                    if gating_function is not None:
                        stats = gating_function.get_stats_summary()
                        for key, value in stats.items():
                            log_dict[f"m3po/{key}"] = value
                        gating_function.reset_stats()
                    wandb.log(log_dict)
                    phase_str = ' [warmup]' if in_warmup_phase else ''
                    print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{total_steps}, "
                          f"GRPO iter {grpo_iter+1}/{mu}, loss: {step_loss:.4f}, "
                          f"lr: {scheduler.get_last_lr()[0]:.2e}{phase_str}")

            step += 1

    except KeyboardInterrupt:
        if is_main_process():
            print("\n\nCtrl+C detected! Stopping training early...")
            print("Model weights will be saved and evaluation will proceed.")

    return raw_model

"""
Part 7: Training Setup and Execution
We begin by loading the pre-trained model and tokenizer, prepare evaluation data, and then do reinforcement learning (RL) fine-tuning using the our own train_with_grpo we implemented from scratch above.
In the code below:
The device is determined (GPU if available, otherwise CPU).
The pre-trained Qwen2.5-1.5B-Instruct model and tokenizer are loaded. The tokenizer's pad token is set to the eos_token.
A small subset of the dataset is reserved for evaluation to provide a baseline.
The model is optimized for memory efficiency by enabling gradient checkpointing and disabling KV caching.
Step 1: The model is evaluated before fine-tuning to establish a baseline accuracy.
Step 2: Reinforcement learning fine-tuning is performed using the train_with_grpo function with our defined reward functions (format_reward and correctness_reward, combined into combined_reward). The model is trained using a multi-GPU.
Step 3: The final, fine-tuned model and tokenizer are saved to disk.
"""
def reserve_gpu_memory(fraction=0.80):
    """
    Pre-allocate GPU memory on all visible devices so other processes cannot use them.

    PyTorch's CUDA caching allocator holds onto memory even after tensors are freed.
    By allocating a large tensor and immediately freeing it, the memory stays in
    PyTorch's cache (shown as "used" in nvidia-smi to other processes).

    IMPORTANT: Do NOT call torch.cuda.empty_cache() during training — that releases
    the reserved memory back to CUDA, allowing other processes to steal it.

    Args:
        fraction: Fraction of free GPU memory to reserve (default: 0.92).
    """
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        alloc_bytes = int(free_mem * fraction)
        alloc_elements = alloc_bytes // 4  # float32 = 4 bytes
        if alloc_elements > 0:
            tmp = torch.empty(alloc_elements, dtype=torch.float32, device=f'cuda:{i}')
            del tmp
            reserved_gb = alloc_bytes / (1024**3)
            total_gb = total_mem / (1024**3)
            print(f"  GPU {i}: reserved {reserved_gb:.1f} GiB / {total_gb:.1f} GiB")
    print(f"  GPU memory locked — other processes will see these GPUs as occupied.")


def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

if __name__ == "__main__":
    # DDP setup — torchrun sets LOCAL_RANK, RANK, WORLD_SIZE env vars
    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda")
    if is_main_process():
        print(f"DDP initialized: {world_size} GPU(s), primary device: {device}")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    base_output_dir = "outputs"

    if is_main_process():
        print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank}  # Each rank loads to its own GPU
    )
    if is_main_process():
        print("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    train_data = prepare_dataset("train")
    eval_data = prepare_dataset("test")

    # Use only half of the data for testing
    train_data = train_data[:len(train_data)//4]
    eval_data = eval_data[:len(eval_data)//4]

    # Rank-specific seed so each GPU samples different batches
    set_random_seed(BASE_SEED + rank)

    model = optimize_model_memory(model)

    if is_main_process():
        print("\nStarting RL fine-tuning using M3PO (Multi-Path Perception Policy Optimization)...")
        
    # This config follows the M3PO paper (Table 3, page 12)
    training_config = {
        'num_iterations': 1,
        'num_steps': math.ceil(len(train_data) / 4),  # Full epoch
        'batch_size': 4,                   # 4 prompts per step
        'num_generations': 4,              # 4 rollouts per prompt
        'max_completion_length': 400,      # Reduced for 1x A100 40GB
        'beta': 0.005,                     # Paper value (KL penalty coefficient)
        'learning_rate': 5e-6,             # Paper value
        'mu': 1,                           # 1 gradient updates per rollout
        'epsilon': 0.1,
        # M3PO-specific parameters (from paper Table 3)
        'lambda_blend': 0.1,               # Blending coefficient λ
        'temperature_m3po': 0.1,           # Attention temperature T
        'use_m3po': True,                  # Enable M3PO cross-path interaction
        # Gating function selection (for research on alternative gating mechanisms)
        'gating_type': 'raw_dot',         # Options: 'baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'luong', 'bahdanau'
        'gating_config': {                 # Configuration for gating function
            'temperature': 0.1,            # T=0.1 throughout — identity init produces similarities in right range
            'rank': 64,                   # Rank 256 for larger projection → stronger similarities at T=0.1
            'init_strategy': 'xavier',    # Identity init: right numerical regime (~0.001 grad norms)
            'debug': False,                # Enable debug logging
        },
        'gating_warmup_steps': 50,         # Warmup steps for gating-only training
        'gating_lr': 5e-4,                 # Gating learning rate
        'gating_grad_clip': 1.0,           # Separate grad clip for gating (10x less aggressive)
        'gradient_accumulation_steps': 4,  # Paper Table 3
        'warmup_ratio': 0.1,              # Paper Table 3: cosine schedule with warmup
        'seed': BASE_SEED,                 # Base seed; each rank uses seed + rank
    }

    # Initialize Weights & Biases (rank 0 only)
    gating_type = training_config['gating_type']
    seed = training_config['seed']
    trial_number = get_next_trial_number(base_output_dir, gating_type)
    if is_main_process():
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=f"M3PO-{gating_type}-trial{trial_number}-seed{seed}",
            config=training_config,
            reinit=True
        )
        print(f"Weights & Biases initialized. Gating: {gating_type}, Trial: {trial_number}, Seed: {seed}")

    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=combined_reward,
        local_rank=local_rank,
        **training_config
    )

    # Post-training: save, push, then eval (rank 0 only)
    # Other ranks wait at the barrier below until rank 0 finishes.
    if local_rank == 0:
        wandb.finish()
        print("Training completed and wandb run finished.")

        # Create structured output directory: output/{gating_type}/trial_{N}/
        save_dir = os.path.join(base_output_dir, gating_type, f"trial_{trial_number}_seed{seed}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving GRPO fine-tuned model to {save_dir}...")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Save training config for experiment tracking
        serializable_config = {k: v for k, v in training_config.items() if isinstance(v, (int, float, str, bool, dict, list))}
        serializable_config['trial_number'] = trial_number
        serializable_config['model_name'] = model_name
        serializable_config['gating_type'] = gating_type
        with open(os.path.join(save_dir, "training_config.json"), "w") as f:
            json.dump(serializable_config, f, indent=2)

        print(f"Model saved to: {save_dir}")

        # Push to Hugging Face Hub
        print("\nPushing model to Hugging Face Hub...")
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
        hf_repo = f"Alienpenguin10/M3PO-{gating_type}-trial{trial_number}-seed{seed}"
        model.push_to_hub(hf_repo)
        tokenizer.push_to_hub(hf_repo)
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=os.path.join(save_dir, "training_config.json"),
            path_in_repo="training_config.json",
            repo_id=hf_repo,
        )
        print(f"Model pushed to Hugging Face Hub: {hf_repo}")

        # # Evaluate after save/push so model is preserved even if eval fails
        # print("\nFinal model evaluation after GRPO RL fine-tuning:")
        # post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
        # print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

        # # Save results alongside config
        # with open(os.path.join(save_dir, "results.json"), "w") as f:
        #     json.dump({"accuracy": post_grpo_accuracy, "seed": seed}, f, indent=2)

    # Barrier AFTER rank 0 finishes save/push/eval so other ranks don't exit early
    if dist.is_initialized():
        dist.barrier()
        cleanup_ddp()