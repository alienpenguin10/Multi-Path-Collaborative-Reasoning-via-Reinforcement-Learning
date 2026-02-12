"""
Part 1: Basic Setup and Imports
"""
# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import os
import numpy as np
import wandb


# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["M3PO_DEBUG"] = "-1"  # Enable M3PO debug prints

# Call the function to set random seed for reproducibility
from utils import set_random_seed   
set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_PROJECT"] = "latent-space-reasoning"

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
Part 6: DataParallel GRPO From Scratch
In this section, we implement all the building blocks of the GRPO algorithm from scratch. 
The implementation assumes that the machine running the code has at least 2 GPUs. 
We use PyTorch's DataParallel API to distribute the policy model across the GPU cores, one copy of the model per GPU core. The batch is split between the GPU cores.
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
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        ids_to_keep = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]
        return selective_log_softmax(logits, ids_to_keep)

    # Process in chunks to avoid OOM
    results = []
    for i in range(0, batch_size, chunk_size):
        chunk_ids = input_ids[i:i+chunk_size]
        chunk_mask = attention_mask[i:i+chunk_size]

        chunk_logits = model(input_ids=chunk_ids, attention_mask=chunk_mask).logits[:, :-1, :]
        chunk_ids_to_keep = chunk_ids[:, -logits_to_keep:]
        chunk_logits = chunk_logits[:, -logits_to_keep:, :]

        # Compute log probs for this chunk (no need for extra chunking since already small)
        log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
        chunk_result = log_probs.gather(dim=-1, index=chunk_ids_to_keep.unsqueeze(-1)).squeeze(-1)
        results.append(chunk_result)

        # Free memory
        del chunk_logits, log_probs

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
    except:
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        except:
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

        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )

    # print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length,
                          lambda_blend=0.1, temperature_m3po=0.1, use_m3po=True, gating_function=None):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
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

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2, verbose=False):
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
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
                              learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, device_ids=None,
                              lambda_blend=0.1, temperature_m3po=0.1, use_m3po=True,
                              gating_type='baseline', gating_config=None):
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
        device_ids (list): List of GPU device IDs for DataParallel.
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
    assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_m3po:
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
            print(f"[M3PO] Using {gating_type} gating function with config: {gating_config}")
            if gating_function and gating_function.has_learnable_parameters:
                print(f"[M3PO] Warning: Learnable gating parameters detected. Gradient flow will be enabled during loss computation.")
        except ValueError as e:
            print(f"[M3PO] Error creating gating function: {e}")
            print(f"[M3PO] Falling back to baseline (cosine similarity)")
            gating_function = None

    # Wrap model with DataParallel if multiple GPUs are available.

    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel across GPUs: {device_ids}")

    # Outer loop: iterative GRPO updates.
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        # Create a reference model (deep copy) and set it to eval mode.
        ref_model = copy.deepcopy(model.module)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        # print("Reference model created.")

        # Reinitialize the optimizer for this iteration with M3PO paper parameters.
        # Paper Table 3: weight_decay=0.1, betas=(0.9, 0.99)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.1,
            betas=(0.9, 0.99)
        )
        model.train()

        # Inner loop: your original training steps.
        for step in range(num_steps):
            batch_samples = random.sample(train_data, batch_size)
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model.module,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length,
                    lambda_blend=lambda_blend,
                    temperature_m3po=temperature_m3po,
                    use_m3po=use_m3po,
                    gating_function=gating_function
                )
            for grpo_iter in range(mu):
                # Enable verbose output on first step to show detailed breakdown
                # verbose_output = (step == 0 and grpo_iter == 0)
                verbose_output = False  # Disabled for full training
                loss, avg_reward = grpo_loss(
                    model.module,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon,
                    verbose=verbose_output
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}")
                #for i in range(torch.cuda.device_count()):
                #    print(f"GPU {i} Usage: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MiB, "
                #          f"Utilization: {torch.cuda.utilization(i)}%")
                # Uncomment to see the GPU utilization stats
    return model.module

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
    # Main execution
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir = "math_solver_model"

    print("Loading fine-tuned model from grpo_finetuned_model...")
    model = AutoModelForCausalLM.from_pretrained(
        "grpo_finetuned_model",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("Fine-tuned model loaded")

    tokenizer = AutoTokenizer.from_pretrained("grpo_finetuned_model", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    device_ids = list(range(num_gpus)) if num_gpus > 1 else None

    all_data = prepare_dataset("test")
    random.shuffle(all_data)
    size_of_eval_data = 30 # change to a smaller value to save time or to a larger number for a more reliable estimate
    eval_data = all_data[:size_of_eval_data]
    train_data = all_data[size_of_eval_data:]  # Use all remaining data for training (~7400 examples)

    # print("\nInitial model evaluation before finetuning:")
    # pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    # print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = optimize_model_memory(model)

    print("\nStarting RL fine-tuning using M3PO (Multi-Path Perception Policy Optimization)...")
    # This config follows the M3PO paper (Table 3, page 12)
    # Tested on 8xA100 node with 80GB VRAM each
    training_config = {
        'num_iterations': 1,
        'num_steps': 500,                  # Full training: ~500 steps (500 * 5 batch = 2500 examples per iteration)
        'batch_size': 5,                   # 5 examples per batch
        'num_generations': 4,              # Paper uses 4 or 8 (using 4 for faster output)
        'max_completion_length': 512,      # Max tokens per completion
        'beta': 0.005,                     # Paper value (KL penalty coefficient)
        'learning_rate': 5e-6,             # Paper value
        'mu': 1,
        'epsilon': 0.1,
        # M3PO-specific parameters (from paper Table 3)
        'lambda_blend': 0.1,               # Blending coefficient λ
        'temperature_m3po': 0.1,           # Attention temperature T
        'use_m3po': True,                  # Enable M3PO cross-path interaction
        # Gating function selection (for research on alternative gating mechanisms)
        'gating_type': 'baseline',         # Options: 'baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'luong', 'bahdanau'
        'gating_config': {                 # Configuration for gating function
            'temperature': 0.1,            # Will override temperature_m3po if gating is used
            'debug': False,                # Enable debug logging
        },
    }

    # Initialize Weights & Biases
    # wandb.init(project=os.environ["WANDB_PROJECT"], name="M3PO", reinit=True)
    # print("Weights & Biases initialized.")

    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        reward_function=combined_reward,
        device_ids=device_ids,
        **training_config
    )

    # wandb.finish()
    # print("Training completed and wandb run finished.")

    print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    # print("\nSaving GRPO fine-tuned model...")
    # model.save_pretrained("grpo_finetuned_model")
    # tokenizer.save_pretrained("grpo_finetuned_model")

    # # Push to Hugging Face Hub
    # print("\nPushing model to Hugging Face Hub...")
    # from huggingface_hub import login
    # login(token="")
    # model.push_to_hub("Alienpenguin10/M3PO")
    # tokenizer.push_to_hub("Alienpenguin10/M3PO")
    # print("Model pushed to Hugging Face Hub: Alienpenguin10/M3PO")