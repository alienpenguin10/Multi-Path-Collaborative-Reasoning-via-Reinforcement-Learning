# Cross-Path Interaction Integration Guide for H3PO

## Overview
This guide explains how to integrate M3PO's cross-path interaction mechanism into your existing H3PO codebase, which already implements discrete token/thought blending via HRPO.

## Key Finding: No Trainable Weights Needed ✅
The cross-path interaction is **completely parameter-free** - it only requires runtime computation of similarity-based gates.

---

## Your Current Architecture

### Current Flow
```
GRPO Trainer (N parallel paths)
    ↓
Generate N completions per prompt (generation/utils.py::_sample)
    ↓
For each token:
    - Compute last_thinking_states (probability-weighted embedding)
    - Track thinking_embeds, thinking_mask, embeds_ratio
    ↓
Model Forward (unsloth/llama.py::LlamaModel_fast_forward)
    - Blend token embeddings with thinking_embeds during thinking phase
    ↓
Return to trainer for loss computation
```

### Current Thinking Mechanism
- **Single-path**: Each completion is independent
- **Blending**: `h = (1-λ)·token_embed + λ·thinking_state`
- **Phase detection**: Uses `self.answer_start` ("####") to identify thinking vs answer

---

## Required Modifications for Cross-Path Interaction

### 1. **Modify Generation Loop** (`generation/utils.py::_sample`)

#### Current State (Lines 2823-2924)
```python
# You currently generate N paths independently
thinking_embeds = [...]  # Per-path tracking
thinking_mask = [...]
embeds_ratio = [...]

while generating:
    # Each path computes its own last_thinking_states
    last_thinking_states = torch.einsum('bv,vd->bd', probs, embedding_matrix)
```

#### Required Changes

**Add cross-path state tracking:**
```python
def _sample(
    self,
    ...
    enable_cross_path: bool = False,  # NEW: Enable M3PO
    cross_path_lambda: float = 0.1,   # NEW: Blending coefficient
    cross_path_temp: float = 0.1,     # NEW: Softmax temperature
    **model_kwargs,
):
    # Initialize per-path tracking (EXISTING)
    thinking_embeds = [...]
    thinking_mask = [...]
    embeds_ratio = [...]
    
    # NEW: Cross-path interaction state
    # Shape: (batch_size, hidden_dim) for each of N paths
    path_embeddings = []  # Store current embeddings for all paths
    path_distributions = []  # Store probability distributions for gating
    
    while generating:
        # EXISTING: Compute next token probabilities
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        
        # EXISTING: Compute thinking states per path
        last_thinking_states = torch.einsum('bv,vd->bd', probs, self.get_input_embeddings().weight)
        last_thinking_states /= torch.sqrt((probs ** 2).sum(-1, keepdim=True))
        
        # NEW: Cross-path interaction (only during thinking phase)
        if enable_cross_path and is_thinking is not None:
            # Store current path states
            path_embeddings.append(last_thinking_states.clone())
            path_distributions.append(probs.clone())
            
            # Apply cross-path contextual blending
            if len(path_embeddings) >= 2:  # Need at least 2 paths
                last_thinking_states = apply_cross_path_interaction(
                    current_embeds=last_thinking_states,
                    path_embeds_history=path_embeddings,
                    path_probs_history=path_distributions,
                    is_thinking=is_thinking,  # Binary mask per path
                    lambda_blend=cross_path_lambda,
                    temperature=cross_path_temp,
                )
        
        # Pass blended states to model (EXISTING)
        model_inputs.update({
            "is_thinking": is_thinking,
            "last_thinking_states": last_thinking_states
        })
```

**Add the cross-path interaction function:**
```python
def apply_cross_path_interaction(
    current_embeds: torch.Tensor,      # (N, hidden_dim) - N paths
    path_embeds_history: list,          # List of (N, hidden_dim) tensors
    path_probs_history: list,           # List of (N, vocab_size) tensors
    is_thinking: list,                  # List of bool, length N
    lambda_blend: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Apply M3PO cross-path interaction.
    
    Returns:
        Blended embeddings: (N, hidden_dim)
    """
    N = current_embeds.shape[0]  # Number of paths
    device = current_embeds.device
    
    # Get current step's probability distributions (last in history)
    current_probs = path_probs_history[-1]  # (N, vocab_size)
    
    # Compute pairwise cosine similarity between distributions
    # A_ij = cosine_similarity(p_i, p_j)
    normalized_probs = current_probs / (current_probs.norm(dim=1, keepdim=True) + 1e-8)
    similarity_matrix = torch.mm(normalized_probs, normalized_probs.t())  # (N, N)
    
    # Mask diagonal (no self-interaction)
    mask = torch.eye(N, device=device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Mask inactive paths (those that exited thinking mode)
    thinking_mask = torch.tensor(is_thinking, device=device, dtype=torch.bool)
    # If path i is not thinking, it shouldn't contribute or receive
    active_mask = thinking_mask.unsqueeze(0) & thinking_mask.unsqueeze(1)  # (N, N)
    active_mask = active_mask & ~mask  # Exclude diagonal
    similarity_matrix = similarity_matrix.masked_fill(~active_mask, float('-inf'))
    
    # Temperature-scaled softmax to get attention weights
    attention_weights = F.softmax(similarity_matrix / temperature, dim=1)  # (N, N)
    
    # Compute contextual embeddings: c_i = Σ_j A_ij * e_j
    contextual_embeds = torch.mm(attention_weights, current_embeds)  # (N, hidden_dim)
    
    # Blend: h_i = (1 - λ) * e_i + λ * c_i
    blended_embeds = (1 - lambda_blend) * current_embeds + lambda_blend * contextual_embeds
    
    # Only apply blending to paths still in thinking mode
    result = torch.where(
        thinking_mask.unsqueeze(1),
        blended_embeds,
        current_embeds  # Keep original for non-thinking paths
    )
    
    return result
```

---

### 2. **Modify GRPO Trainer** (`grpo_trainer.py`)

#### Update `_generate_single_turn` (Line 1318)
```python
# CURRENT
prompt_completion_ids, thinking_embeds, thinking_mask, embeds_ratio = unwrapped_model.generate(
    **generate_inputs,
    generation_config=self.generation_config,
    processing_class=self.processing_class,
    return_thinking_embeds=True,
    disable_compile=True
)

# ADD M3PO parameters
prompt_completion_ids, thinking_embeds, thinking_mask, embeds_ratio = unwrapped_model.generate(
    **generate_inputs,
    generation_config=self.generation_config,
    processing_class=self.processing_class,
    return_thinking_embeds=True,
    enable_cross_path=True,  # NEW: Enable M3PO
    cross_path_lambda=0.1,   # NEW: M3PO blending coefficient
    cross_path_temp=0.1,     # NEW: M3PO temperature
    disable_compile=True
)
```

#### Add Configuration to `GRPOConfig`
```python
# In trl/trainer/grpo_config.py (or wherever GRPOConfig is defined)

@dataclass
class GRPOConfig:
    # ... existing fields ...
    
    # M3PO (Multi-Path Perception) parameters
    enable_cross_path: bool = False
    """Enable cross-path interaction during thinking phase (M3PO)"""
    
    cross_path_lambda: float = 0.1
    """Blending coefficient for cross-path contextual embeddings (default: 0.1)"""
    
    cross_path_temp: float = 0.1
    """Temperature for softmax in cross-path attention weights (default: 0.1)"""
```

---

### 3. **Model Forward Pass** (unsloth/llama.py)

#### Current Implementation (Lines 1193-1205)
Your model already handles `is_thinking` and `last_thinking_states` correctly:
```python
if is_thinking is not None and last_thinking_states is not None:
    thinking_embeds = last_thinking_states
    X_hat, a_t = self.model.thinking_residual(X, last_thinking_states.unsqueeze(1))
    embeds_ratio = a_t.mean(-1).squeeze()
    embeds_ratio[~torch.tensor(is_thinking)] = 1.
```

**No changes needed here!** The cross-path interaction happens BEFORE this point (in generation loop), so the model just receives the already-blended `last_thinking_states`.

---

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add `apply_cross_path_interaction()` function to `generation/utils.py`
- [ ] Modify `_sample()` to track path embeddings and distributions
- [ ] Add M3PO parameters to `_sample()` signature
- [ ] Add configuration fields to `GRPOConfig`

### Phase 2: Integration
- [ ] Update `grpo_trainer._generate_single_turn()` to pass M3PO flags
- [ ] Ensure `self.answer_start` is properly defined (should be "####")
- [ ] Add debug prints to verify cross-path interaction is working

### Phase 3: Testing
- [ ] Test with `enable_cross_path=False` (should match current behavior)
- [ ] Test with `enable_cross_path=True` (should show collaboration)
- [ ] Verify that answer phase remains isolated (no cross-path after "####")
- [ ] Check memory usage (O(N²) attention computation)

---

## Critical Implementation Notes

### 1. **Batch Dimension Handling**
Your GRPO generates completions in batches:
- `batch_size = per_device_train_batch_size`
- `num_generations = N` (e.g., 8)
- **Total sequences** = `batch_size × num_generations`

**Important:** Cross-path interaction should occur **within each group of N generations for the same prompt**, NOT across different prompts.

**Fix:**
```python
def apply_cross_path_interaction_batched(
    current_embeds: torch.Tensor,      # (batch_size * N, hidden_dim)
    path_probs: torch.Tensor,          # (batch_size * N, vocab_size)
    is_thinking: list,                  # Length: batch_size * N
    num_generations: int,               # N
    lambda_blend: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    batch_size = current_embeds.shape[0] // num_generations
    hidden_dim = current_embeds.shape[1]
    
    # Reshape to separate batches and paths
    # (batch_size * N, hidden_dim) -> (batch_size, N, hidden_dim)
    embeds = current_embeds.view(batch_size, num_generations, hidden_dim)
    probs = path_probs.view(batch_size, num_generations, -1)
    
    # Process each batch independently
    blended_list = []
    for b in range(batch_size):
        batch_embeds = embeds[b]  # (N, hidden_dim)
        batch_probs = probs[b]     # (N, vocab_size)
        batch_thinking = is_thinking[b*num_generations:(b+1)*num_generations]
        
        blended = apply_cross_path_interaction(
            batch_embeds, [batch_embeds], [batch_probs],
            batch_thinking, lambda_blend, temperature
        )
        blended_list.append(blended)
    
    # Reshape back to (batch_size * N, hidden_dim)
    return torch.cat(blended_list, dim=0)
```

### 2. **Answer Start Token**
Ensure `self.answer_start` is set:
```python
# In generation/utils.py::_sample or __init__
self.answer_start = "####"  # Same as HRPO paper
```

### 3. **Memory Optimization**
For large N (e.g., N=8):
- Similarity matrix: O(N²) = 64 elements per batch item
- Acceptable overhead
- If N > 16, consider sparse attention or top-k collaboration

---

## Expected Behavior

### Without Cross-Path (Current)
```
Prompt: "Calculate 23 × 17"
Path 1: "Let me compute... [thinking] ... #### 391"
Path 2: "First I'll... [thinking] ... #### 391"
Path 3: "23 × 17 = ... [thinking] ... #### 391"
```
Each path thinks independently.

### With Cross-Path (M3PO)
```
Prompt: "Calculate 23 × 17"
Path 1: "Let me break it down... [collaborates] ... #### 391"
Path 2: "I'll try 20×17 + 3×17... [sees Path 1's approach] ... #### 391"
Path 3: "Direct multiplication... [informed by others] ... #### 391"
```
Paths exchange insights during thinking, improving consensus and reducing variance.

---

## Testing Script

```python
# test_cross_path.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Test prompt
prompt = "Calculate 23 × 17. Show your reasoning step by step."

# Generate with M3PO
outputs = model.generate(
    tokenizer(prompt, return_tensors="pt").input_ids,
    return_thinking_embeds=True,
    enable_cross_path=True,
    cross_path_lambda=0.1,
    cross_path_temp=0.1,
    max_new_tokens=200,
    num_return_sequences=8,  # N paths
)

# Check collaboration happened
print("Outputs:", tokenizer.batch_decode(outputs[0]))
print("Thinking embeds shape:", outputs[1].shape)
```

---

## Performance Tuning

### Hyperparameters
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `λ` (lambda) | 0.1 | [0.05, 0.3] | Higher = more collaboration |
| `T` (temperature) | 0.1 | [0.01, 1.0] | Lower = sharper attention |

### Ablation Studies to Run
1. **λ = 0**: Disable cross-path (should match baseline)
2. **λ = 0.1**: Moderate collaboration
3. **λ = 0.3**: Strong collaboration
4. **No phase isolation**: Allow cross-path in answer phase (should hurt diversity)

---

## Debugging Checklist

If cross-path interaction doesn't work:

1. **Check path separation**: Are N generations properly grouped?
2. **Verify thinking phase detection**: Is `is_thinking` correctly computed?
3. **Inspect similarity matrix**: Are paths actually similar during thinking?
4. **Check masking**: Are diagonal and inactive paths properly masked?
5. **Validate shapes**: Do tensors have expected dimensions?

Add debug prints:
```python
if debug_cross_path:
    print(f"Similarity matrix:\n{similarity_matrix}")
    print(f"Attention weights:\n{attention_weights}")
    print(f"Active paths: {sum(is_thinking)}/{len(is_thinking)}")
    print(f"Blend ratio: λ={lambda_blend}")
```

---

## Summary

**What you need to do:**
1. ✅ **No new weights** - just runtime computation
2. ✏️ **Modify `_sample()`** in `generation/utils.py` to track and blend paths
3. ✏️ **Add M3PO config** to `GRPOConfig`
4. ✏️ **Update trainer call** to pass M3PO flags
5. ✅ **Model forward pass** already handles blended states correctly

**What stays the same:**
- Your existing thinking/discrete blending mechanism
- Model architecture (no new layers)
- Training loop structure
- Loss computation

**Key insight:**
M3PO is a "meta-algorithm" that sits on top of your existing HRPO implementation. It just makes the N parallel rollouts in GRPO talk to each other during the thinking phase!
