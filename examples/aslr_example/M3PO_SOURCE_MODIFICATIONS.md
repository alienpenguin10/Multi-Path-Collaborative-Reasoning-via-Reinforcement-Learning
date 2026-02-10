# M3PO/ASLR Source Code Modifications Summary

This document details every modification made to the source code of Transformers, TRL, and Unsloth to implement M3PO (Multi-Path Perception Policy Optimization) with ASLR (Adaptive Stochastic Latent Reasoning).

---

## Table of Contents
1. [Transformers Library Modifications](#transformers-library-modifications)
2. [TRL Library Modifications](#trl-library-modifications)
3. [Unsloth Library Modifications](#unsloth-library-modifications)
4. [Integration Flow](#integration-flow)
5. [Training vs Inference Behavior](#training-vs-inference-behavior)

---

## 1. Transformers Library Modifications

### 1.1 Configuration Files

#### `transformers/src/transformers/models/llama/configuration_llama.py`

**Purpose:** Add ASLR hyperparameters to model configuration.

**Changes:**
```python
class LlamaConfig(PretrainedConfig):
    def __init__(
        self,
        # ... existing parameters ...
        
        # M3PO/ASLR Parameters
        aslr_num_paths=4,            # Number of parallel reasoning paths (N)
        aslr_max_iterations=2,       # Number of iterative refinements (T)
        aslr_threshold=0.5,          # Router activation threshold
        aslr_gumbel_temp=1.0,        # Gumbel-Softmax temperature for stochasticity
        aslr_cross_path_lambda=0.1,  # Cross-path attention weight
        aslr_cross_path_temp=0.1,    # Cross-path collaboration temperature
        **kwargs,
    ):
        # Store ASLR parameters
        self.aslr_num_paths = aslr_num_paths
        self.aslr_max_iterations = aslr_max_iterations
        self.aslr_threshold = aslr_threshold
        self.aslr_gumbel_temp = aslr_gumbel_temp
        self.aslr_cross_path_lambda = aslr_cross_path_lambda
        self.aslr_cross_path_temp = aslr_cross_path_temp
```

**Why:** These parameters control the M3PO/ASLR behavior and need to be part of the model's persistent configuration.

---

#### `transformers/src/transformers/models/qwen2/configuration_qwen2.py`

**Purpose:** Same as LLaMA config - add ASLR hyperparameters.

**Changes:** Identical ASLR parameters added to `Qwen2Config.__init__`.

**Why:** Qwen2 models need the same ASLR configuration as LLaMA models.

---

### 1.2 Model Architecture Files

#### `transformers/src/transformers/models/llama/modeling_llama.py`

**Purpose:** Implement M3PO components and integrate ASLR logic into the model.

**New Components Added:**

##### 1.2.1 M3PORouter
```python
class M3PORouter(nn.Module):
    """
    Learnable router that decides when to activate latent thinking mode.
    Uses a single linear layer to predict activation probability from hidden states.
    """
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, 1)
        self.threshold = config.aslr_threshold  # Fixed threshold from config
```

**Key Points:**
- Single linear layer: `hidden_size -> 1`
- Threshold is NOT learnable (fixed from config) to prevent drift
- Returns sigmoid activation for soft gating

##### 1.2.2 M3POPathCollaborator
```python
class M3POPathCollaborator(nn.Module):
    """
    Enables cross-path attention and collaboration among N latent paths.
    Uses multi-head attention mechanism to allow paths to exchange information.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            batch_first=True
        )
        self.cross_path_lambda = config.aslr_cross_path_lambda
```

**Key Points:**
- Multi-head attention with 8 heads
- Batch-first format: `(batch, N, hidden_size)`
- Lambda controls blending: `output = (1-λ)*input + λ*attention_output`

##### 1.2.3 M3POPathAggregator
```python
class M3POPathAggregator(nn.Module):
    """
    Aggregates N latent paths back into a single hidden state.
    Uses learned weighted combination followed by projection.
    """
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.aslr_num_paths))
        self.weight_proj = nn.Linear(config.hidden_size, config.hidden_size)
```

**Key Points:**
- Learnable path weights (softmax normalized)
- Weighted average across N paths
- Linear projection for final output

##### 1.2.4 LlamaModel Modifications

**In `__init__`:**
```python
# Initialize M3PO modules
self.m3po_router = M3PORouter(config)
self.m3po_path_collaborator = M3POPathCollaborator(config)
self.m3po_path_aggregator = M3POPathAggregator(config)

# Store ASLR parameters
self.aslr_num_paths = config.aslr_num_paths
self.aslr_max_iterations = config.aslr_max_iterations
# ... other ASLR params ...

# Initialize router statistics collection
self._m3po_router_stats = {'weights': [], 'activations': []}
```

**In `forward` (lines ~400-500):**

The forward method was modified to include M3PO/ASLR logic, but **this is NOT used during training** because Unsloth patches the forward method. This implementation exists for:
1. Non-Unsloth inference
2. Reference implementation
3. Debugging purposes

**Debug prints added:**
- "DEBUG: Qwen2Model initialized with M3PO/ASLR"
- Router activation decisions

---

#### `transformers/src/transformers/models/qwen2/modeling_qwen2.py`

**Purpose:** Same modifications as LLaMA.

**Changes:**
- Imports M3PO modules from `modeling_llama`
- Identical `__init__` modifications
- Identical `forward` modifications
- Same debug prints

**Why:** Qwen2 and LLaMA share the same architecture pattern.

---

### 1.3 Generation Utils

#### `transformers/src/transformers/generation/utils.py`

**Purpose:** Pass M3PO/ASLR parameters through the generation pipeline.

**Changes Made:**

##### 1.3.1 `_sample` Method Signature (line ~2650)
```python
def _sample(
    self,
    input_ids: torch.LongTensor,
    # ... existing params ...
    
    # M3PO/ASLR parameters
    aslr_num_paths: int = 1,
    aslr_max_iterations: int = 1,
    aslr_threshold: float = 0.5,
    aslr_gumbel_temp: float = 1.0,
    aslr_cross_path_lambda: float = 0.1,
    aslr_cross_path_temp: float = 0.1,
    is_m3po_inference: bool = False,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
```

##### 1.3.2 Parameter Extraction (line ~850)
```python
def _extract_generation_mode_kwargs(self, generation_config, model_kwargs):
    # ... existing code ...
    
    # Extract M3PO/ASLR parameters
    aslr_num_paths = generation_config.get("aslr_num_paths", 1)
    aslr_max_iterations = generation_config.get("aslr_max_iterations", 1)
    # ... other ASLR params ...
    
    return {
        # ... existing returns ...
        "aslr_num_paths": aslr_num_paths,
        "aslr_max_iterations": aslr_max_iterations,
        # ... other ASLR params ...
    }
```

##### 1.3.3 Fixed IndentationError (line 2915)
Fixed missing pass statement after else clause.

**Key Points:**
- The M3PO/ASLR logic is NOT implemented in `_sample` itself
- These parameters are passed through to the model's forward pass
- Unsloth's patched methods actually use these parameters
- The generation loop remains unchanged

---

## 2. TRL Library Modifications

### 2.1 `trl/trainer/grpo_trainer.py`

**Purpose:** Integrate M3PO/ASLR into GRPO training pipeline.

#### 2.1.1 GRPOConfig Parameters (lines ~100-150)

**Added to `GRPOConfig` class:**
```python
@dataclass
class GRPOConfig(GRPOTrainerConfig):
    # ... existing parameters ...
    
    # M3PO/ASLR Parameters
    aslr_num_paths: int = 4
    aslr_max_iterations: int = 2
    aslr_threshold: float = 0.5
    aslr_gumbel_temp: float = 1.0
    aslr_cross_path_lambda: float = 0.1
    aslr_cross_path_temp: float = 0.1
```

**Why:** These config parameters control M3PO behavior during GRPO training.

---

#### 2.1.2 GRPOTrainer.__init__ (lines ~300-350)

**Store ASLR parameters:**
```python
def __init__(self, model, args, ...):
    # ... existing init ...
    
    # Store M3PO/ASLR parameters
    self.aslr_num_paths = args.aslr_num_paths
    self.aslr_max_iterations = args.aslr_max_iterations
    self.aslr_threshold = args.aslr_threshold
    self.aslr_gumbel_temp = args.aslr_gumbel_temp
    self.aslr_cross_path_lambda = args.aslr_cross_path_lambda
    self.aslr_cross_path_temp = args.aslr_cross_path_temp
```

---

#### 2.1.3 _generate_single_turn Method (lines ~380-400)

**Pass ASLR parameters to generation:**
```python
def _generate_single_turn(self, prompts, generation_config=None):
    # ... existing code ...
    
    # Generate completions with M3PO/ASLR
    generated = unwrapped_model.generate(
        **inputs,
        generation_config=generation_config,
        # M3PO/ASLR parameters
        aslr_num_paths=self.aslr_num_paths,
        aslr_max_iterations=self.aslr_max_iterations,
        aslr_threshold=self.aslr_threshold,
        aslr_gumbel_temp=self.aslr_gumbel_temp,
        aslr_cross_path_lambda=self.aslr_cross_path_lambda,
        aslr_cross_path_temp=self.aslr_cross_path_temp,
        is_m3po_inference=False,  # False = training mode, full ASLR active
    )
```

**Key Point:** `is_m3po_inference=False` during GRPO generation ensures full M3PO/ASLR is active.

---

#### 2.1.4 Debug Prints Added

**Lines 380-383:**
```python
print(f"[GRPO Trainer] _generate_single_turn called with batch size: {len(prompts)}")
```

**Lines 1394-1396:**
```python
print(f"[GRPO Trainer] About to call _generate_and_score_completions")
```

**Lines 1620-1626:**
```python
print(f"[GRPO Trainer] _get_per_token_logps_and_entropies called")
```

These help trace the GRPO execution flow.

---

## 3. Unsloth Library Modifications

### 3.1 `unsloth/models/llama.py`

**Purpose:** This is the CRITICAL file. Unsloth patches the model's forward pass for speed, so we must implement M3PO/ASLR directly in Unsloth's fast_forward methods.

---

#### 3.1.1 get_peft_model Function (lines ~200-250)

**Allow M3PO modules in modules_to_save:**
```python
def get_peft_model(model, ...):
    # ... existing code ...
    
    # Allow M3PO modules
    ALLOWED_MODULES = frozenset((
        # ... existing modules ...
        "m3po_router",
        "m3po_path_collaborator", 
        "m3po_path_aggregator",
    ))
```

**Set M3PO modules to Float32 for stable training:**
```python
# After PEFT wrapping
base_model = model.model.model if hasattr(model.model, 'model') else model.model

if hasattr(base_model, 'm3po_router'):
    base_model.m3po_router = base_model.m3po_router.to(torch.float32)
    base_model.m3po_path_collaborator = base_model.m3po_path_collaborator.to(torch.float32)
    base_model.m3po_path_aggregator = base_model.m3po_path_aggregator.to(torch.float32)
    
    # Enable gradients
    for p in base_model.m3po_router.parameters():
        p.requires_grad = True
    # ... same for collaborator and aggregator ...
```

**Why:**
- M3PO modules need Float32 for stable gradient flow
- Main model uses BFloat16 for memory efficiency
- Must explicitly set requires_grad=True after freezing

---

#### 3.1.2 LlamaModel_fast_forward Function (lines ~850-1000)

**THIS IS WHERE M3PO/ASLR ACTUALLY RUNS DURING TRAINING**

The entire M3PO/ASLR inner loop is implemented here because this is the function Unsloth uses during training.

**Key Code Blocks:**

##### A. Activation Condition (line ~890)
```python
if not is_m3po_inference and hasattr(self, 'm3po_router'):
    # M3PO/ASLR logic goes here
```

**Why `not is_m3po_inference`:**
- During GRPO generation: `is_m3po_inference=False` → M3PO ACTIVE
- During policy training: `is_m3po_inference` not set → M3PO ACTIVE  
- During evaluation: `is_m3po_inference=True` → M3PO DISABLED (single-path)

##### B. Router Decision (lines ~895-905)
```python
# Get router gate score
router_gate_scores = torch.sigmoid(self.m3po_router.gate(hidden_states))
last_token_gate = router_gate_scores[:, -1, :].squeeze(-1)  # (batch_size,)

# Soft gating mechanism (differentiable)
gate_weight = torch.sigmoid((last_token_gate - self.m3po_router.threshold) * 10.0)
active_mask = (last_token_gate > self.m3po_router.threshold)
active_indices = torch.where(active_mask)[0]

# Collect statistics for TensorBoard
if hasattr(self, '_m3po_router_stats'):
    for i in range(batch_size):
        self._m3po_router_stats['weights'].append(last_token_gate[i].item())
        self._m3po_router_stats['activations'].append(active_mask[i].item())
```

**Key Points:**
- Soft gating uses sigmoid on scaled threshold difference
- Hard mask determines which sequences activate M3PO
- Statistics collected for TensorBoard logging

##### C. Latent Expansion (lines ~910-920)
```python
if active_indices.numel() > 0:
    active_hidden = hidden_states[active_indices, -1, :]  # (num_active, hidden_dim)
    N = self.aslr_num_paths
    
    # Expand to N paths
    latent_paths = active_hidden.unsqueeze(1).expand(-1, N, -1)  # (num_active, N, hidden_dim)
```

##### D. Iterative Refinement with Gumbel-Softmax (lines ~925-950)
```python
for iteration in range(self.aslr_max_iterations):
    # Add Gumbel noise for stochastic exploration
    gumbel_noise = torch.nn.functional.gumbel_softmax(
        torch.randn_like(latent_paths),
        tau=self.aslr_gumbel_temp,
        hard=False
    )
    
    # Inject stochasticity
    latent_paths_noisy = latent_paths + 0.01 * gumbel_noise
    
    # Cross-path collaboration via attention
    attended_paths, _ = self.m3po_path_collaborator.attention(
        latent_paths_noisy,
        latent_paths_noisy,
        latent_paths_noisy
    )
    
    # Blend with residual connection
    latent_paths = (1 - self.m3po_path_collaborator.cross_path_lambda) * latent_paths + \
                   self.m3po_path_collaborator.cross_path_lambda * attended_paths
```

**Key Points:**
- T iterations of refinement
- Gumbel-Softmax provides differentiable stochasticity
- Cross-path attention allows information exchange
- Residual connection preserves information

##### E. Path Aggregation (lines ~955-965)
```python
# Aggregate N paths → 1 hidden state
collapsed_hidden = self.m3po_path_aggregator(latent_paths)  # (num_active, hidden_dim)

# Soft gating: blend original with M3PO output
gate_weight_expanded = gate_weight[active_indices].unsqueeze(-1)
blended_hidden = (1 - gate_weight_expanded) * active_hidden + \
                 gate_weight_expanded * collapsed_hidden
```

**Key Points:**
- Aggregator uses learned weighted combination
- Soft gating blends original and M3PO states
- Maintains differentiability throughout

##### F. Update Hidden States (lines ~970-980)
```python
# Non-in-place update to preserve gradients
hidden_states_new = hidden_states.clone()
hidden_states_new[active_indices, -1, :] = blended_hidden.to(hidden_states.dtype)
hidden_states = hidden_states_new
```

**Critical:** Must use non-in-place update to maintain gradient flow.

##### G. Debug Prints
```python
print(f"[M3PO/ASLR] Training mode - ASLR inner loop active in model forward")
print(f"[M3PO/ASLR] Router activated for {num_active}/{batch_size} sequences")
print(f"[M3PO/ASLR] Running {self.aslr_max_iterations} iterations with {N} paths")
```

---

#### 3.1.3 _LlamaModel_fast_forward_inference_custom (lines ~1800-1850)

**Same M3PO/ASLR logic copied here**

**Why:** This method is called during the GRPO generation phase (even though model is in eval mode). The M3PO logic must be present here too.

**Key difference:** This runs during generation, not during policy training forward pass.

---

## 4. Integration Flow

### 4.1 Training Flow (GRPO)

```
1. GRPOTrainer calls _generate_single_turn()
   ↓
2. Passes ASLR params to model.generate()
   is_m3po_inference=False → M3PO ACTIVE
   ↓
3. Generation calls Unsloth's LlamaModel_fast_forward_inference_custom
   ↓
4. M3PO/ASLR logic executes:
   - Router decides activation
   - Expands to N paths
   - T iterations of refinement
   - Cross-path collaboration
   - Aggregation back to 1 path
   ↓
5. Tokens generated with M3PO reasoning
   ↓
6. GRPOTrainer computes rewards
   ↓
7. Policy training forward pass uses LlamaModel_fast_forward
   (Same M3PO logic, gradients flow back)
   ↓
8. M3PO modules updated via backprop
```

### 4.2 Evaluation Flow (Should Be)

```
1. Evaluation script calls model.generate()
   is_m3po_inference=True → M3PO DISABLED
   ↓
2. Generation uses Unsloth's fast_forward_inference
   ↓
3. M3PO logic is SKIPPED (due to is_m3po_inference=True)
   ↓
4. Single-path generation (no latent reasoning)
   ↓
5. Should rely on internalized reasoning from training
```

**POTENTIAL BUG:** If `is_m3po_inference` is not properly passed or handled, evaluation might:
- Still use full M3PO (N=4, T=2) → slower but should work
- Not use M3PO when it should → poor performance

---

## 5. Training vs Inference Behavior

### 5.1 Parameter: `is_m3po_inference`

| Mode | Setting | M3PO Active? | Purpose |
|------|---------|--------------|---------|
| GRPO Generation | `False` | YES | Generate diverse trajectories with reasoning |
| Policy Training | Not set | YES | Train with full M3PO gradients |
| Evaluation | `True` | NO | Fast single-path inference |

### 5.2 Key Hypothesis: Internalization

**Training (N=4, T=2):**
- Model learns to reason with 4 parallel paths
- 2 iterations of refinement per token
- Cross-path collaboration teaches diverse perspectives
- Router learns when to "think hard"

**Inference (N=1, T=1 or disabled):**
- Model should have internalized the reasoning
- Single forward pass should be sufficient
- Faster inference without explicit multi-path computation

**If evaluation fails:** The internalization hypothesis may be wrong, or:
1. Model hasn't trained long enough
2. Router is activating incorrectly
3. M3PO is being bypassed during evaluation
4. Configuration mismatch between training and eval

---

## 6. Critical Files for Debugging

### 6.1 Check These First

1. **`unsloth/models/llama.py`** (lines 890-980)
   - Is M3PO logic being called during eval?
   - Check `is_m3po_inference` value
   - Check router activation stats

2. **`trl/trainer/grpo_trainer.py`** (line ~390)
   - Verify `is_m3po_inference=False` during generation
   - Check ASLR parameters are passed correctly

3. **Evaluation script** (`c3po_gsm8k_eval.py`)
   - Check `is_m3po_inference=True` is set
   - Verify M3PO modules are loaded
   - Check ASLR config is loaded

### 6.2 Debug Questions

1. **Are M3PO modules actually being trained?**
   - Check TensorBoard for m3po/* metrics
   - Verify gradient flow (weight updates)

2. **Is router activating during training?**
   - Check `m3po/activation_rate_actual` in TensorBoard
   - Should be > 0% (not all zeros)

3. **Is evaluation using the same config?**
   - Print `model.config.aslr_num_paths` in eval script
   - Should match training config

4. **Are M3PO modules loaded during eval?**
   - Check `m3po_modules.bin` exists
   - Verify `load_state_dict` succeeds
   - Print parameter counts

---

## 7. Summary of All Modified Lines

### Transformers
- `configuration_llama.py`: Added 6 ASLR parameters
- `configuration_qwen2.py`: Added 6 ASLR parameters
- `modeling_llama.py`: Added 3 M3PO classes + init/forward modifications (~200 lines)
- `modeling_qwen2.py`: Same as llama (~200 lines)
- `generation/utils.py`: Added ASLR parameters to _sample signature + extraction (~50 lines)

### TRL
- `grpo_trainer.py`: Added 6 ASLR config params + storage + generation passing (~100 lines)

### Unsloth  
- `llama.py`: 
  - PEFT modifications (~50 lines)
  - M3PO logic in fast_forward (~150 lines)
  - M3PO logic in fast_forward_inference (~150 lines)

**Total:** ~900 lines of modifications across 8 files

---

## 8. Common Issues & Solutions

### Issue 1: Training works, eval fails
**Cause:** M3PO not loaded or `is_m3po_inference` wrong
**Fix:** Verify module loading, check inference flag

### Issue 2: Router never activates
**Cause:** Threshold too high, weights not training
**Fix:** Check router in TensorBoard, verify Float32 dtype

### Issue 3: NaN gradients
**Cause:** BFloat16/Float32 mismatch, in-place ops
**Fix:** Ensure M3PO modules are Float32, no in-place updates

### Issue 4: Slow training
**Cause:** N and T too high, inefficient collaboration
**Fix:** Reduce N=2, T=1 for faster experimentation

### Issue 5: No improvement over baseline
**Cause:** Insufficient training, wrong reward signal
**Fix:** Train longer, verify rewards are increasing

---

**End of Document**
