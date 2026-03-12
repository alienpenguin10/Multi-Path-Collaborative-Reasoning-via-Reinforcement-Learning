# Detailed Report: OLD M3PO vs NEW M3PO Codebase Differences

## 1. Overview

| Metric | OLD Codebase | NEW Codebase |
|--------|-------------|--------------|
| **Location** | `OLD_M3PO/` | `./M3PO/` (current) |
| **Core Python files** | 3 (`grpo_train.py`, `grpo_eval.py`, `m3po_utils.py`) | 6 core + 5 new modules |
| **Total new code added** | — | ~2,400+ lines |
| **New packages** | — | `m3po_gating/` (5 files) |
| **New test files** | — | `tests/` (3 files) |
| **Architecture change** | Monolithic | Modular with factory pattern |

---

## 2. File-by-File Comparison

---

### 2.1 `grpo_train.py` (Main Training Script)

**OLD: 746 lines → NEW: 920 lines (+174 lines, +23%)**

#### 2.1.1 Imports & Setup (Lines 1-50)

| Change | OLD | NEW |
|--------|-----|-----|
| Transformers path | Direct import (system transformers) | `sys.path.insert(0, os.path.abspath("transformers/src"))` — uses local fork |
| CUDA memory config | Not set | `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` |
| Environment variables | Hardcoded WANDB credentials | `dotenv` for `.env` file loading |
| Module imports | All functions defined inline | Imports from `utils.py`: `set_random_seed`, `extract_answer_from_model_output`, `prepare_dataset`, `evaluate_model`, `combined_reward` |
| Extra imports | `re`, `load_dataset` | `json`, `sys`, `dotenv` |

#### 2.1.2 `selective_log_softmax()` — IDENTICAL

No changes between old and new.

#### 2.1.3 `compute_log_probs()` — MINOR CHANGE

| Aspect | OLD (line 545) | NEW (line 133) |
|--------|----------------|----------------|
| Model forward call | `model(input_ids=input_ids, attention_mask=attention_mask)` | `model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1)` |

The new version passes `logits_to_keep` to the Qwen2 model's forward method, which is a **memory optimization** — it only computes logits for the last N+1 positions instead of all positions. The resulting logits for the kept positions are **numerically identical**. This is a valid optimization supported by the Qwen2 model architecture (`modeling_qwen2.py:429`).

#### 2.1.4 `compute_log_probs_with_m3po()` — ENTIRELY NEW (Lines 158-231)

**73 lines of new functionality.** This function enables gradient flow through M3PO blending during loss computation (not just during generation).

Key logic:
- Groups sequences by question (batch_size × num_generations)
- Calls `apply_m3po_to_logits()` from the M3PO utils module
- Supports optional `gating_function` parameter for learnable gates
- Creates a differentiable path: `loss → log_probs → blended_logits → gating_parameters`

**This function is only called when `use_m3po=True AND gating_function is not None AND gating_function.has_learnable_parameters`.** For baseline gating, this code path is never executed.

#### 2.1.5 `detect_thinking_phase_end()` — IDENTICAL

No changes.

#### 2.1.6 `create_completion_mask()` — IDENTICAL

No changes.

#### 2.1.7 `generate_completions()` — SIGNATURE CHANGE

| Aspect | OLD (line 643) | NEW (line 306) |
|--------|----------------|----------------|
| Parameters | 8 params | 9 params: added `gating_function=None` |
| M3PO call | `generate_with_m3po(...)` (10 args) | `generate_with_m3po(..., gating_function=gating_function)` (11 args) |

The `gating_function` is passed through to `generate_with_m3po()` but the function body is otherwise identical. When `gating_function=None`, behavior is exactly the same as old.

#### 2.1.8 `generate_rollout_data()` — SIGNATURE CHANGE

| Aspect | OLD (line 735) | NEW (line 399) |
|--------|----------------|----------------|
| Parameters | 9 params | 10 params: added `gating_function=None` |

Passes `gating_function` through to `generate_completions()`. Function body otherwise identical.

#### 2.1.9 `grpo_loss()` — SIGNIFICANT CHANGES

| Aspect | OLD (line 793) | NEW (line 458) |
|--------|----------------|----------------|
| Parameters | 8 params | 12 params: added `use_m3po=False`, `lambda_blend=0.1`, `temperature_m3po=0.1`, `gating_function=None` |
| Log prob computation | Always `compute_log_probs()` | Conditional: if learnable gating → `compute_log_probs_with_m3po()`, else → `compute_log_probs()` |
| Loss formula | Unchanged | Unchanged |
| Advantage computation | Unchanged | Unchanged |
| KL divergence | Unchanged | Unchanged |

**Critical: For baseline gating (`gating_function=None`), the condition at line 493 (`use_m3po and gating_function is not None and gating_function.has_learnable_parameters`) is False, so it falls through to the standard `compute_log_probs()` — identical behavior to old code.**

#### 2.1.10 `train_with_grpo()` — SIGNIFICANT CHANGES

| Aspect | OLD (line 910) | NEW (line 585) |
|--------|----------------|----------------|
| Parameters | 16 params | 18 params: added `gating_type='baseline'`, `gating_config=None` |

**New logic blocks added:**

1. **Gating function creation (lines 634-653):**
   ```python
   if use_m3po and gating_type != 'baseline':
       from transformers.models.qwen2.m3po_gating import create_gating_function
       gating_function = create_gating_function(gating_type, gating_config)
   ```
   When `gating_type='baseline'`, this block is skipped and `gating_function = None`.

2. **Optimizer setup (lines 673-676):**
   ```python
   params_to_optimize = list(model.parameters())
   if gating_function is not None and gating_function.has_learnable_parameters:
       params_to_optimize.extend(gating_function.parameters())
   ```
   When `gating_function=None`, optimizer has exactly the same parameters as old code.

3. **Gradient clipping (lines 721-724):**
   ```python
   all_params = list(model.parameters())
   if gating_function is not None and gating_function.has_learnable_parameters:
       all_params.extend(gating_function.parameters())
   ```
   When `gating_function=None`, clips same parameters as old code.

4. **Wandb logging (lines 734-739):** Adds gating statistics to log dict if gating_function exists.

5. **Loss call (line 717):** Passes `use_m3po`, `lambda_blend`, `temperature_m3po`, `gating_function` to `grpo_loss()`.

6. **Rollout call (line 699):** Passes `gating_function` to `generate_rollout_data()`.

#### 2.1.11 `optimize_model_memory()` — IDENTICAL

No changes.

#### 2.1.12 `reserve_gpu_memory()` — NEW (Lines 770-806)

New utility function that pre-allocates GPU memory after model loading to prevent fragmentation. Not present in old code.

#### 2.1.13 Training Config (`__main__` block)

| Setting | OLD (line 1111) | NEW (line 867) |
|---------|-----------------|----------------|
| num_iterations | 1 | 1 |
| num_steps | 500 | 500 |
| batch_size | 5 | 5 |
| num_generations | 4 | 4 |
| max_completion_length | 512 | 512 |
| beta | 0.005 | 0.005 |
| learning_rate | 5e-6 | 5e-6 |
| mu | 1 | 1 |
| epsilon | 0.1 | 0.1 |
| lambda_blend | 0.1 | 0.1 |
| temperature_m3po | 0.1 | 0.1 |
| use_m3po | True | True |
| **gating_type** | N/A | `'kl_divergence'` (NEW) |
| **gating_config** | N/A | `{'temperature': 0.1, 'debug': False}` (NEW) |

**Note:** The default `gating_type` in `grpo_train.py` is `'kl_divergence'`, but when run via `monitor_and_run.py` → `run_m3po_experiment.py`, it uses `--gating_type baseline` which overrides this.

---

### 2.2 `grpo_eval.py` (Evaluation Script)

**OLD: 222 lines → NEW: 239 lines (+17 lines, +8%)**

| Change | OLD | NEW |
|--------|-----|-----|
| Function imports | `from grpo_train import SYSTEM_PROMPT, build_prompt, ...` | `from utils import SYSTEM_PROMPT, build_prompt, ...` |
| Transformers path | Not explicitly set | `sys.path.insert(0, os.path.abspath("transformers/src"))` |
| Model path | `"grpo_finetuned_model"` | `"outputs/kl_divergence"` |
| Result logging | Print to console only | Also saves JSON with metadata (gating_type, accuracy, timestamp) |
| New imports | — | `json`, `datetime` |

**Evaluation logic itself is identical** — same `evaluate_model()` function, same generation parameters (temperature=0.7, do_sample=True), same answer extraction.

---

### 2.3 `utils.py` — NEW FILE (507 lines)

This file was **extracted from `grpo_train.py`** to enable code reuse between training, evaluation, and experiment scripts. It contains:

| Function | Lines | Source | Changes from OLD |
|----------|-------|--------|-----------------|
| `set_random_seed()` | 22-55 | OLD grpo_train.py:21-51 | None |
| `SYSTEM_PROMPT` | 68-76 | OLD grpo_train.py | None |
| `extract_answer_from_model_output()` | 78-106 | OLD grpo_train.py | None |
| `extract_answer_from_dataset()` | 108-126 | OLD grpo_train.py | None |
| `prepare_dataset()` | 136-168 | OLD grpo_train.py | None |
| `build_prompt()` | 170-186 | OLD grpo_train.py | None |
| `extract_last_number()` | 197-216 | OLD grpo_train.py | None |
| `extract_single_number()` | 218-234 | OLD grpo_train.py | None |
| `evaluate_model()` | 236-338 | OLD grpo_train.py | None |
| `correctness_reward()` | 352-390 | OLD grpo_train.py:322-360 | None |
| `format_reward()` | 392-421 | OLD grpo_train.py:362-391 | None |
| `eos_reward()` | 424-470 | OLD grpo_train.py:394-440 | None |
| `combined_reward()` | 472-507 | OLD grpo_train.py:442-477 | None |

**Every function is identical to its old counterpart.** This is a pure refactoring — no behavioral changes.

---

### 2.4 `m3po_utils.py` (M3PO Core Module)

**OLD: 366 lines → NEW: 521 lines (+155 lines, +42%)**

Located at: `transformers/src/transformers/models/qwen2/m3po_utils.py`

#### 2.4.1 New Import

```python
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # NEW
```

#### 2.4.2 `M3POConfig` dataclass — IDENTICAL

No changes.

#### 2.4.3 `compute_cross_path_attention()` — EXTENDED

| Aspect | OLD (line 40) | NEW (line 41) |
|--------|---------------|---------------|
| Parameters | 3: `output_distributions`, `thinking_mask`, `temperature` | 4: added `gating_function=None` |

**New conditional logic (lines 73-83):**
```python
if gating_function is not None:
    similarity_matrix = gating_function.compute_similarity_matrix(output_distributions, hidden_states=None)
    attention_weights, _ = gating_function.compute_attention_weights(similarity_matrix, thinking_mask, mask_diagonal=True)
    return attention_weights
else:
    # Original cosine similarity logic (unchanged)
```

When `gating_function=None`, the else branch executes the **exact same cosine similarity logic** as the old code.

#### 2.4.4 `blend_token_embeddings()` — MINOR FIX

| Aspect | OLD (line 134) | NEW (line 152) |
|--------|----------------|----------------|
| Matrix multiply | `torch.mm(attention_weights, token_embeddings)` | `torch.mm(attention_weights.to(token_embeddings.dtype), token_embeddings)` |

Added `.to(token_embeddings.dtype)` to ensure type compatibility in mixed-precision training. This prevents potential dtype mismatch errors when attention_weights (float32) and token_embeddings (bfloat16) differ. The blending logic is otherwise identical.

#### 2.4.5 `apply_m3po_step()` — EXTENDED

| Aspect | OLD (line 151) | NEW (line 168) |
|--------|----------------|----------------|
| Parameters | 5 params | 6 params: added `gating_function=None` |

Passes `gating_function` through to `compute_cross_path_attention()`. All other logic identical.

#### 2.4.6 `apply_m3po_to_logits()` — ENTIRELY NEW (Lines 254-384)

**130 lines of new functionality.** This is the key addition enabling gradient flow through M3PO during loss computation (Phase 3 of the gating infrastructure).

**Purpose:** During training, the standard path computes log probs from raw model logits. This function blends logits across paths using M3PO attention before computing log probs, creating a differentiable path to learnable gating parameters.

**Two code paths:**

1. **Custom gating with gradient checkpointing (lines 293-346):** For learnable gating functions. Processes in chunks of 32 positions, using `torch.utils.checkpoint` to trade compute for memory.

2. **Baseline vectorized path (lines 348-384):** For parameter-free gating or cosine similarity. Processes all positions in parallel using `torch.bmm` — much faster (~512× vs position loop).

**Blending formula per position t:**
```
blended_i[t] = (1-λ) * logits_i[t] + λ * Σ_j A_ij * logits_j[t]
```

#### 2.4.7 `generate_with_m3po()` — EXTENDED

| Aspect | OLD (line 235) | NEW (line 388) |
|--------|----------------|----------------|
| Parameters | 10 params | 11 params: added `gating_function=None` |

Passes `gating_function` through to `apply_m3po_step()`. All generation logic identical.

#### 2.4.8 Exports

```python
# OLD __all__
["M3POConfig", "compute_cross_path_attention", "blend_token_embeddings", "apply_m3po_step", "generate_with_m3po"]

# NEW __all__ — added:
"apply_m3po_to_logits"
```

---

### 2.5 `m3po_gating/` — ENTIRELY NEW PACKAGE

This package implements a **modular gating function framework** with 6 alternative similarity measures for M3PO cross-path attention.

#### 2.5.1 `__init__.py` (20 lines)

Package initialization. Exports `BaseM3POGating`, `create_gating_function`, `GATING_REGISTRY`, `register_gating_function`. Triggers auto-registration of all gating implementations.

#### 2.5.2 `base.py` (221 lines) — Abstract Base Class

`BaseM3POGating(nn.Module)` provides:

| Method | Type | Purpose |
|--------|------|---------|
| `__init__(config)` | Concrete | Stores config, temperature, debug flag, initializes stats |
| `has_learnable_parameters` | Property | Returns False (override in learnable subclasses) |
| `compute_similarity_matrix(output_distributions, hidden_states)` | **Abstract** | Each subclass must implement |
| `compute_attention_weights(similarity_matrix, thinking_mask, mask_diagonal)` | Concrete | Temperature-scaled softmax with masking, NaN handling, stats tracking |
| `_update_stats(attention_weights, similarity_matrix)` | Concrete | Tracks entropy, max weight, variance of attention |
| `get_stats_summary()` | Concrete | Returns aggregated stats for wandb logging |
| `reset_stats()` | Concrete | Clears accumulated statistics |

**Attention weight computation** uses the same algorithm as the baseline (temperature-scaled softmax with diagonal and inactive-path masking) but adds `torch.where()` for more robust NaN handling.

#### 2.5.3 `factory.py` (87 lines) — Registry Factory

| Function | Purpose |
|----------|---------|
| `create_gating_function(gating_type, config)` | Factory that creates gating instances by name |
| `register_gating_function(name, cls)` | Register new gating implementations |
| `list_gating_functions()` | List all registered names |

Registry contents after auto-registration:
```python
{
    "baseline": None,           # Original cosine similarity (no object created)
    "raw_dot": RawDotProductGating,
    "scaled_dot": ScaledDotProductGating,
    "kl_divergence": KLDivergenceGating,
    "bhattacharyya": BhattacharyyaGating,
    "luong": LuongAttentionGating,
    "bahdanau": BahdanauAttentionGating,
}
```

#### 2.5.4 `parameter_free_gates.py` (218 lines) — 4 Parameter-Free Variants

| Class | Formula | Key Property |
|-------|---------|-------------|
| `RawDotProductGating` | `S_ij = p_i · p_j` | Confidence-weighted, no normalization. Range [0,1] for probability distributions |
| `ScaledDotProductGating` | `S_ij = (p_i · p_j) / √vocab_size` | Transformer-style scaling. Prevents large values with large vocabularies |
| `KLDivergenceGating` | `S_ij = 1 - √JSD(p_i ∥ p_j)` | Jensen-Shannon divergence (symmetric). Most theoretically principled. Slowest: O(N² × vocab) |
| `BhattacharyyaGating` | `S_ij = Σ√(p_i · p_j)` | Probability overlap coefficient. Efficient: O(vocab) per pair |

All have `has_learnable_parameters = False`.

#### 2.5.5 `learnable_gates.py` (217 lines) — 2 Learnable Variants

| Class | Formula | Parameters | Size (Qwen2.5 vocab=151936) |
|-------|---------|------------|---------------------------|
| `LuongAttentionGating` | `S_ij = p_i^T (U·V^T) p_j` | U: (vocab, rank), V: (vocab, rank) | ~38.9M params (rank=128) |
| `BahdanauAttentionGating` | `S_ij = v^T tanh(W1·p_i + W2·p_j)` | W1, W2: (attn_dim, vocab), v: (attn_dim,) | ~77.8M params (attn_dim=256) |

Both have `has_learnable_parameters = True`. Xavier initialization. These enable gradient flow: `loss → blended_logits → attention_weights → similarity_matrix → gating_parameters`.

---

### 2.6 `run_m3po_experiment.py` — NEW FILE (383 lines)

Comprehensive experiment runner that replaces the `__main__` block of `grpo_train.py` for systematic experiments.

| Feature | Details |
|---------|---------|
| **Experiment variants** | 8 types: `none`, `baseline`, `raw_dot`, `scaled_dot`, `kl_divergence`, `bhattacharyya`, `luong`, `bahdanau` |
| **Multi-trial support** | `--num_trials N` runs each variant N times with different seeds (seed = seed_base + trial - 1) |
| **Resume-friendly** | Skips experiments where `results.json` already exists (unless `--force`) |
| **Output structure** | `outputs/{gating_type}/trial_{n}/` — model, tokenizer, training_config.json, results.json |
| **Evaluation** | Full GSM8K test set evaluation after training |
| **Summary table** | Prints mean accuracy ± std per variant |
| **CLI args** | `--gating_type`, `--trial`, `--run_all`, `--num_steps`, `--batch_size`, `--num_generations`, `--max_completion_length`, `--eval_only`, `--force` |

---

### 2.7 `monitor_and_run.py` — NEW FILE (152 lines)

GPU monitoring script for automatic experiment launch.

| Feature | Details |
|---------|---------|
| **Polls** | `nvidia-smi` every 60-240 seconds |
| **Trigger** | Launches when ≥4 GPUs are free (0% util, <500MB memory) |
| **Action** | Runs `run_m3po_experiment.py` with configured args |
| **Config** | `REQUIRED_FREE_GPUS=4`, `EXPERIMENT_ARGS=["--gating_type", "baseline", "--trial", "1"]` |

---

### 2.8 `analyze_gating_results.py` — NEW FILE (437 lines)

Statistical analysis and visualization script.

| Feature | Details |
|---------|---------|
| **Data loading** | Reads all `results.json` from `outputs/` directory tree |
| **Summary table** | Mean accuracy ± std per variant, sorted by performance |
| **Statistical tests** | Pairwise Welch's t-tests with Cohen's d effect sizes |
| **Plots** | Bar charts with error bars, box plots, significance heatmaps, convergence curves |

---

### 2.9 `tests/` — NEW DIRECTORY (3 test files)

| File | Purpose |
|------|---------|
| `test_gating_infrastructure.py` | Tests all 7 gating functions: creation, similarity computation, attention weights, masking, stats |
| `test_gradient_flow.py` | Tests gradient flow through M3PO blending to learnable parameters, numerical stability |
| `test_learnable_gates.py` | Tests learnable gating specifically: parameter updates, optimizer integration, full-scale param counts |

---

### 2.10 `M3PO_GATING_PROGRESS.md` — NEW FILE (338 lines)

Implementation tracking document covering 5 phases of gating infrastructure development, architecture details, usage examples, and known limitations.

---

## 3. Behavioral Differences Summary

### 3.1 When Running with `gating_type='baseline'`

**The new code takes the EXACT SAME code path as the old code.** All conditionals gate on `gating_function is not None` or `gating_type != 'baseline'`, which are both False for baseline. The only non-behavioral difference is the `logits_to_keep` memory optimization in `compute_log_probs()`.

### 3.2 When Running with Non-Baseline Gating

The new code activates additional functionality:

| Feature | Baseline | Non-Baseline (parameter-free) | Non-Baseline (learnable) |
|---------|----------|-------------------------------|--------------------------|
| Similarity computation | Cosine similarity (inline) | Via gating_function.compute_similarity_matrix() | Same + gradient flow |
| Attention computation | Temperature softmax (inline) | Via gating_function.compute_attention_weights() | Same |
| Generation blending | embed space via generate_with_m3po | Same + gating_function passed through | Same |
| Loss-time blending | Not applied | Not applied | compute_log_probs_with_m3po() with apply_m3po_to_logits() |
| Optimizer | Model params only | Model params only | Model params + gating params |
| Gradient clipping | Model params only | Model params only | Model params + gating params |
| Wandb logging | Standard metrics | + gating statistics | + gating statistics |

---

## 4. Paper vs Implementation Differences

Both OLD and NEW codebases share the same deviations from the M3PO paper (Table 3):

| Setting | Paper | Both Implementations |
|---------|-------|---------------------|
| Fine-tuning | LoRA (rank=32, α=64) | Full fine-tuning |
| Framework | Unsloth | Custom training loop |
| LR scheduler | Cosine with warmup (ratio=0.1) | Constant LR |
| Gradient accumulation | 4 steps | None |
| Effective batch size | 64 | 20 (5×4) |
| Group size (GSM8K) | 8 generations | 4 generations |
| Completion length | 1024 | 512 |
| Prompt format | Qwen chat template, `####` delimiter | Raw text, `<reasoning>/<answer>` XML |
| Evaluation decoding | Greedy | temperature=0.7 sampling |
| Optimizer | AdamW 8-bit | AdamW standard |

The paper reports no seeds, error bars, or variance — results appear to be single runs. Paper Table 2 shows Qwen2.5-1.5B on GSM8K: GRPO=68.2%, M3PO=70.2%.

---

## 5. Summary of What Changed and Why

| Category | What Changed | Why |
|----------|-------------|-----|
| **Refactoring** | Functions extracted to `utils.py` | Enable code reuse across train/eval/experiment scripts |
| **Modularity** | Gating function framework (`m3po_gating/`) | Research infrastructure for comparing alternative similarity measures |
| **Differentiability** | `apply_m3po_to_logits()` + `compute_log_probs_with_m3po()` | Enable gradient flow to learnable gating parameters during training |
| **Experiment infra** | `run_m3po_experiment.py`, `monitor_and_run.py`, `analyze_gating_results.py` | Systematic multi-trial experiments with statistical analysis |
| **Testing** | `tests/` directory | Validate gating functions, gradient flow, numerical stability |
| **Memory** | `logits_to_keep` optimization, `reserve_gpu_memory()` | Reduce VRAM usage on multi-GPU setups |
| **Backward compat** | All new params default to `None`/`'baseline'` | Old behavior preserved when not using new features |
