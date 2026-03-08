# Phase 3: Learnable Gating Infrastructure — Completion Report

## Objective

Create a differentiable path through M3PO logit blending so that learnable gating parameters (Phase 4: Luong/Bahdanau attention) can receive gradients during the GRPO loss computation forward pass. Previously, M3PO was only applied during generation under `@torch.no_grad()`, making it impossible for any learnable gating parameters to train.

**Gradient path created:**
```
loss → log_probs → blended_logits → attention_weights → similarity_matrix → gating_parameters
```

## What Was Implemented

### 1. `apply_m3po_to_logits()` — New function in `m3po_utils.py`

**Location:** `transformers/src/transformers/models/qwen2/m3po_utils.py`, lines 253–367

**Purpose:** Core differentiable function that blends logits across paths within a single question group during loss computation. This is the training-time analogue of the generation-time embedding blending.

**Blending formula (per position t, per question group):**
```
1. p_i = softmax(logits_i[t])                                    # output distributions
2. S_ij = gating_fn.compute_similarity_matrix(p)                  # differentiable similarity
3. A_ij = softmax(S_ij / T)                                       # attention weights
4. blended_i[t] = (1 - λ) * logits_i[t] + λ * Σ_j A_ij * logits_j[t]  # logit blending
```

**Two code paths by design:**

| Path | When | Strategy | Rationale |
|------|------|----------|-----------|
| Baseline (cosine) | `gating_function=None` | Vectorized with `torch.bmm` across all positions at once | Reshapes to `(seq_len, N, vocab_size)`, computes batched similarity `(seq_len, N, N)`, blends all positions in parallel. Fast. |
| Custom gating | `gating_function` provided | Per-position loop, calling gating API each iteration | The gating API (`compute_similarity_matrix`, `compute_attention_weights`) expects `(N, vocab_size)` per position. Looping over ~512 positions with small tensors is acceptable for Phase 3. |

**Additional features:**
- `completion_mask` parameter: skips fully-masked positions (all zeros in a column) to avoid wasted computation on padding tokens
- Input validation: raises `ValueError` if `logits.shape[0] != num_generations`
- Debug output controlled by `M3PO_DEBUG=1` environment variable

---

### 2. `compute_log_probs_with_m3po()` — New function in `grpo_train.py`

**Location:** `grpo_train.py`, lines 154–225

**Purpose:** Replaces `compute_log_probs()` when M3PO is active during loss computation. The key difference is chunking strategy.

**Original `compute_log_probs()`:** Chunks by `chunk_size=2` (arbitrary pairs of sequences). This breaks M3PO because cross-path interaction requires all N paths from the same question to be processed together.

**New `compute_log_probs_with_m3po()`:** Chunks by **question group** — all N paths for a single question are processed as one unit. For each group:
1. Forward pass through model → logits `(N, logits_to_keep, vocab_size)`
2. `apply_m3po_to_logits()` → blended logits (differentiable)
3. `log_softmax` → `gather` → per-token log probabilities

**Memory impact:** With N=4 paths per group (vs chunk_size=2), each chunk is ~2x larger. On A100 80GB, this is well within budget.

---

### 3. Modified `grpo_loss()` — Conditional M3PO path

**Location:** `grpo_train.py`, lines 454–455, 489–498

**Changes:**
- Added parameters: `use_m3po`, `lambda_blend`, `temperature_m3po`, `gating_function`
- Added conditional routing at the `token_log_probs` computation:

```python
if use_m3po and gating_function is not None:
    token_log_probs = compute_log_probs_with_m3po(...)
else:
    token_log_probs = compute_log_probs(...)  # original path
```

**Design decision — only `token_log_probs` gets M3PO:**
- `old_log_probs`: Computed during rollout generation (frozen, detached). No gradients needed.
- `ref_log_probs`: Computed from reference model (frozen). No gradients needed.
- `token_log_probs`: Computed from current policy model. This is where gradients flow, so this is the only one that needs M3PO blending.

**Activation scope:** M3PO during loss activates when `use_m3po=True AND gating_function is not None`. When `gating_function=None` (i.e., baseline cosine gating type), the original `compute_log_probs()` path is used. This means baseline behavior is completely unchanged — zero risk of regression.

---

### 4. Modified `train_with_grpo()` — Learnable parameter support

**Location:** `grpo_train.py`, lines 669–734

Four changes were made:

**4a. M3PO params forwarded to `grpo_loss()`** (lines 710–713):
```python
loss, avg_reward = grpo_loss(
    ..., use_m3po=use_m3po, lambda_blend=lambda_blend,
    temperature_m3po=temperature_m3po, gating_function=gating_function,
)
```

**4b. Learnable gating params in optimizer** (lines 669–672):
```python
params_to_optimize = list(model.parameters())
if gating_function is not None and gating_function.has_learnable_parameters:
    gating_function = gating_function.to(device)
    params_to_optimize.extend(gating_function.parameters())
optimizer = torch.optim.AdamW(params_to_optimize, ...)
```
The gating function is moved to the correct device, and its parameters are appended to the optimizer's parameter list. This runs once per outer iteration (when the optimizer is reinitialized).

**4c. Learnable gating params in gradient clipping** (lines 717–720):
```python
all_params = list(model.parameters())
if gating_function is not None and gating_function.has_learnable_parameters:
    all_params.extend(gating_function.parameters())
torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.1)
```

**4d. Gating statistics logged to wandb** (lines 730–734):
```python
if gating_function is not None:
    stats = gating_function.get_stats_summary()
    for key, value in stats.items():
        log_dict[f"m3po/{key}"] = value
    gating_function.reset_stats()
wandb.log(log_dict)
```
Statistics are accumulated during the forward pass (similarity computation triggers `_update_stats()` in the base class), then flushed to wandb and reset after each GRPO iteration. Logged keys include `m3po/attention_entropy_mean`, `m3po/attention_max_mean`, `m3po/similarity_mean_mean`, etc.

---

### 5. Bug Fix: KL Divergence Gradient Stability

**Location:** `transformers/src/transformers/models/qwen2/m3po_gating/parameter_free_gates.py`, line 161

**Problem:** `torch.sqrt(jsd.clamp(min=0))` produces infinite gradient when `jsd=0` (which happens on the diagonal where a distribution is compared with itself). Even though diagonal entries are later masked out by `compute_attention_weights()`, they still exist in the computation graph and propagate NaN gradients backward.

**Fix:** Changed clamp lower bound from `0` to `1e-12`:
```python
# Before:
similarity = 1.0 - torch.sqrt(jsd.clamp(min=0))

# After:
similarity = 1.0 - torch.sqrt(jsd.clamp(min=1e-12))
```

This ensures `sqrt` always operates on a strictly positive value, giving a finite gradient. The impact on forward-pass accuracy is negligible (`sqrt(1e-12) = 1e-6`).

---

### 6. Test Suite: `test_gradient_flow.py`

**Location:** `test_gradient_flow.py` (new file, 270 lines)

Six tests covering all aspects of the gradient flow infrastructure:

| # | Test | What It Validates | Key Assertions |
|---|------|-------------------|----------------|
| 1 | Gradient flow to logits (baseline) | Backward through vectorized cosine path | `logits.grad is not None`, no NaN/Inf, non-zero |
| 2 | Gradient flow to learnable params | Backward through custom gating with `nn.Parameter` | `W.grad is not None`, no NaN, non-zero |
| 3 | Numerical stability | Extreme logit values: peaked, uniform, large, mixed | No NaN/Inf in output or gradients |
| 4 | Backward compatibility | `lambda_blend=0` produces identity | Max diff < 1e-6 with lambda=0; mean diff > 1e-6 with lambda=0.1 |
| 5 | All gating functions | Every registered gating function through `apply_m3po_to_logits` | Shape, values, gradient correctness for all 5 types |
| 6 | Completion mask | Masked positions stay unchanged | Zero diff on masked positions, non-zero diff on unmasked |

The test includes a `MockLearnableGating` class (bilinear: `S_ij = p_i^T W p_j` with `W` as `nn.Parameter`) to validate gradient flow to learnable parameters without needing the actual Luong/Bahdanau implementations from Phase 4.

**All 6 tests pass.** The existing `test_gating_infrastructure.py` (Phase 1-2) also continues to pass with all 5 gating types.

---

## Files Changed

| File | Type | Lines Changed | Summary |
|------|------|---------------|---------|
| `transformers/src/transformers/models/qwen2/m3po_utils.py` | Modified | +118 | Added `apply_m3po_to_logits()`, updated `__all__` |
| `grpo_train.py` | Modified | +85 | Added `compute_log_probs_with_m3po()`, modified `grpo_loss()` signature + routing, modified `train_with_grpo()` optimizer/clipping/logging |
| `transformers/src/transformers/models/qwen2/m3po_gating/parameter_free_gates.py` | Modified | +1 | Fixed `sqrt(0)` gradient issue in KL divergence |
| `test_gradient_flow.py` | New | +270 | 6 gradient flow tests with mock learnable gating |
| `M3PO_GATING_PROGRESS.md` | Modified | +20 | Phase 3 marked complete, progress updated to 53% |

## Design Decisions & Rationale

### 1. Logit blending (not embedding blending) during training

During generation, M3PO blends **token embeddings** (as per the paper). During loss computation, we blend **logits** instead. This is the natural differentiable equivalent:
- Generation: `h_i = (1-λ)*e_i + λ*c_i` where `e_i` is the token embedding
- Training: `blended_i[t] = (1-λ)*logits_i[t] + λ*Σ_j A_ij*logits_j[t]` where `logits_i[t]` are model outputs

Both achieve the same goal — cross-path information exchange — but logit blending during training preserves the differentiable chain from loss back to gating parameters.

### 2. Activation scope: `use_m3po=True AND gating_function is not None`

Baseline (cosine similarity with `gating_function=None`) stays on the original `compute_log_probs()` path. This ensures:
- Zero behavior change for existing baseline experiments
- No performance overhead when not using custom gating
- Clear separation: M3PO during loss is only for custom/learnable gating functions

### 3. Thinking mask omitted during training

All completion positions receive M3PO blending. The generation-time thinking mask (detecting `</reasoning>` tokens) doesn't directly translate to loss computation where all tokens are processed in parallel. This simplification is acceptable for Phase 3; a training-time thinking mask can be added in Phase 5 if needed.

### 4. Only `token_log_probs` modified

The `old_log_probs` (from rollout, detached) and `ref_log_probs` (from frozen reference model) are left unchanged. Since gradients only flow through `token_log_probs`, this is sufficient. Full consistency (applying M3PO to all log prob computations) can be explored in Phase 5 if experiments suggest it matters.

### 5. Chunking by question group

Original `compute_log_probs()` processes 2 sequences at a time (arbitrary chunking). The M3PO version processes N=4 sequences at a time (one question group). Memory increases ~2x per chunk but remains well within A100 80GB budget. The alternative — processing all `batch_size * N` sequences at once — would risk OOM.

## Test Results

```
M3PO Phase 3: Gradient Flow Tests
============================================================
  ✓ PASS: Gradient flow to logits
  ✓ PASS: Gradient flow to learnable params
  ✓ PASS: Numerical stability
  ✓ PASS: Backward compatibility
  ✓ PASS: All gating functions
  ✓ PASS: Completion mask

✓ All gradient flow tests passed!
```

```
M3PO Gating Infrastructure Test (Phase 1-2, backward compat)
============================================================
  ✓ PASS: baseline
  ✓ PASS: raw_dot
  ✓ PASS: scaled_dot
  ✓ PASS: kl_divergence
  ✓ PASS: bhattacharyya

✓ All tests passed!
```

## What This Enables (Phase 4)

With the gradient infrastructure in place, Phase 4 can now implement learnable gating functions that will automatically receive gradients during training:

1. **Luong Attention** (`S_ij = p_i^T W p_j`): The weight matrix `W` is an `nn.Parameter`. Gradients flow through `apply_m3po_to_logits()` → `compute_similarity_matrix()` → `W`.

2. **Bahdanau Attention** (`S_ij = v^T tanh(W1*p_i + W2*p_j)`): Parameters `v`, `W1`, `W2` are `nn.Parameter`s. Same gradient path.

Both only need to:
- Inherit from `BaseM3POGating`
- Implement `compute_similarity_matrix()`
- Override `has_learnable_parameters` to return `True`
- Register in the factory

The training loop (`train_with_grpo`) will automatically:
- Include their parameters in the optimizer
- Include their parameters in gradient clipping
- Log their attention statistics to wandb

## Remaining Verification

The following verification steps from the plan require GPU access and are deferred to when training runs are scheduled:

- [ ] Short training run (5-10 steps) with `gating_type='raw_dot'` — verify loss decreases, no crashes
- [ ] Verify wandb logs include `m3po/*` statistics during actual training
