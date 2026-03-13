# Session 2026-03-12: Gradient Checkpointing & Temperature Fixes

## Summary of Changes

This session addressed three issues: a harmless but noisy gradient checkpointing warning during Phase 1 warmup, the temperature/rank configuration for Luong gating, and an OOM crash.

---

## Change 1: Gradient Checkpointing Warning (REVERTED)

### Problem
During Phase 1 (gating warmup, steps 0-150), PyTorch emitted:
```
torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
```

**Root cause:** The model's internal gradient checkpointing (transformer layer checkpoints) sees no grad-requiring inputs because all model params are frozen during Phase 1. The warning is harmless — M3PO's gating gradients flow through their own computation path independent of model gradients.

### What we tried
- Added `raw_model.gradient_checkpointing_disable()` after freezing model params in Phase 1
- Added `raw_model.gradient_checkpointing_enable()` + `raw_model.enable_input_require_grads()` at Phase 2 transition

### Why it was reverted
**CUDA OOM at step 78.** Without gradient checkpointing, PyTorch keeps all intermediate activations in memory. This worked for 77 steps but a longer sequence at step 78 pushed memory over the 39.5 GiB A100 limit. The error occurred at `log_softmax(blended_logits)` in `compute_log_probs_with_m3po` (line 248).

### Final state
Gradient checkpointing stays **enabled throughout training**. The warning is harmless and the memory savings are essential. Added a comment explaining why we keep it enabled.

---

## Change 2: Temperature Configuration — Remove T=1.0 Warmup

### Background (from advisor analysis of prior runs)
The advisor analyzed wandb plots from earlier Luong gating experiments:

1. **Attention Entropy** hovered at ~0.98-1.08 throughout training (max entropy for 4 paths is log(3) ≈ 1.099). The gating never learned selective attention — it behaved like uniform averaging.

2. **Gating Grad Norms** were 0.001-0.005 in original runs. The diagnosis: gradients are attenuated by the softmax bottleneck. With T=0.1 and near-equal similarities (identity init), softmax gradient is near-zero on the flat plateau.

### What was tried (chronological)

**Attempt 1: T=1.0 warmup + xavier init + rank=32**
- Advisor suggested raising temperature to 1.0 during warmup for gradient flow
- Result: **200x worse** — grad norms dropped from ~0.001 to ~5e-6
- Why: With xavier init at rank=32, similarities are tiny (~0.01-0.1). At T=1.0, `S/1.0` produces near-uniform softmax (inputs too small to differentiate). Gradients vanish for a different reason than T=0.1 (flat inputs instead of saturated outputs).

**Attempt 2: T=1.0 warmup + identity init + rank=32**
- Reverted to identity init but kept T=1.0 warmup
- Result: U grad norm = 1.005e-05, V grad norm = 9.341e-06
- Still ~100x below the ~0.001 target

**Attempt 3: T=0.1 throughout + identity init + rank=32**
- Removed T=1.0 override entirely, temperature stays at config value (0.1)
- Result: U grad norm = 3.37e-05, V grad norm = 3.40e-05
- Better (~3x improvement) but still ~30x below target
- Diagnosis: rank=32 produces smaller similarities than rank=256 because the projection captures less of the probability distribution in 151,936-dim vocab space

### Final configuration (current)
Removed all temperature annealing. Temperature is a constant 0.1 throughout training:
- Removed `gating_function.set_temperature(1.0)` at Phase 1 start
- Removed `gating_function.set_temperature(1.0)` at Phase 2 start
- Removed `temp_anneal_steps` and `target_temperature` variables
- Removed the temperature annealing block in the training loop (was: linearly decay from 1.0 → 0.1 over first 100 Phase 2 steps)

---

## Change 3: Rank 32 → 256

### Problem
Rank=32 + identity init + T=0.1 produced grad norms of ~3.4e-05 — still ~30x below the ~0.001 from original runs which used rank=256.

**Why rank matters for identity init:** `S_ij = p_i^T Q Q^T p_j` where Q has orthonormal columns. `Q Q^T` is a rank-k projection matrix. At rank=32, it captures a tiny fraction of the 151,936-dim probability distribution, producing smaller similarity magnitudes. At rank=256, the projection captures more, producing similarities in the right range for T=0.1 softmax to produce meaningful gradients.

### Change
`gating_config.rank`: 32 → 256

### Current full config
```python
'gating_config': {
    'temperature': 0.1,       # T=0.1 throughout, no annealing
    'rank': 256,              # Larger projection for stronger similarities at T=0.1
    'init_strategy': 'identity',
    'debug': False,
},
'gating_warmup_steps': 150,   # 3x original (was 50)
'gating_lr': 1e-3,            # 2x original (was 5e-4)
'gating_grad_clip': 1.0,
```
Weight decay = 0.0 for gating params in both Phase 1 and Phase 2 optimizers.

---

## Change 4: Gating Gradient Verification Diagnostic

Added a one-time diagnostic after `scaled_loss.backward()` to confirm gating params receive gradients during Phase 1:
```python
if is_main_process() and in_warmup_phase and not gating_grad_verified:
    for name, p in gating_function.named_parameters():
        if p.grad is not None:
            print(f"[M3PO] Verified: {name} grad norm = {p.grad.norm().item():.6e}")
        else:
            print(f"[M3PO] WARNING: {name} has no gradient!")
    gating_grad_verified = True
```
Flag `gating_grad_verified = False` initialized alongside `in_warmup_phase`.

---

## Change 5: torch.compile Cache Size

### Problem
`torch._dynamo` warnings about recompilation due to variable sequence lengths:
```
torch._dynamo hit config.cache_size_limit (8)
last reason: tensor 'L['___stack0']' size mismatch at index 1. expected 627, actual 606
```

### Fix
Added near top of `m3po_train.py`:
```python
torch._dynamo.config.cache_size_limit = 32  # default is 8
```

---

## Key Learnings

1. **Gradient checkpointing cannot be disabled during Phase 1** — it causes OOM on longer sequences even though no model gradients flow. The activations still need to be stored for the forward pass.

2. **Temperature and init_strategy are tightly coupled.** Identity init produces similarities in a specific magnitude range that works with T=0.1. Xavier init produces much smaller similarities that don't work well at any temperature without additional scaling.

3. **Rank affects similarity magnitude with identity init.** Higher rank = larger projection = larger similarities = stronger gradients through softmax at T=0.1. Rank 256 is needed to match the numerical regime where Luong gating showed ~0.001 grad norms.

4. **The softmax bottleneck is the core gradient challenge.** Whether from saturation (T too low, similarities too large) or uniformity (T too high or similarities too small), the softmax gradient vanishes when attention weights are near one-hot or near-uniform.

5. **Weight decay on gating params is harmful** when gradients are small — decay becomes the dominant force, pulling params away from their init without sufficient learning signal to counteract it.
