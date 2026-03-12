# Experimental Conditions & WandB Metrics

This document catalogues every experiment that can be run from the codebase, the configurable hyperparameters, and the exact WandB metrics logged per condition.

---

## All Experimental Conditions

### From `grpo_train_single.py` — Gating Function Investigation

7 gating function conditions via `training_config['gating_type']`:

**Parameter-Free (no learnable params):**

1. **`'baseline'`** — Original cosine similarity (M3PO paper default). The **control condition**.
2. **`'raw_dot'`** — Raw dot-product `S_ij = p_i · p_j`. Confidence-weighted, no L2 normalisation.
3. **`'scaled_dot'`** — Scaled dot-product `S_ij = (p_i · p_j) / √d`. Transformer-style scaling for numerical stability.
4. **`'kl_divergence'`** — Jensen-Shannon Divergence similarity `S_ij = 1 - √JSD(p_i ‖ p_j)`. Information-theoretic, most principled measure.
5. **`'bhattacharyya'`** — Bhattacharyya coefficient `S_ij = Σ√(p_i · p_j)`. Distributional overlap measure.

**Learnable (jointly trained with model):**

6. **`'luong'`** — Bilinear `S_ij = p_i^T (U V^T) p_j`, low-rank factorised (~38.9M params at rank=256). Configurable `rank` and `init_strategy` (`'identity'` vs `'xavier'`).
7. **`'bahdanau'`** — Additive/MLP `S_ij = v^T tanh(W1·p_i + W2·p_j)` (~77.8M params at attn_dim=256). Captures nonlinear relationships.

**Ablation baseline:**

8. **`use_m3po=False`** — Pure GRPO, no cross-path interaction. Quantifies M3PO's contribution in isolation.

### From `old_alambda/grpo_train.py` — Adaptive Entropy-Gated Lambda

9. **Adaptive lambda** — Uses cosine similarity gating but with **per-trajectory, per-step dynamic λ** instead of fixed λ=0.1. Triggered by setting `lambda_blend=None`. The formula:

```
λ_i = λ_min + (λ_max - λ_min) · σ((H_i - μ_H) / (σ_H · τ_H))
```

Where H_i is the output entropy of path i, and μ_H/σ_H are running EMA statistics maintained by an `EntropyTracker`. Paths with **high entropy** (uncertain predictions) receive **more cross-path blending**, while confident paths get less. Additional configurable parameters: `lambda_min=0.01`, `lambda_max=0.3`, `tau_H=1.0`, `entropy_ema_decay=0.95`.

---

## Summary Table: 9 Conditions

| # | Condition | Script | Key Variable | Learnable? |
|---|---|---|---|---|
| 1 | No M3PO (pure GRPO) | `grpo_train_single.py` | `use_m3po=False` | No |
| 2 | Cosine similarity (baseline) | `grpo_train_single.py` | `gating_type='baseline'` | No |
| 3 | Raw dot-product | `grpo_train_single.py` | `gating_type='raw_dot'` | No |
| 4 | Scaled dot-product | `grpo_train_single.py` | `gating_type='scaled_dot'` | No |
| 5 | Jensen-Shannon Divergence | `grpo_train_single.py` | `gating_type='kl_divergence'` | No |
| 6 | Bhattacharyya coefficient | `grpo_train_single.py` | `gating_type='bhattacharyya'` | No |
| 7 | Luong (bilinear) | `grpo_train_single.py` | `gating_type='luong'` | Yes |
| 8 | Bahdanau (additive) | `grpo_train_single.py` | `gating_type='bahdanau'` | Yes |
| 9 | Adaptive entropy-gated λ | `old_alambda/grpo_train.py` | `lambda_blend=None` | No |

Conditions 1–8 use fixed λ=0.1. Condition 9 uses cosine similarity but with dynamic per-path λ ∈ [0.01, 0.3].

---

## Configurable Hyperparameters Per Experiment

Beyond gating type, each run can vary:

| Parameter | Default | What it controls |
|---|---|---|
| `lambda_blend` | 0.1 | M3PO blending coefficient λ — how much cross-path signal is mixed in |
| `temperature_m3po` | 0.1 | Attention temperature T — sharpness of attention weights |
| `num_generations` | 4 | Number of paths (N) per prompt — more paths = richer collaboration |
| `beta` | 0.005 | KL penalty coefficient — regularisation strength |
| `learning_rate` | 5e-6 | Model LR |
| `gating_lr` | 5e-4 (100× model LR) | Learnable gating LR (Luong/Bahdanau only) |
| `gating_warmup_steps` | 50 | Steps to pre-train gating before unfreezing model (learnable only) |
| `gating_grad_clip` | 1.0 | Separate gradient clipping for gating parameters |
| `gating_config.rank` | 256 | Low-rank dimension for Luong |
| `gating_config.init_strategy` | `'identity'` | Luong initialisation: `'identity'` (QR-based) vs `'xavier'` |
| `gradient_accumulation_steps` | 4 | Effective batch size multiplier |
| `warmup_ratio` | 0.1 | Cosine schedule warmup fraction |

**Adaptive lambda additional parameters** (`old_alambda/grpo_train.py` only):

| Parameter | Default | What it controls |
|---|---|---|
| `lambda_min` | 0.01 | Floor for adaptive λ |
| `lambda_max` | 0.3 | Ceiling for adaptive λ |
| `tau_H` | 1.0 | Sharpness of entropy gating sigmoid |
| `entropy_ema_decay` | 0.95 | Smoothing for entropy running statistics |

---

## WandB Metrics Logged — Exact Breakdown

### Core metrics (every run, every step)

| WandB key | Source | Type | What it measures |
|---|---|---|---|
| `loss` | `grpo_loss()` return | float | GRPO surrogate loss (PPO clipped objective − β·KL). **Training stability indicator.** |
| `average_reward` | `grpo_loss()` return | float | Mean combined reward across all completions in the batch (correctness 0–2.0 + format 0–0.8). **Convergence speed indicator.** |
| `learning_rate` | `scheduler.get_last_lr()[0]` | float | Current LR from cosine schedule with warmup. |
| `iteration` | loop counter | int | Outer iteration (reference model update cycle). |
| `step` | loop counter | int | Training step within the iteration. |
| `grpo_iter` | loop counter | int | GRPO policy update iteration within the step (1 to μ). |

### Gating function statistics (conditions 3–8, when `gating_function is not None`)

These come from `BaseM3POGating.get_stats_summary()` which computes mean and std across all generation steps since last reset. Logged under the `m3po/` prefix:

| WandB key | Source | What it measures |
|---|---|---|
| `m3po/attention_entropy_mean` | `_update_stats()` | Mean entropy of attention weight distributions. **Higher = more uniform blending across paths.** |
| `m3po/attention_entropy_std` | `_update_stats()` | Std of attention entropy across generation steps. |
| `m3po/attention_max_mean` | `_update_stats()` | Mean of per-path max attention weight. **Higher = one path dominates.** |
| `m3po/attention_max_std` | `_update_stats()` | Std of attention max. |
| `m3po/attention_variance_mean` | `_update_stats()` | Mean variance of attention weights per path. **Measures spread/focus of attention.** |
| `m3po/attention_variance_std` | `_update_stats()` | Std of attention variance. |
| `m3po/similarity_mean_mean` | `_update_stats()` | Mean of pairwise similarity values between paths. **Key metric for H₃ — reveals magnitude bias.** |
| `m3po/similarity_mean_std` | `_update_stats()` | Std of similarity mean. |
| `m3po/similarity_std_mean` | `_update_stats()` | Mean of similarity std (how spread out similarities are). |
| `m3po/similarity_std_std` | `_update_stats()` | Std of similarity std. |

**Note**: Condition 1 (pure GRPO, `use_m3po=False`) and condition 2 (`baseline`) log **no** `m3po/*` metrics — the baseline uses the hardcoded cosine similarity in `m3po_utils.py` directly, bypassing the gating framework and its stats tracking.

### Learnable gating only (conditions 7–8, Luong/Bahdanau)

| WandB key | Source | What it measures |
|---|---|---|
| `m3po/gating_grad_norm` | Computed inline in training loop | L2 norm of gradients for gating parameters. **Monitors gating training health.** |

### Adaptive lambda only (condition 9, `old_alambda/grpo_train.py`)

These come from `generate_with_m3po()` aggregated across all generation steps per rollout, passed through `rollout_data["m3po_stats"]`:

| WandB key | Source | What it measures |
|---|---|---|
| `m3po/lambda_mean` | `apply_m3po_step()` | Mean adaptive λ across all paths and generation steps. **Shows average blending strength.** |
| `m3po/lambda_min` | `apply_m3po_step()` | Minimum λ observed (most confident path). |
| `m3po/lambda_max` | `apply_m3po_step()` | Maximum λ observed (most uncertain path). |
| `m3po/active_steps` | Step counter | Number of generation steps where M3PO blending was active (paths still in thinking phase). |

### Post-training (once per run)

| Metric | Source | What it measures |
|---|---|---|
| GSM8K test accuracy | `evaluate_model()` after training | Final reasoning accuracy on held-out test set. **Primary outcome metric.** |

---

## Suggested Experiment Plan for Hypotheses

(parameter-free alternatives differ from cosine): Run `baseline`, `raw_dot`, `scaled_dot`, `kl_divergence`, `bhattacharyya` — all with identical hyperparameters. Compare `loss`, `average_reward` curves, and final accuracy.

(learnable outperform parameter-free): Run `luong` and `bahdanau` alongside the parameter-free set. You can also ablate `rank` (e.g., 128 vs 256 vs 512) and `init_strategy` for Luong. Monitor `m3po/gating_grad_norm` to verify learning is healthy.

(cosine normalisation bias): Compare `baseline` vs `raw_dot` directly — the only difference is L2 normalisation. The `m3po/similarity_mean_mean` and `m3po/attention_max_mean` metrics should reveal whether cosine favours high-magnitude paths.

**Ablation baseline**: Run with `use_m3po=False` to quantify M3PO's contribution independent of gating choice.

**Adaptive lambda investigation**: Run condition 9 and compare with fixed-λ cosine (condition 2). The `m3po/lambda_mean`, `m3po/lambda_min`, `m3po/lambda_max` metrics show how blending adapts during training.

**Sensitivity studies** (if time permits): Vary `lambda_blend` (e.g., 0.05, 0.1, 0.2) or `temperature_m3po` (e.g., 0.05, 0.1, 0.5) for the best-performing gating function.


 every gating function operates on probability distributions, never on raw tokens.                                                                                                                              
                                                                                                                                                                                                                            
  Here's the exact code path:                                                                                                                                                                                               
                                                                                                                                                                                                                            
  1. Logits → softmax → probability distribution at m3po_utils.py:209:                                                                                                                                                      
  output_distributions = F.softmax(logits, dim=-1)  # (batch_size * N, vocab_size)                                                                                                                                          
  2. Those distributions are passed to compute_cross_path_attention() at line 231:
  attention = compute_cross_path_attention(
      output_distributions=batch_dists,  # (N, vocab_size) — softmax probabilities
      ...
      gating_function=gating_function,
  )
  3. Every gating function receives output_distributions — the (N, vocab_size) softmax probability vectors. The signature in base.py enforces this:
  def compute_similarity_matrix(self, output_distributions: torch.Tensor, hidden_states=None)

    - Baseline cosine: norm_dists = output_distributions / ||output_distributions||, then S = norm_dists @ norm_dists^T
    - Raw dot: S = output_distributions @ output_distributions^T
    - Scaled dot: same but divided by √vocab_size
    - JSD: computes KL(p_i || m) + KL(q_i || m) where p, q are the probability distributions
    - Bhattacharyya: √output_distributions @ √output_distributions^T
    - Luong: output_distributions @ U and output_distributions @ V, then bilinear product
    - Bahdanau: W1 @ output_distributions^T and W2 @ output_distributions^T, then additive attention

  No gating function ever sees the sampled token ID. The token embeddings are only used later in the blending step (Eq. 2: h̄_i = (1-λ)·e_i + λ·Σ A_ij·e_j), which is separate from the similarity computation.