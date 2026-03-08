# Phase 5: Experiments & Evaluation — Completion Report

## Objective

Create the experiment automation and analysis infrastructure to systematically evaluate all 8 gating conditions (7 M3PO gating variants + no-M3PO control) across multiple trials. This is the final development phase — after this, the remaining work is running compute and writing up results.

## What Was Implemented

### 1. `grpo_train.py` — Configurable Output Directory

**Location:** `grpo_train.py`, lines 874–885

**Problem:** The main block hardcoded the model save path to `"grpo_finetuned_model"`, making it impossible for an external experiment runner to control where each experiment's outputs go.

**Changes:**

1. Added `import json` (line 8) for saving structured metadata.

2. Replaced the hardcoded save path (line 874):
```python
# Before:
model.save_pretrained("grpo_finetuned_model")
tokenizer.save_pretrained("grpo_finetuned_model")

# After:
save_dir = training_config.get('output_dir', 'grpo_finetuned_model')
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
```

3. Added structured metadata saving alongside the model (lines 880–885):
```python
# Save training config (filtered to JSON-serializable types)
serializable_config = {k: v for k, v in training_config.items()
                       if isinstance(v, (int, float, str, bool, dict, list))}
with open(os.path.join(save_dir, "training_config.json"), "w") as f:
    json.dump(serializable_config, f, indent=2)

# Save results
with open(os.path.join(save_dir, "results.json"), "w") as f:
    json.dump({"accuracy": post_grpo_accuracy}, f, indent=2)
```

**Backward compatibility:** When `'output_dir'` is not in the training config (i.e., when running `grpo_train.py` directly without modification), it defaults to `"grpo_finetuned_model"` — identical to the old behavior.

---

### 2. `run_m3po_experiment.py` — Experiment Runner CLI

**Location:** `run_m3po_experiment.py` (new file, 374 lines)

A command-line tool that automates training + evaluation for each gating variant. Designed for running 24 experiments (8 variants × 3 trials) with minimal manual intervention.

**8 Experiment Conditions:**

| # | `--gating_type` | `use_m3po` | Gating Function | Purpose |
|---|----------------|-----------|-----------------|---------|
| 1 | `none` | `False` | N/A | Control — no cross-path interaction |
| 2 | `baseline` | `True` | Cosine similarity | Original M3PO paper method |
| 3 | `raw_dot` | `True` | Raw dot product | Confidence-weighted |
| 4 | `scaled_dot` | `True` | Scaled dot product | Transformer-style |
| 5 | `kl_divergence` | `True` | Jensen-Shannon divergence | Distribution-theoretic |
| 6 | `bhattacharyya` | `True` | Bhattacharyya coefficient | Probability overlap |
| 7 | `luong` | `True` | Bilinear (low-rank) | Learnable, asymmetric |
| 8 | `bahdanau` | `True` | MLP (additive) | Learnable, nonlinear |

The `none` condition (`use_m3po=False`) was added as a control. It measures the effect of M3PO itself, separate from the choice of gating function. This is important — without it, you can only compare gating functions to each other but not demonstrate that M3PO improves over standard GRPO.

**CLI Arguments:**

```
Experiment selection:
  --gating_type {none,baseline,...,bahdanau}   Single variant to run
  --trial TRIAL                                 Trial number (default: 1)
  --run_all                                     Run all 8 variants × all trials
  --num_trials NUM_TRIALS                       Trials per variant (default: 3)

Training overrides:
  --num_steps NUM_STEPS                         Training steps (default: 500)
  --batch_size BATCH_SIZE                       Batch size (default: 5)
  --num_generations NUM_GENERATIONS             Paths per prompt (default: 4)
  --max_completion_length MAX_COMPLETION_LENGTH  Max tokens (default: 512)
  --eval_size EVAL_SIZE                         Eval examples (default: 30)
  --seed_base SEED_BASE                         Base seed (default: 42)

Modes:
  --eval_only                                   Skip training, evaluate existing model
  --force                                       Re-run even if results exist
```

**Usage examples:**
```bash
# Single experiment
python run_m3po_experiment.py --gating_type raw_dot --trial 1

# Full sweep: 8 × 3 = 24 experiments
python run_m3po_experiment.py --run_all --num_trials 3

# Quick smoke test (5 training steps)
python run_m3po_experiment.py --gating_type raw_dot --trial 1 --num_steps 5

# Re-evaluate saved model without retraining
python run_m3po_experiment.py --gating_type raw_dot --trial 1 --eval_only

# No-M3PO control
python run_m3po_experiment.py --gating_type none --trial 1
```

**Key architectural decisions:**

1. **Reuses `grpo_train.py` functions — no code duplication.** The experiment runner imports `train_with_grpo` and `optimize_model_memory` from `grpo_train.py`, and `prepare_dataset`, `evaluate_model`, `combined_reward` from `utils.py`. The only new logic is the CLI orchestration, config construction, and result saving.

2. **Resume-friendly.** Before starting an experiment, checks if `results.json` already exists in the output directory. If so, skips that experiment and loads the existing results. This means a multi-day `--run_all` sweep can be interrupted and restarted without re-running completed experiments. Use `--force` to override.

3. **Deterministic seeding.** Trial seed = `seed_base + trial - 1` (default: trial 1 → seed 42, trial 2 → seed 43, trial 3 → seed 44). The same seed is used for both the random state and the train/eval data split, so all gating types in the same trial see identical training data. This makes cross-variant comparison within a trial directly controlled.

4. **Output structure:**
```
outputs/
├── none/
│   ├── trial_1/   → model, tokenizer, training_config.json, results.json
│   ├── trial_2/
│   └── trial_3/
├── baseline/
│   ├── trial_1/
│   ...
├── raw_dot/
│   ...
└── bahdanau/
    └── trial_3/
```

5. **Wandb integration.** Each experiment creates a wandb run with a descriptive name (`"M3PO raw_dot trial 1"` or `"no-M3PO trial 1"`) and tags (`["raw_dot", "trial_1"]`) for easy filtering in the dashboard.

6. **Summary table in `--run_all` mode.** After all experiments complete, prints two tables:
   - Per-experiment: gating type, trial, accuracy, duration
   - Per-variant averages: mean accuracy, std, N trials — sorted by mean accuracy

**`run_single_experiment()` flow (lines 131–254):**
1. Check for existing results (skip if found)
2. Set random seed based on trial number
3. Prepare train/eval data split (deterministic)
4. Detect available GPUs
5. If `--eval_only`: load model from output dir; else: load base model, build training config, init wandb, call `train_with_grpo()`
6. Evaluate with `evaluate_model()`
7. Save model, `training_config.json`, and `results.json`
8. Finish wandb run

---

### 3. `analyze_gating_results.py` — Results Analysis & Plotting

**Location:** `analyze_gating_results.py` (new file, 437 lines)

Reads experiment results from the output directory structure and produces summary statistics, comparison plots, and pairwise statistical significance tests.

**Usage:**
```bash
# Analyze all results
python analyze_gating_results.py

# Analyze specific variants only
python analyze_gating_results.py --variants raw_dot kl_divergence baseline none

# Custom directories
python analyze_gating_results.py --results_dir outputs/ --plot_dir plots/

# Skip wandb (faster, doesn't need API access)
python analyze_gating_results.py --no_wandb
```

**Components:**

**3a. Data Loading — `load_all_results()`** (lines 27–69)

Walks the `outputs/{gating_type}/trial_{n}/results.json` directory tree and builds a pandas DataFrame. Handles missing files gracefully and auto-fills `gating_type` and `trial` from directory names if not present in the JSON.

**3b. Wandb History Loading — `load_wandb_history()`** (lines 72–99)

Optional. Uses `wandb.Api()` to fetch per-step training curves (loss, average_reward, attention entropy) for convergence analysis. Falls back gracefully if wandb is unavailable.

**3c. Summary Table — `print_summary_table()`** (lines 105–131)

Aggregates results by gating type:
- Mean accuracy ± std
- N trials, min, max
- Mean training duration (if available)
- Sorted by mean accuracy (best first)

Example output (from mock data verification):
```
======================================================================
RESULTS SUMMARY
======================================================================
               Mean Acc %   Std  N Trials   Min   Max
gating_type
kl_divergence       67.23  0.93         3  66.2  68.0
raw_dot             65.10  0.90         3  64.2  66.0
baseline            65.00  1.70         3  63.3  66.7
none                61.93  1.72         3  60.0  63.3
======================================================================
```

Also saved as `plots/results_summary.csv`.

**3d. Statistical Significance — `compute_pairwise_significance()`** (lines 137–202)

For every pair of gating types (with ≥2 trials each):
- **Welch's t-test** (unequal variance): computes p-values for pairwise accuracy differences
- **Cohen's d effect size**: standardized measure of how large the difference is
- Reports all pairs with p < 0.05 as "statistically significant"

Gracefully skips if scipy is not installed (prints install instructions).

**3e. Plots** (4 types, all saved as 150 DPI PNGs):

| Plot | Function | Description |
|------|----------|-------------|
| `accuracy_comparison.png` | `plot_accuracy_comparison()` | Bar chart of mean accuracy per variant with std error bars and value labels |
| `accuracy_boxplot.png` | `plot_accuracy_boxplot()` | Box plot showing accuracy distribution per variant, sorted by median, color-coded |
| `significance_heatmap.png` | `plot_significance_heatmap()` | Matrix of pairwise p-values on -log10 scale, annotated with actual values |
| `convergence_curves.png` | `plot_convergence_curves()` | Loss and reward vs. training step from wandb history |

All plots use `matplotlib.use("Agg")` for headless rendering (no display needed).

**Error handling:**
- Missing scipy: skips significance testing with install instructions
- No wandb: skips convergence plots
- Single trial per variant: skips box plot (meaningless with 1 point)
- Empty results dir: exits with informative message

---

### 4. `M3PO_GATING_PROGRESS.md` — Updated

- Phase 5 marked as complete with all 4 tasks checked off
- Progress updated from ~79% to 100%
- Timeline updated: Phase 5 marked DONE
- "Next Steps" section updated to reflect that development is finished — remaining work is running compute and writing up results
- File structure updated with `run_m3po_experiment.py` and `analyze_gating_results.py`

---

## Files Changed

| File | Type | Lines | Summary |
|------|------|-------|---------|
| `grpo_train.py` | Modified | +13 | `import json`; configurable `output_dir`; save `training_config.json` and `results.json` |
| `run_m3po_experiment.py` | New | 374 | CLI experiment runner: 8 conditions, resume-friendly, wandb integration |
| `analyze_gating_results.py` | New | 437 | Results analysis: summary table, 4 plot types, Welch's t-test, Cohen's d |
| `M3PO_GATING_PROGRESS.md` | Modified | ~25 | Phase 5 complete, progress to 100% |

---

## Design Decisions & Rationale

### 1. Eight conditions, not seven

The experiment set includes `none` (use_m3po=False) as a control alongside the 7 gating variants. This is critical for the dissertation — without a no-M3PO baseline, you can only compare gating functions to each other but cannot demonstrate that M3PO itself provides value. The `none` condition isolates the effect of cross-path interaction from the effect of the gating function choice.

### 2. No code duplication from grpo_train.py

The experiment runner imports and calls the same `train_with_grpo()`, `optimize_model_memory()`, `prepare_dataset()`, `evaluate_model()`, and `combined_reward()` functions. The only new code is the CLI parsing, config construction, and result-saving orchestration. This means:
- Bug fixes in `grpo_train.py` automatically apply to experiments
- Training behavior is guaranteed identical to manual runs
- No risk of divergent implementations

### 3. Resume-friendly execution

Long experiment sweeps (24 experiments at ~2-4 hours each = 48-96 hours of compute) will likely be interrupted. The runner checks for `results.json` before each experiment and skips completed ones. This means you can:
- `Ctrl+C` and restart later
- Add `--force` to re-run a specific experiment
- Run experiments in any order or on different machines

### 4. Fixed seed per trial, same across variants

`seed = 42 + trial - 1` ensures:
- **Within a trial**: All 8 gating types see identical training data in identical order. This is the strongest form of controlled comparison — any accuracy difference is attributable to the gating function, not the data.
- **Across trials**: Different random states provide variance estimates for statistical testing.

### 5. Welch's t-test (not paired)

Though the same training data is used within a trial, the training dynamics are highly nonlinear and the same data ordering doesn't guarantee paired outcomes. Welch's t-test (unequal variance assumption) is more conservative and appropriate here than a paired t-test.

### 6. Graceful degradation for missing dependencies

- `scipy` not installed → skip significance testing, still produce summary and plots
- `wandb` API inaccessible → skip convergence curves, still produce accuracy plots
- No results found → exit with clear message pointing to `run_m3po_experiment.py`

---

## Verification

### Import tests
```
✓ run_m3po_experiment.py imports correctly
  - ALL_GATING_TYPES: ['none', 'baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'bhattacharyya', 'luong', 'bahdanau']
  - get_output_dir('raw_dot', 2) → 'outputs/raw_dot/trial_2'

✓ analyze_gating_results.py imports correctly
  - load_all_results, print_summary_table, compute_pairwise_significance all importable
```

### CLI help
```
✓ run_m3po_experiment.py --help prints full usage with all 12 arguments and examples
```

### Mock data analysis
Created mock results for 4 variants × 3 trials and ran the analyzer:
```
✓ Loaded 12 experiment results across 4 gating types
✓ Summary table: correct sorting (kl_divergence > raw_dot > baseline > none), correct mean/std
✓ results_summary.csv saved correctly
✓ accuracy_comparison.png generated (bar chart with error bars)
✓ accuracy_boxplot.png generated (color-coded box plot)
✓ scipy gracefully skipped (not installed in test env)
```

### Existing test suites
All 22 tests across 3 suites pass after the `grpo_train.py` modification:
```
✓ test_gating_infrastructure.py — 7 tests (all 7 gating types)
✓ test_gradient_flow.py — 6 tests (gradient flow, stability, all gates)
✓ test_learnable_gates.py — 9 tests (creation, gradients, optimizer, compat)
```

---

## Running the Experiments

With all infrastructure in place, the full experiment sweep can be launched with:

```bash
# Full sweep: 8 variants × 3 trials = 24 experiments
python run_m3po_experiment.py --run_all --num_trials 3

# Or use monitor_and_run.py for auto-launch when GPUs are available
# (would need modification to call run_m3po_experiment.py instead of grpo_train.py)
```

After experiments complete:

```bash
# Generate all analysis outputs
python analyze_gating_results.py

# Outputs in plots/:
#   accuracy_comparison.png   — bar chart
#   accuracy_boxplot.png      — distribution plot
#   significance_heatmap.png  — p-value matrix
#   convergence_curves.png    — loss/reward over time
#   results_summary.csv       — summary table
```

---

## Project Status: Development Complete

With Phase 5 finished, all 5 development phases are complete:

| Phase | Status | Deliverables |
|-------|--------|-------------|
| Phase 1: Infrastructure | Complete | `base.py`, `factory.py`, modified `m3po_utils.py`, `grpo_train.py` |
| Phase 2: Parameter-Free | Complete | 4 gating functions: `raw_dot`, `scaled_dot`, `kl_divergence`, `bhattacharyya` |
| Phase 3: Gradient Flow | Complete | `apply_m3po_to_logits()`, `compute_log_probs_with_m3po()`, modified GRPO loss |
| Phase 4: Learnable Gates | Complete | `luong` (38.9M params), `bahdanau` (77.8M params) |
| Phase 5: Experiments | Complete | `run_m3po_experiment.py`, `analyze_gating_results.py` |

**Remaining work** is compute and writing:
1. Run 24 experiments (~48-96 hours on 8×A100)
2. Analyze results with `analyze_gating_results.py`
3. Write up findings for dissertation
