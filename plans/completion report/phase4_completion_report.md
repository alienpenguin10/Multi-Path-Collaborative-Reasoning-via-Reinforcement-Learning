# Phase 4: Learnable Gating Variants — Completion Report

## Objective

Implement gating functions with learnable parameters — Luong (bilinear) and Bahdanau (additive/MLP) attention — that receive gradients during GRPO loss computation via the Phase 3 differentiable infrastructure. These complete the full set of 7 gating variants (5 parameter-free from Phases 1-2 + 2 learnable from Phase 4) for the dissertation's experimental comparison.

## What Was Implemented

### 1. `LuongAttentionGating` — Bilinear Attention with Low-Rank Factorization

**Location:** `transformers/src/transformers/models/qwen2/m3po_gating/learnable_gates.py`, lines 27–105

**Formula:** `S_ij = p_i^T W p_j` where `W = U @ V^T`

**The problem with a full W:** For Qwen2.5 with vocab_size=151936, a full weight matrix `W` would be (151936, 151936) = ~23 billion parameters. This is larger than the model itself and completely intractable.

**Low-rank factorization solution:** Factor `W = U @ V^T` where:
- `U`: (vocab_size, rank) = (151936, 128)
- `V`: (vocab_size, rank) = (151936, 128)
- Total: `2 * 151936 * 128 = 38,895,616` parameters (~38.9M)

**Efficient computation** avoids materializing the full W matrix:
```
1. a = p @ U        → (N, rank)    [O(N * V * r)]
2. b = p @ V        → (N, rank)    [O(N * V * r)]
3. S = a @ b^T      → (N, N)       [O(N^2 * r)]
```
Total cost: `O(N * V * r)` instead of `O(N * V^2)` if W were materialized.

**Initialization:** Xavier-style with `std = sqrt(2 / (vocab_size + rank))`. This produces small initial similarity values (order 1e-5), which means the gating starts near-uniform — all paths attend equally to each other. As training progresses, the model learns to differentiate.

**Key property — asymmetry:** Since `U != V`, the similarity matrix is asymmetric: `S_ij != S_ji` in general. This captures directional influence — path i's relevance to path j need not equal path j's relevance to path i. The test suite verifies this property holds with random initialization (`max |S_ij - S_ji| ≈ 8.3e-5` in practice).

**Configurable parameters:**
| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Rank | `rank` | 128 | Low-rank factorization dimension |
| Vocab size | `vocab_size` | 151936 | Must match model vocabulary |
| Temperature | `temperature` | 0.1 | Softmax temperature for attention |

**Memory footprint:**
| Precision | Parameters | Memory |
|-----------|-----------|--------|
| float32 | 38,895,616 | 148.4 MB |
| bfloat16 | 38,895,616 | 74.2 MB |

---

### 2. `BahdanauAttentionGating` — Additive (MLP) Attention

**Location:** `transformers/src/transformers/models/qwen2/m3po_gating/learnable_gates.py`, lines 108–209

**Formula:** `S_ij = v^T tanh(W1 * p_i + W2 * p_j)`

This projects each distribution into a lower-dimensional attention space, combines them additively, applies a tanh nonlinearity, then scores with a learned vector. The additive form captures nonlinear relationships between distributions that the bilinear Luong form cannot.

**Parameters:**
- `W1`: (attn_dim, vocab_size) — projects the "query" distribution
- `W2`: (attn_dim, vocab_size) — projects the "key" distribution
- `v`: (attn_dim,) — scoring vector
- Total: `2 * 256 * 151936 + 256 = 77,791,488` parameters (~77.8M)

**Vectorized pairwise computation** avoids an explicit double loop over paths:
```python
h1 = W1 @ p^T                        # (attn_dim, N) — project all queries
h2 = W2 @ p^T                        # (attn_dim, N) — project all keys
combined = h1[:, :, None] + h2[:, None, :]  # (attn_dim, N, N) — broadcast pairwise
activated = tanh(combined)            # (attn_dim, N, N)
S = (v[:, None, None] * activated).sum(dim=0)  # (N, N) — score
```
This computes all `N^2` pairwise similarities in a single vectorized operation.

**Initialization:**
- `W1`, `W2`: Xavier with `std = sqrt(2 / (vocab_size + attn_dim))` — appropriate for layers feeding into tanh
- `v`: Scaled with `std = sqrt(1 / attn_dim)` — keeps initial similarities near 0 for stable training start

**Key property — nonlinearity:** The tanh activation means Bahdanau can learn similarity functions that are nonlinear in the input distributions. For example, it could learn that paths whose distributions peak on different-but-related tokens (e.g., synonyms) should have high similarity, which a linear function like dot product cannot express.

**Configurable parameters:**
| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Attention dim | `attn_dim` | 256 | Hidden dimension of MLP |
| Vocab size | `vocab_size` | 151936 | Must match model vocabulary |
| Temperature | `temperature` | 0.1 | Softmax temperature for attention |

**Memory footprint:**
| Precision | Parameters | Memory |
|-----------|-----------|--------|
| float32 | 77,791,488 | 296.8 MB |
| bfloat16 | 77,791,488 | 148.4 MB |

---

### 3. Registration and Package Integration

**`__init__.py` change** (line 12):
```python
from . import learnable_gates
```
This single import triggers the registration at module load time. The `learnable_gates.py` file ends with:
```python
register_gating_function("luong", LuongAttentionGating)
register_gating_function("bahdanau", BahdanauAttentionGating)
```

After registration, the full registry contains 7 gating functions:
```
['baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'bhattacharyya', 'luong', 'bahdanau']
```

No changes were needed to `factory.py` or `base.py` — the existing infrastructure from Phase 1 handled learnable gates cleanly.

---

### 4. Phase 1-2 and Phase 3 Test Compatibility Fixes

**Problem:** The existing `test_gating_infrastructure.py` (Phase 1-2) and `test_gradient_flow.py` (Phase 3) iterate over all registered gating functions using `GATING_REGISTRY.keys()`. With the new learnable gates registered, these tests now encounter `luong` and `bahdanau` — but they create gating functions with `config = {'temperature': 0.1}` without specifying `vocab_size`. The learnable gates default to `vocab_size=151936`, while the tests use dummy tensors with `vocab_size=1000`, causing a shape mismatch.

**Fix:** Added `vocab_size`, `rank`, and `attn_dim` to the test config in both files:

`test_gating_infrastructure.py` (line 28):
```python
config = {'temperature': 0.1, 'debug': False, 'vocab_size': vocab_size, 'rank': 32, 'attn_dim': 64}
```

`test_gradient_flow.py` (line 226):
```python
config = {"temperature": 0.1, "debug": False, "vocab_size": vocab_size, "rank": 32, "attn_dim": 64}
```

Parameter-free gates simply ignore the extra config keys (they don't read `vocab_size`, `rank`, or `attn_dim`), so this change is backward-compatible.

---

### 5. Test Suite: `test_learnable_gates.py`

**Location:** `test_learnable_gates.py` (new file, 460 lines)

Nine tests covering all aspects of the learnable gating implementations:

| # | Test | What It Validates | Key Assertions |
|---|------|-------------------|----------------|
| 1 | Factory creation | Correct config parsing, shapes, defaults | `rank=64 → U=(1000,64)`, `attn_dim=128 → W1=(128,1000)`, defaults match Qwen2.5 |
| 2 | Similarity computation | Output shape, no NaN/Inf, value ranges | Both produce (N, N) matrices with finite values |
| 3 | Attention weights | Sum-to-1, diagonal masking, inactive masking, stats | Active rows sum to 1.0, inactive row is all zeros |
| 4 | Gradient flow | All params receive non-zero gradients via `apply_m3po_to_logits` | `U.grad`, `V.grad` (Luong); `W1.grad`, `W2.grad`, `v.grad` (Bahdanau) all non-None, non-NaN, non-zero |
| 5 | Optimizer step | Parameters actually change after `optimizer.step()` | All params differ from initial values; second step also works |
| 6 | Numerical stability | Extreme distributions: peaked, uniform, large magnitude | No NaN/Inf in output or gradients for all 6 combinations |
| 7 | Full-scale params | Parameter counts at Qwen2.5 vocab size | Luong: exactly 38,895,616; Bahdanau: exactly 77,791,488 |
| 8 | Backward compat | All 5 Phase 1-2 gates still work | Each produces correct output and gradients through `apply_m3po_to_logits` |
| 9 | Luong asymmetry | Bilinear form is asymmetric | `max |S_ij - S_ji| > 1e-6` with random initialization |

---

## Files Changed

| File | Type | Lines Changed | Summary |
|------|------|---------------|---------|
| `transformers/src/transformers/models/qwen2/m3po_gating/learnable_gates.py` | New | 210 | Luong and Bahdanau implementations + factory registration |
| `transformers/src/transformers/models/qwen2/m3po_gating/__init__.py` | Modified | +1 | Added `from . import learnable_gates` |
| `test_learnable_gates.py` | New | 460 | 9 tests for learnable gates |
| `test_gating_infrastructure.py` | Modified | +1 | Added `vocab_size`/`rank`/`attn_dim` to config |
| `test_gradient_flow.py` | Modified | +1 | Added `vocab_size`/`rank`/`attn_dim` to config |
| `M3PO_GATING_PROGRESS.md` | Modified | ~30 | Phase 4 marked complete, table updated, progress to 79% |

---

## Design Decisions & Rationale

### 1. Low-rank factorization for Luong (rank=128)

A full bilinear weight matrix `W` of shape `(vocab_size, vocab_size)` is completely intractable at 23 billion parameters. The low-rank factorization `W = U @ V^T` reduces this to ~39M parameters while preserving the ability to learn pairwise interactions between vocabulary positions. Rank 128 was chosen as a balance:
- Too low (e.g., 8): insufficient expressiveness to capture meaningful cross-path dynamics
- Too high (e.g., 1024): approaches ~310M parameters with diminishing returns
- 128: ~39M parameters, comparable to a small transformer layer, provides a rich interaction space

The rank is configurable via `gating_config['rank']` for experimentation.

### 2. Attention dimension 256 for Bahdanau

The MLP attention dimension determines the capacity of the nonlinear similarity function. With `attn_dim=256`:
- W1, W2 each have `256 * 151936 ≈ 39M` parameters
- The total (~78M) is roughly 2x Luong, providing a fair comparison of the bilinear vs. MLP approaches
- 256 is large enough to capture complex distribution relationships without being wasteful

Configurable via `gating_config['attn_dim']`.

### 3. Xavier initialization

Both implementations use Xavier-style initialization scaled to their respective architectures:
- **Luong U, V:** `std = sqrt(2 / (vocab_size + rank))` — produces initial similarities near zero (order 1e-5), so attention starts near-uniform and gradually sharpens as the model learns
- **Bahdanau W1, W2:** `std = sqrt(2 / (vocab_size + attn_dim))` — standard for layers feeding into tanh
- **Bahdanau v:** `std = sqrt(1 / attn_dim)` — smaller to keep initial similarity scores near 0, preventing saturated attention at initialization

This was chosen over alternatives like:
- Zero init (all similarities equal → no learning signal to break symmetry)
- Kaiming init (designed for ReLU, not appropriate for tanh or bilinear)
- Large random init (immediate attention saturation, gradient instability)

### 4. Separate U and V (not tied)

The Luong factorization uses separate `U` and `V` matrices rather than tying them (`W = U @ U^T`). This deliberately produces asymmetric similarity: `S_ij != S_ji`. The rationale is that influence between paths is directional — if path i has high confidence on token A and path j has high confidence on token B, the relevance of j to i may differ from the relevance of i to j. Tying U=V would force symmetry and reduce expressiveness.

### 5. vocab_size as a config parameter (not inferred)

The vocab_size is passed explicitly in `gating_config` rather than inferred from the first `compute_similarity_matrix` call. This design:
- Allows parameter allocation at construction time (needed for `nn.Module` registration)
- Makes the parameter count deterministic and inspectable before any forward pass
- Matches the convention of other PyTorch modules that take dimension parameters at init
- Defaults to 151936 (Qwen2.5-1.5B-Instruct vocab) for convenience

---

## Integration with Phase 3 Infrastructure

The learnable gates work seamlessly with the Phase 3 differentiable path. No modifications to Phase 3 code were needed. The gradient flow chain:

```
loss
  → log_softmax(blended_logits)
    → apply_m3po_to_logits()
      → gating_function.compute_similarity_matrix()   ← gradients flow to U, V (Luong) or W1, W2, v (Bahdanau)
        → gating_function.compute_attention_weights()
          → logit blending
```

The `train_with_grpo()` function from Phase 3 automatically:
1. Detects `has_learnable_parameters == True`
2. Moves the gating function to the correct device (`.to(device)`)
3. Adds gating parameters to the optimizer's parameter list
4. Includes gating parameters in gradient clipping (`max_norm=0.1`)
5. Logs gating statistics to wandb under `m3po/*` keys
6. Resets statistics after each logging step

---

## Test Results

All 22 tests across 3 test suites pass:

```
=== Phase 1-2 (test_gating_infrastructure.py) ===
  ✓ PASS: baseline
  ✓ PASS: raw_dot
  ✓ PASS: scaled_dot
  ✓ PASS: kl_divergence
  ✓ PASS: bhattacharyya
  ✓ PASS: luong
  ✓ PASS: bahdanau
✓ All tests passed!

=== Phase 3 (test_gradient_flow.py) ===
  ✓ PASS: Gradient flow to logits
  ✓ PASS: Gradient flow to learnable params
  ✓ PASS: Numerical stability
  ✓ PASS: Backward compatibility
  ✓ PASS: All gating functions
  ✓ PASS: Completion mask
✓ All gradient flow tests passed!

=== Phase 4 (test_learnable_gates.py) ===
  ✓ PASS: Factory creation
  ✓ PASS: Similarity computation
  ✓ PASS: Attention weights
  ✓ PASS: Gradient flow
  ✓ PASS: Optimizer step
  ✓ PASS: Numerical stability
  ✓ PASS: Full-scale params
  ✓ PASS: Backward compat
  ✓ PASS: Luong asymmetry
✓ All learnable gate tests passed!
```

---

## Complete Gating Function Registry

With Phase 4 complete, the full set of 7 gating variants is available:

| Name | Type | Formula | Parameters | Characteristics |
|------|------|---------|-----------|-----------------|
| `baseline` | Parameter-free | Cosine similarity | 0 | Original M3PO paper, normalized, symmetric |
| `raw_dot` | Parameter-free | `p_i · p_j` | 0 | Confidence-weighted, no normalization |
| `scaled_dot` | Parameter-free | `(p_i · p_j) / sqrt(d)` | 0 | Transformer-style, numerically stable |
| `kl_divergence` | Parameter-free | `1 - sqrt(JSD(p_i \|\| p_j))` | 0 | Distribution-theoretic, most principled |
| `bhattacharyya` | Parameter-free | `sum(sqrt(p_i * p_j))` | 0 | Probability overlap measure |
| `luong` | Learnable | `p_i^T (U @ V^T) p_j` | ~38.9M | Low-rank bilinear, asymmetric |
| `bahdanau` | Learnable | `v^T tanh(W1*p_i + W2*p_j)` | ~77.8M | MLP-style, nonlinear, additive |

---

## Usage

### Training with Luong gating:
```python
training_config = {
    'use_m3po': True,
    'gating_type': 'luong',
    'gating_config': {
        'temperature': 0.1,
        'rank': 128,           # Low-rank dimension (default: 128)
        'vocab_size': 151936,  # Qwen2.5 vocab (default)
    },
}
```

### Training with Bahdanau gating:
```python
training_config = {
    'use_m3po': True,
    'gating_type': 'bahdanau',
    'gating_config': {
        'temperature': 0.1,
        'attn_dim': 256,       # MLP hidden dimension (default: 256)
        'vocab_size': 151936,  # Qwen2.5 vocab (default)
    },
}
```

### Running tests:
```bash
source /homes/vk545/Neuralese/miniconda3/bin/activate ant

python test_gating_infrastructure.py  # Phase 1-2: all 7 gates
python test_gradient_flow.py          # Phase 3: gradient flow through all 7 gates
python test_learnable_gates.py        # Phase 4: learnable gate-specific tests
```

---

## Remaining Verification

The following require GPU access and are deferred to Phase 5 experiment runs:

- [ ] Full training run with `gating_type='luong'` — verify loss decreases, gating params update, no OOM on A100
- [ ] Full training run with `gating_type='bahdanau'` — same verification
- [ ] Memory profiling at full scale (vocab=151936) on actual hardware
- [ ] Wandb dashboard showing `m3po/*` statistics diverging from initial values over training
