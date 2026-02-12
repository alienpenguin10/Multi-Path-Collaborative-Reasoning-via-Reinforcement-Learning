# M3PO Alternative Gating Functions - Implementation Progress

## Overview

This document tracks the implementation of alternative gating functions for M3PO (Multi-Path Collaborative Reasoning) as part of a dissertation research project investigating how different cross-path similarity measures affect multi-path collaborative reasoning.

**Research Question**: How does the choice of cross-path similarity function affect the quality, stability, and convergence speed of multi-path collaborative reasoning?

## Implementation Status

### ✅ Phase 1: Infrastructure (COMPLETED)

**Goal**: Create a modular, extensible framework for alternative gating functions.

- [x] **Task 1.1**: Base architecture (`base.py`)
  - Abstract base class `BaseM3POGating` with shared functionality
  - Attention weight computation with temperature scaling
  - Statistics tracking (entropy, max weight, variance, similarity stats)
  - Debug logging utilities

- [x] **Task 1.2**: Factory pattern (`factory.py`)
  - Registry-based factory for creating gating instances
  - `create_gating_function(gating_type, config)` API
  - Dynamic registration system

- [x] **Task 1.3**: Modified M3PO utilities (`m3po_utils.py`)
  - Added `gating_function` parameter to all M3PO functions
  - Falls back to baseline cosine similarity when `gating_function=None`
  - Full backward compatibility maintained

- [x] **Task 1.4**: Modified training script (`grpo_train.py`)
  - Added `gating_type` and `gating_config` to training config
  - Gating function creation and device placement
  - Propagated through generation pipeline

- [x] **Task 1.5**: Testing infrastructure
  - Comprehensive test suite (`test_gating_infrastructure.py`)
  - Validates similarity computation, attention weights, masking, statistics
  - All tests passing ✓

### ✅ Phase 2: Parameter-Free Variants (COMPLETED)

**Goal**: Implement and test gating functions without learnable parameters.

- [x] **Task 2.1**: Raw Dot-Product (`RawDotProductGating`)
  - Formula: `S_ij = p_i · p_j`
  - Confidence-weighted (no normalization)
  - Range: [0, 1]

- [x] **Task 2.2**: Scaled Dot-Product (`ScaledDotProductGating`)
  - Formula: `S_ij = (p_i · p_j) / sqrt(d)`
  - Transformer-style attention
  - Better numerical stability for large vocabularies

- [x] **Task 2.3**: KL/JSD Divergence (`KLDivergenceGating`)
  - Formula: `S_ij = 1 - sqrt(JSD(p_i || p_j))`
  - Most principled distribution-theoretic measure
  - Symmetric and bounded

- [x] **Bonus**: Bhattacharyya Coefficient (`BhattacharyyaGating`)
  - Formula: `S_ij = sum(sqrt(p_i * p_j))`
  - Efficient probability overlap measure
  - More interpretable than JSD

**Status**: All parameter-free variants implemented and tested ✓

### 🔄 Phase 3: Learnable Gating Infrastructure (NEXT)

**Goal**: Enable gradient flow for learnable gating parameters.

- [ ] **Task 3.1**: Modify GRPO loss to apply M3PO during forward pass
  - Apply gating to logits before computing log probabilities
  - Enables gradient flow to learnable parameters

- [ ] **Task 3.2**: Create `apply_m3po_to_logits()` function
  - Applies M3PO gating to output logits
  - Handles batching per question group

- [ ] **Task 3.3**: Test gradient flow
  - Unit test for gradient computation
  - Verify no NaN/Inf in gradients

### ⏳ Phase 4: Learnable Variants (PENDING)

**Goal**: Implement gating functions with learnable parameters.

- [ ] **Task 4.1**: Luong Attention (`LuongAttentionGating`)
  - Formula: `S_ij = p_i^T W p_j`
  - Low-rank factorization: `W = U @ V^T`
  - Rank: 128 (38M parameters instead of 23B)

- [ ] **Task 4.2**: Bahdanau Attention (`BahdanauAttentionGating`)
  - Formula: `S_ij = v^T tanh(W1·p_i + W2·p_j)`
  - Attention dimension: 256

- [ ] **Task 4.3**: Test learnable variants
  - Small-scale training (10 steps)
  - Verify parameter updates and gradient flow

### ⏳ Phase 5: Experiments & Evaluation (PENDING)

**Goal**: Run comprehensive experiments and analyze results.

- [ ] **Task 5.1**: Create experiment runner (`run_m3po_experiment.py`)
  - Command-line interface for running experiments
  - Automates training + evaluation for each variant

- [ ] **Task 5.2**: Run full experiments
  - 8 variants × 3 trials = 24 experiments
  - Track metrics: accuracy, convergence, stability, attention stats

- [ ] **Task 5.3**: Create comparison script (`analyze_gating_results.py`)
  - Generate comparison plots
  - Statistical significance testing

## Available Gating Functions

| Name | Type | Formula | Characteristics |
|------|------|---------|-----------------|
| `baseline` | Parameter-free | Cosine similarity (original M3PO) | Normalized, symmetric |
| `raw_dot` | Parameter-free | `p_i · p_j` | Confidence-weighted |
| `scaled_dot` | Parameter-free | `(p_i · p_j) / sqrt(d)` | Transformer-style, stable |
| `kl_divergence` | Parameter-free | `1 - sqrt(JSD(p_i ‖ p_j))` | Distribution-theoretic |
| `bhattacharyya` | Parameter-free | `sum(sqrt(p_i * p_j))` | Probability overlap |
| `luong` | Learnable | `p_i^T W p_j` | Low-rank projection (not yet implemented) |
| `bahdanau` | Learnable | `v^T tanh(W1·p_i + W2·p_j)` | MLP-style (not yet implemented) |

## Usage

### Basic Usage in Training

```python
# In grpo_train.py, modify training_config:

training_config = {
    # ... other parameters ...
    'use_m3po': True,
    'gating_type': 'raw_dot',  # Choose from: baseline, raw_dot, scaled_dot, kl_divergence, bhattacharyya
    'gating_config': {
        'temperature': 0.1,
        'debug': False,  # Set to True for detailed logging
    },
}

model = train_with_grpo(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_function=combined_reward,
    device_ids=device_ids,
    **training_config
)
```

### Testing Gating Functions

```bash
# Activate conda environment
source /homes/vk545/Neuralese/miniconda3/bin/activate ant

# Run infrastructure tests
python test_gating_infrastructure.py

# Expected output: ✓ All tests passed!
```

### Programmatic Usage

```python
from transformers.models.qwen2.m3po_gating import create_gating_function, list_gating_functions

# List available gating functions
print(list_gating_functions())
# Output: ['baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'bhattacharyya']

# Create a gating function
config = {'temperature': 0.1, 'debug': True}
gating_fn = create_gating_function('raw_dot', config)

# Use in M3PO generation
from transformers.models.qwen2.m3po_utils import generate_with_m3po

outputs = generate_with_m3po(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    num_generations=4,
    lambda_blend=0.1,
    temperature_m3po=0.1,
    gating_function=gating_fn,  # Pass the gating function
)
```

## File Structure

```
transformers/src/transformers/models/qwen2/
├── m3po_utils.py                      # Modified for gating support
├── m3po_gating/                       # New package
│   ├── __init__.py                    # Package exports
│   ├── base.py                        # Abstract base class
│   ├── parameter_free_gates.py        # Parameter-free implementations
│   ├── learnable_gates.py             # Learnable implementations (pending)
│   └── factory.py                     # Factory pattern

grpo_train.py                          # Modified for gating support
test_gating_infrastructure.py          # Test suite
```

## Architecture Details

### Base Class Design

The `BaseM3POGating` abstract class provides:

1. **Similarity Computation**: Abstract method `compute_similarity_matrix()` that each variant implements
2. **Attention Weights**: Shared method `compute_attention_weights()` that applies temperature-scaled softmax
3. **Masking**: Handles diagonal masking (no self-attention) and inactive path masking
4. **Statistics**: Tracks attention entropy, max weights, variance, and similarity statistics
5. **Debug Logging**: Optional detailed logging of attention computations

### Key Design Decisions

1. **Backward Compatibility**: When `gating_function=None`, falls back to original cosine similarity
2. **Modular Design**: Each gating function is self-contained and independently testable
3. **Statistics Tracking**: All variants track the same metrics for fair comparison
4. **Device Handling**: Gating functions inherit device placement from model
5. **Gradient Flow**: (Phase 3) Learnable parameters will receive gradients during loss computation

## Test Results

All parameter-free gating functions pass the following tests:

✓ Similarity matrix computation (correct shape, no NaN/Inf)
✓ Attention weight computation (correct shape, sum to 1)
✓ Diagonal masking (no self-attention)
✓ Inactive path masking (zero attention for finished paths)
✓ Statistics tracking (entropy, max, variance, similarity)

Sample output from `raw_dot` gating:
```
Similarity matrix (first 3x3):
  [[0.0025, 0.0010, 0.0010],
   [0.0010, 0.0027, 0.0011],
   [0.0010, 0.0011, 0.0024]]

Attention weights (first 3x3):
  [[0.0000, 0.4998, 0.5002],
   [0.4998, 0.0000, 0.5002],
   [0.5000, 0.5000, 0.0000]]
```

## Next Steps

1. **Phase 3**: Implement learnable gating infrastructure
   - Modify GRPO loss to apply M3PO during forward pass
   - Test gradient flow for learnable parameters

2. **Phase 4**: Implement Luong and Bahdanau attention
   - Low-rank Luong attention (rank=128)
   - Bahdanau MLP-style attention (dim=256)

3. **Phase 5**: Run experiments and analysis
   - 3 trials × 8 variants = 24 experiments
   - Statistical analysis and visualization
   - Dissertation writeup

## Timeline Estimate

- ✅ Phase 1 (Infrastructure): 2 days → **DONE**
- ✅ Phase 2 (Parameter-Free): 5 days → **DONE** (completed early!)
- ⏳ Phase 3 (Learnable Infrastructure): 3 days → **IN PROGRESS**
- ⏳ Phase 4 (Learnable Variants): 5 days
- ⏳ Phase 5 (Experiments): 4 days + 3-4 days compute

**Total Progress**: ~35% complete (7/19 days of development)

## Known Issues & Limitations

1. **JSD Computational Cost**: O(N² × vocab_size) - slowest variant (~2-3x slower than baseline)
2. **Temperature Sensitivity**: Different similarity ranges may need different temperatures
3. **Learnable Parameters**: Not yet implemented (Phase 4)
4. **Evaluation Metrics**: Need to add M3PO-specific evaluator (Phase 1, Task 1.5)

## References

- M3PO Paper: "Multi-Path Collaborative Reasoning via Reinforcement Learning"
- Transformers: Vaswani et al., "Attention is All You Need"
- Bhattacharyya Coefficient: Classical information theory measure
- Jensen-Shannon Divergence: Lin, "Divergence measures based on the Shannon entropy"
