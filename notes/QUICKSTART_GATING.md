# Quick Start: Using Alternative M3PO Gating Functions

## Testing the Infrastructure

```bash
# Activate conda environment
source /homes/vk545/Neuralese/miniconda3/bin/activate ant

# Test all gating functions
python test_gating_infrastructure.py

# Expected output: ✓ All tests passed!
```

## Training with Different Gating Functions

Edit `grpo_train.py` around lines 700-720:

```python
training_config = {
    'num_iterations': 1,
    'num_steps': 500,
    'batch_size': 5,
    'num_generations': 4,
    'max_completion_length': 512,
    'beta': 0.005,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1,
    'lambda_blend': 0.1,
    'temperature_m3po': 0.1,
    'use_m3po': True,

    # ⭐ CHANGE THIS LINE to experiment with different gating functions
    'gating_type': 'raw_dot',  # Options: 'baseline', 'raw_dot', 'scaled_dot', 'kl_divergence', 'bhattacharyya'

    'gating_config': {
        'temperature': 0.1,     # Attention temperature (use same as temperature_m3po)
        'debug': False,         # Set to True for detailed M3PO logging
    },
}
```

Then run training:

```bash
# Monitor GPU and run when available
python monitor_and_run.py

# Or run directly (requires 2+ GPUs)
python grpo_train.py
```

## Available Gating Functions

### Parameter-Free Variants (Currently Available)

1. **`baseline`** (Default - Original M3PO)
   - Cosine similarity: normalized dot-product
   - Use this as your baseline for comparison

2. **`raw_dot`**
   - Raw dot-product without normalization
   - Confidence-weighted: high-probability predictions have more influence
   - Hypothesis: May improve stability by focusing on confident paths

3. **`scaled_dot`**
   - Transformer-style scaled dot-product
   - Formula: `similarity / sqrt(vocab_size)`
   - Hypothesis: Better numerical stability than raw_dot

4. **`kl_divergence`**
   - Jensen-Shannon Divergence (symmetric KL)
   - Most principled distribution-theoretic measure
   - Hypothesis: Better for distribution-heavy reasoning
   - ⚠️ Slower: ~2-3x compute cost vs baseline

5. **`bhattacharyya`**
   - Bhattacharyya coefficient (probability overlap)
   - Formula: `sum(sqrt(p_i * p_j))`
   - More interpretable than JSD, faster to compute

### Learnable Variants (Coming in Phase 4)

6. **`luong`** (Not yet implemented)
   - Learnable weight matrix W
   - Formula: `p_i^T W p_j`

7. **`bahdanau`** (Not yet implemented)
   - MLP-style attention
   - Formula: `v^T tanh(W1·p_i + W2·p_j)`

## Debugging

Enable detailed M3PO logging:

```python
# Option 1: In config
'gating_config': {
    'debug': True,  # Enables detailed logging
}

# Option 2: Environment variable (affects all code)
import os
os.environ["M3PO_DEBUG"] = "1"
```

This will print:
- Similarity matrices (for batches ≤ 8 paths)
- Attention weights
- Number of active (thinking) paths
- Attention statistics (entropy, max weight, variance)

## Quick Experiments

### Experiment 1: Baseline vs Raw Dot-Product

```python
# Run 1: Baseline
'gating_type': 'baseline',

# Run 2: Raw Dot-Product
'gating_type': 'raw_dot',
```

Compare final accuracy, convergence speed, and attention statistics.

### Experiment 2: Distribution-Based Methods

```python
# Run 1: Cosine Similarity (baseline)
'gating_type': 'baseline',

# Run 2: JSD
'gating_type': 'kl_divergence',

# Run 3: Bhattacharyya
'gating_type': 'bhattacharyya',
```

Hypothesis: Distribution-based methods may perform better on math reasoning tasks.

### Experiment 3: Scaling Effects

```python
# Run 1: Raw dot-product
'gating_type': 'raw_dot',

# Run 2: Scaled dot-product
'gating_type': 'scaled_dot',
```

Test if scaling improves stability (especially with large vocab_size=151,936).

## Expected Behavior

After implementing a gating function, you should see:

```
[M3PO] Training with cross-path interaction: lambda=0.1, temp=0.1
[M3PO] Using raw_dot gating function with config: {'temperature': 0.1, 'debug': False}
Model wrapped with DataParallel across GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
```

During training, M3PO will use your selected gating function for computing cross-path attention.

## Verifying It Works

1. **Check logs**: Look for the gating function initialization message
2. **Enable debug**: Set `debug=True` to see attention computations
3. **Monitor metrics**: Track if attention patterns differ between variants
4. **Test inference**: Generated text should use the selected gating function

## Common Issues

### Issue 1: "Unknown gating type"

```
ValueError: Unknown gating type: 'my_gate'. Available types: baseline, raw_dot, ...
```

**Solution**: Check spelling, use one of the registered types from `list_gating_functions()`

### Issue 2: NaN in attention weights

```
AssertionError: Attention matrix contains NaN
```

**Solution**: This should be fixed in base.py, but if it occurs:
- Check input distributions are valid (sum to 1, no NaN)
- Verify thinking_mask is correct
- Enable debug logging to see where NaN occurs

### Issue 3: Slow training with kl_divergence

KL divergence is O(N² × vocab_size), so it's ~2-3x slower than other variants.

**Solution**:
- Use for final experiments, not initial debugging
- Or reduce `num_generations` temporarily (e.g., 2 instead of 4)

## Next Steps

1. Run `test_gating_infrastructure.py` to verify everything works
2. Run a short training experiment (e.g., `num_steps=10`) with `baseline`
3. Run same experiment with `raw_dot` to compare
4. Enable `debug=True` to understand attention patterns
5. Scale up to full training once you verify it works

## Questions?

See `M3PO_GATING_PROGRESS.md` for detailed documentation.
