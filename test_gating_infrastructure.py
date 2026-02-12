"""
Simple test script to verify M3PO gating infrastructure.

Tests:
1. Factory can create all registered gating functions
2. Each gating function produces valid similarity matrices
3. Attention computation works correctly
4. Statistics tracking works
"""

import torch
import sys
sys.path.insert(0, 'transformers/src')

from transformers.models.qwen2.m3po_gating import (
    create_gating_function,
    GATING_REGISTRY,
)


def test_gating_function(gating_type, vocab_size=1000, num_paths=4):
    """Test a single gating function."""
    print(f"\n{'='*60}")
    print(f"Testing: {gating_type}")
    print(f"{'='*60}")

    # Create gating function
    config = {'temperature': 0.1, 'debug': False}
    gating_fn = create_gating_function(gating_type, config)

    if gating_fn is None:
        print(f"  ✓ Baseline returns None (uses original cosine similarity)")
        return True

    # Create dummy output distributions
    torch.manual_seed(42)
    logits = torch.randn(num_paths, vocab_size)
    output_distributions = torch.softmax(logits, dim=-1)

    # Compute similarity
    try:
        similarity_matrix = gating_fn.compute_similarity_matrix(
            output_distributions=output_distributions,
            hidden_states=None,
        )
        print(f"  ✓ Similarity matrix computed: shape {similarity_matrix.shape}")

        # Check shape
        assert similarity_matrix.shape == (num_paths, num_paths), \
            f"Expected shape ({num_paths}, {num_paths}), got {similarity_matrix.shape}"

        # Check for NaNs or Infs
        assert not torch.isnan(similarity_matrix).any(), "Similarity matrix contains NaN"
        assert not torch.isinf(similarity_matrix).any(), "Similarity matrix contains Inf"
        print(f"  ✓ No NaNs or Infs detected")

        # Compute attention weights
        thinking_mask = torch.tensor([True, True, True, False])  # 3 active, 1 inactive
        attention_weights, valid_mask = gating_fn.compute_attention_weights(
            similarity_matrix=similarity_matrix,
            thinking_mask=thinking_mask,
            mask_diagonal=True,
        )
        print(f"  ✓ Attention weights computed: shape {attention_weights.shape}")

        # Check attention properties
        assert attention_weights.shape == (num_paths, num_paths), \
            f"Expected shape ({num_paths}, {num_paths}), got {attention_weights.shape}"

        # Check that rows sum to ~1 (for active paths with valid targets)
        active_paths = thinking_mask.nonzero(as_tuple=True)[0]
        for i in active_paths:
            has_targets = valid_mask[i].any()
            if has_targets:
                row_sum = attention_weights[i].sum().item()
                assert abs(row_sum - 1.0) < 1e-5, \
                    f"Row {i} sum is {row_sum}, expected 1.0"

        print(f"  ✓ Attention weights sum to 1 for active paths")

        # Check that inactive path (index 3) has zero attention
        inactive_attention = attention_weights[3].sum().item()
        assert inactive_attention < 1e-6, \
            f"Inactive path has non-zero attention: {inactive_attention}"
        print(f"  ✓ Inactive paths have zero attention")

        # Check statistics
        stats = gating_fn.get_stats_summary()
        print(f"  ✓ Statistics tracked: {list(stats.keys())}")

        # Print sample values
        print(f"\n  Sample similarity matrix (first 3x3):")
        print(f"  {similarity_matrix[:3, :3]}")
        print(f"\n  Sample attention weights (first 3x3):")
        print(f"  {attention_weights[:3, :3]}")

        if stats:
            print(f"\n  Statistics:")
            for key, value in list(stats.items())[:5]:  # Print first 5 stats
                print(f"    {key}: {value:.4f}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all registered gating functions."""
    print("\nM3PO Gating Infrastructure Test")
    print("="*60)

    print(f"\nRegistered gating functions: {list(GATING_REGISTRY.keys())}")

    results = {}
    for gating_type in GATING_REGISTRY.keys():
        success = test_gating_function(gating_type)
        results[gating_type] = success

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for gating_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {gating_type}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
