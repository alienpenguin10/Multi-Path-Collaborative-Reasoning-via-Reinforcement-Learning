"""
Test gradient flow through M3PO logit blending (Phase 3).

Tests:
1. Gradient flows back to input logits through apply_m3po_to_logits (baseline)
2. Gradient flows to learnable gating parameters
3. Numerical stability with extreme logit values
4. Backward compatibility: grpo_loss without M3PO matches original
5. All registered gating functions work with apply_m3po_to_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, "transformers/src")

from transformers.models.qwen2.m3po_utils import apply_m3po_to_logits
from transformers.models.qwen2.m3po_gating import (
    create_gating_function,
    GATING_REGISTRY,
)
from transformers.models.qwen2.m3po_gating.base import BaseM3POGating


class MockLearnableGating(BaseM3POGating):
    """Mock learnable gating function for testing gradient flow."""

    def __init__(self, vocab_size=1000, config=None):
        super().__init__(config)
        # Learnable weight matrix (small, for testing)
        self.W = nn.Parameter(torch.eye(vocab_size) * 0.01)

    @property
    def has_learnable_parameters(self):
        return True

    def compute_similarity_matrix(self, output_distributions, hidden_states=None):
        # S_ij = p_i^T W p_j (bilinear form)
        Wp = torch.mm(output_distributions, self.W)  # (N, vocab_size)
        similarity = torch.mm(Wp, output_distributions.t())  # (N, N)
        return similarity


def test_gradient_flow_to_logits():
    """Test 1: Gradient flows back to input logits (baseline cosine)."""
    print(f"\n{'='*60}")
    print("Test 1: Gradient flow to input logits (baseline)")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000
    torch.manual_seed(42)

    logits = torch.randn(N, seq_len, vocab_size, requires_grad=True)

    blended = apply_m3po_to_logits(
        logits=logits,
        num_generations=N,
        lambda_blend=0.1,
        temperature=0.1,
        gating_function=None,
    )

    # Compute a scalar loss from blended logits
    loss = blended.sum()
    loss.backward()

    assert logits.grad is not None, "logits.grad is None — no gradient flow"
    assert not torch.isnan(logits.grad).any(), "logits.grad contains NaN"
    assert not torch.isinf(logits.grad).any(), "logits.grad contains Inf"
    assert logits.grad.abs().sum() > 0, "logits.grad is all zeros"

    print(f"  ✓ logits.grad shape: {logits.grad.shape}")
    print(f"  ✓ logits.grad mean: {logits.grad.mean().item():.6f}")
    print(f"  ✓ logits.grad abs max: {logits.grad.abs().max().item():.6f}")
    return True


def test_gradient_flow_to_learnable_params():
    """Test 2: Gradient flows to learnable gating parameters."""
    print(f"\n{'='*60}")
    print("Test 2: Gradient flow to learnable parameters")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000
    torch.manual_seed(42)

    logits = torch.randn(N, seq_len, vocab_size, requires_grad=True)
    gating = MockLearnableGating(vocab_size=vocab_size, config={"temperature": 0.1})

    assert gating.has_learnable_parameters, "Mock gating should be learnable"
    assert gating.W.grad is None, "W.grad should be None before backward"

    blended = apply_m3po_to_logits(
        logits=logits,
        num_generations=N,
        lambda_blend=0.1,
        temperature=0.1,
        gating_function=gating,
    )

    loss = blended.sum()
    loss.backward()

    assert gating.W.grad is not None, "W.grad is None — no gradient flow to learnable params"
    assert not torch.isnan(gating.W.grad).any(), "W.grad contains NaN"
    assert gating.W.grad.abs().sum() > 0, "W.grad is all zeros"

    print(f"  ✓ W.grad shape: {gating.W.grad.shape}")
    print(f"  ✓ W.grad abs mean: {gating.W.grad.abs().mean().item():.6f}")
    print(f"  ✓ W.grad abs max: {gating.W.grad.abs().max().item():.6f}")

    # Also check logits grad
    assert logits.grad is not None, "logits.grad is None"
    print(f"  ✓ logits.grad also flows correctly")
    return True


def test_numerical_stability():
    """Test 3: Numerical stability with extreme logit values."""
    print(f"\n{'='*60}")
    print("Test 3: Numerical stability with extreme logits")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000

    test_cases = [
        ("Very peaked (softmax → one-hot)", torch.zeros(N, seq_len, vocab_size).scatter_(-1, torch.randint(0, vocab_size, (N, seq_len, 1)), 100.0)),
        ("Very uniform (near-zero logits)", torch.zeros(N, seq_len, vocab_size) + 0.001),
        ("Large magnitude", torch.randn(N, seq_len, vocab_size) * 50),
        ("Mixed extreme", torch.cat([torch.randn(2, seq_len, vocab_size) * 50, torch.zeros(2, seq_len, vocab_size)], dim=0)),
    ]

    all_passed = True
    for name, logits in test_cases:
        logits = logits.requires_grad_(True)

        try:
            blended = apply_m3po_to_logits(
                logits=logits,
                num_generations=N,
                lambda_blend=0.1,
                temperature=0.1,
                gating_function=None,
            )

            has_nan = torch.isnan(blended).any().item()
            has_inf = torch.isinf(blended).any().item()

            if has_nan or has_inf:
                print(f"  ✗ {name}: NaN={has_nan}, Inf={has_inf}")
                all_passed = False
            else:
                # Also check backward
                loss = blended.sum()
                loss.backward()
                grad_ok = logits.grad is not None and not torch.isnan(logits.grad).any()
                if grad_ok:
                    print(f"  ✓ {name}: output OK, gradient OK")
                else:
                    print(f"  ✗ {name}: output OK but gradient has issues")
                    all_passed = False

        except Exception as e:
            print(f"  ✗ {name}: Exception — {e}")
            all_passed = False

    return all_passed


def test_backward_compatibility():
    """Test 4: grpo_loss without M3PO produces identical results."""
    print(f"\n{'='*60}")
    print("Test 4: Backward compatibility (lambda=0 → no change)")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000
    torch.manual_seed(42)

    logits = torch.randn(N, seq_len, vocab_size)

    # With lambda_blend=0, blended should equal original
    blended = apply_m3po_to_logits(
        logits=logits,
        num_generations=N,
        lambda_blend=0.0,
        temperature=0.1,
        gating_function=None,
    )

    max_diff = (blended - logits).abs().max().item()
    assert max_diff < 1e-6, f"lambda=0 should produce identical logits, max diff={max_diff}"
    print(f"  ✓ lambda_blend=0.0 → max diff = {max_diff:.2e} (effectively zero)")

    # With lambda_blend > 0, blended should differ
    torch.manual_seed(42)
    logits2 = torch.randn(N, seq_len, vocab_size)
    blended2 = apply_m3po_to_logits(
        logits=logits2,
        num_generations=N,
        lambda_blend=0.1,
        temperature=0.1,
        gating_function=None,
    )

    diff = (blended2 - logits2).abs().mean().item()
    assert diff > 1e-6, f"lambda=0.1 should produce different logits, mean diff={diff}"
    print(f"  ✓ lambda_blend=0.1 → mean diff = {diff:.6f} (logits changed)")
    return True


def test_all_gating_functions():
    """Test 5: All registered gating functions work with apply_m3po_to_logits."""
    print(f"\n{'='*60}")
    print("Test 5: All gating functions with apply_m3po_to_logits")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000
    all_passed = True

    for gating_type in GATING_REGISTRY.keys():
        torch.manual_seed(42)
        logits = torch.randn(N, seq_len, vocab_size, requires_grad=True)

        config = {"temperature": 0.1, "debug": False, "vocab_size": vocab_size, "rank": 32, "attn_dim": 64}
        gating_fn = create_gating_function(gating_type, config)

        try:
            blended = apply_m3po_to_logits(
                logits=logits,
                num_generations=N,
                lambda_blend=0.1,
                temperature=0.1,
                gating_function=gating_fn,
            )

            # Check output shape and values
            assert blended.shape == logits.shape, f"Shape mismatch: {blended.shape} vs {logits.shape}"
            assert not torch.isnan(blended).any(), "Output contains NaN"
            assert not torch.isinf(blended).any(), "Output contains Inf"

            # Check gradient flow
            loss = blended.sum()
            loss.backward()
            assert logits.grad is not None, "No gradient flow"
            assert not torch.isnan(logits.grad).any(), "Gradient contains NaN"

            print(f"  ✓ {gating_type}: shape OK, values OK, gradient OK")

        except Exception as e:
            print(f"  ✗ {gating_type}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_completion_mask():
    """Test 6: Completion mask is respected (masked positions unchanged)."""
    print(f"\n{'='*60}")
    print("Test 6: Completion mask handling")
    print(f"{'='*60}")

    N, seq_len, vocab_size = 4, 10, 1000
    torch.manual_seed(42)

    logits = torch.randn(N, seq_len, vocab_size)

    # Create mask: first 5 positions valid, last 5 invalid
    mask = torch.zeros(N, seq_len)
    mask[:, :5] = 1.0

    # Custom gating with mask
    config = {"temperature": 0.1}
    gating_fn = create_gating_function("raw_dot", config)

    blended = apply_m3po_to_logits(
        logits=logits,
        num_generations=N,
        lambda_blend=0.1,
        temperature=0.1,
        gating_function=gating_fn,
        completion_mask=mask,
    )

    # Masked positions (last 5) should be unchanged
    masked_diff = (blended[:, 5:, :] - logits[:, 5:, :]).abs().max().item()
    assert masked_diff < 1e-6, f"Masked positions should be unchanged, max diff={masked_diff}"
    print(f"  ✓ Masked positions unchanged (max diff = {masked_diff:.2e})")

    # Unmasked positions (first 5) should be changed
    unmasked_diff = (blended[:, :5, :] - logits[:, :5, :]).abs().mean().item()
    assert unmasked_diff > 1e-6, f"Unmasked positions should be changed, mean diff={unmasked_diff}"
    print(f"  ✓ Unmasked positions changed (mean diff = {unmasked_diff:.6f})")
    return True


def main():
    """Run all gradient flow tests."""
    print("\nM3PO Phase 3: Gradient Flow Tests")
    print("=" * 60)

    tests = [
        ("Gradient flow to logits", test_gradient_flow_to_logits),
        ("Gradient flow to learnable params", test_gradient_flow_to_learnable_params),
        ("Numerical stability", test_numerical_stability),
        ("Backward compatibility", test_backward_compatibility),
        ("All gating functions", test_all_gating_functions),
        ("Completion mask", test_completion_mask),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All gradient flow tests passed!")
        return 0
    else:
        failed = [n for n, s in results.items() if not s]
        print(f"\n✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit(main())
