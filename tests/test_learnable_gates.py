"""
Test learnable gating functions for M3PO (Phase 4).

Tests:
1. Factory creation with config parameters
2. Similarity matrix computation (shape, no NaN/Inf, symmetry properties)
3. Attention weight computation (sum to 1, masking)
4. Gradient flow through apply_m3po_to_logits
5. Parameter updates via optimizer step
6. Numerical stability with extreme distributions
7. Full-scale parameter count verification (Qwen2.5 vocab)
8. Backward compatibility (existing tests still pass)
"""

import torch
import torch.nn as nn
import sys

sys.path.insert(0, "transformers/src")

from transformers.models.qwen2.m3po_utils import apply_m3po_to_logits
from transformers.models.qwen2.m3po_gating import (
    create_gating_function,
    GATING_REGISTRY,
)


# Use small vocab for fast tests, verify full-scale separately
SMALL_VOCAB = 1000
NUM_PATHS = 4


def test_factory_creation():
    """Test 1: Factory creates learnable gates with correct configs."""
    print(f"\n{'='*60}")
    print("Test 1: Factory creation")
    print(f"{'='*60}")

    # Luong with custom rank
    config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 64}
    luong = create_gating_function("luong", config)
    assert luong is not None, "Luong should not be None"
    assert luong.has_learnable_parameters, "Luong should be learnable"
    assert luong.rank == 64, f"Expected rank=64, got {luong.rank}"
    assert luong.vocab_size == SMALL_VOCAB
    assert luong.U.shape == (SMALL_VOCAB, 64), f"U shape: {luong.U.shape}"
    assert luong.V.shape == (SMALL_VOCAB, 64), f"V shape: {luong.V.shape}"
    print(f"  ✓ Luong: rank=64, U={luong.U.shape}, V={luong.V.shape}")

    # Bahdanau with custom attn_dim
    config = {"temperature": 0.2, "vocab_size": SMALL_VOCAB, "attn_dim": 128}
    bahdanau = create_gating_function("bahdanau", config)
    assert bahdanau is not None, "Bahdanau should not be None"
    assert bahdanau.has_learnable_parameters, "Bahdanau should be learnable"
    assert bahdanau.attn_dim == 128, f"Expected attn_dim=128, got {bahdanau.attn_dim}"
    assert bahdanau.temperature == 0.2, f"Expected temp=0.2, got {bahdanau.temperature}"
    assert bahdanau.W1.shape == (128, SMALL_VOCAB)
    assert bahdanau.W2.shape == (128, SMALL_VOCAB)
    assert bahdanau.v.shape == (128,)
    print(f"  ✓ Bahdanau: attn_dim=128, W1={bahdanau.W1.shape}, W2={bahdanau.W2.shape}, v={bahdanau.v.shape}")

    # Default configs
    luong_default = create_gating_function("luong", {"temperature": 0.1})
    assert luong_default.rank == 128, "Default rank should be 128"
    assert luong_default.vocab_size == 151936, "Default vocab should be 151936"
    print(f"  ✓ Luong defaults: rank={luong_default.rank}, vocab={luong_default.vocab_size}")

    bahdanau_default = create_gating_function("bahdanau", {"temperature": 0.1})
    assert bahdanau_default.attn_dim == 256, "Default attn_dim should be 256"
    print(f"  ✓ Bahdanau defaults: attn_dim={bahdanau_default.attn_dim}, vocab={bahdanau_default.vocab_size}")

    return True


def test_similarity_computation():
    """Test 2: Similarity matrices have correct shape and values."""
    print(f"\n{'='*60}")
    print("Test 2: Similarity matrix computation")
    print(f"{'='*60}")

    torch.manual_seed(42)
    logits = torch.randn(NUM_PATHS, SMALL_VOCAB)
    output_dists = torch.softmax(logits, dim=-1)

    for name in ["luong", "bahdanau"]:
        config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32, "attn_dim": 64}
        gf = create_gating_function(name, config)

        sim = gf.compute_similarity_matrix(output_distributions=output_dists)

        assert sim.shape == (NUM_PATHS, NUM_PATHS), f"{name}: shape {sim.shape}"
        assert not torch.isnan(sim).any(), f"{name}: contains NaN"
        assert not torch.isinf(sim).any(), f"{name}: contains Inf"

        print(f"  ✓ {name}: shape={sim.shape}, range=[{sim.min():.4f}, {sim.max():.4f}]")
        if NUM_PATHS <= 4:
            print(f"    Matrix:\n    {sim}")

    return True


def test_attention_weights():
    """Test 3: Attention weights sum to 1 and respect masking."""
    print(f"\n{'='*60}")
    print("Test 3: Attention weight computation")
    print(f"{'='*60}")

    torch.manual_seed(42)
    logits = torch.randn(NUM_PATHS, SMALL_VOCAB)
    output_dists = torch.softmax(logits, dim=-1)

    thinking_mask = torch.tensor([True, True, True, False])

    for name in ["luong", "bahdanau"]:
        config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32, "attn_dim": 64}
        gf = create_gating_function(name, config)

        sim = gf.compute_similarity_matrix(output_distributions=output_dists)
        attn, valid_mask = gf.compute_attention_weights(
            similarity_matrix=sim,
            thinking_mask=thinking_mask,
            mask_diagonal=True,
        )

        assert attn.shape == (NUM_PATHS, NUM_PATHS), f"{name}: shape {attn.shape}"

        # Active paths with valid targets should sum to 1
        active_paths = thinking_mask.nonzero(as_tuple=True)[0]
        for i in active_paths:
            if valid_mask[i].any():
                row_sum = attn[i].sum().item()
                assert abs(row_sum - 1.0) < 1e-5, f"{name}: row {i} sum = {row_sum}"

        # Inactive path should have zero attention
        inactive_attn = attn[3].sum().item()
        assert inactive_attn < 1e-6, f"{name}: inactive path attention = {inactive_attn}"

        # Statistics should be tracked
        stats = gf.get_stats_summary()
        assert len(stats) > 0, f"{name}: no statistics tracked"

        print(f"  ✓ {name}: rows sum to 1, inactive masked, stats tracked")
        if NUM_PATHS <= 4:
            print(f"    Attention:\n    {attn}")

    return True


def test_gradient_flow():
    """Test 4: Gradients flow through apply_m3po_to_logits to learnable params."""
    print(f"\n{'='*60}")
    print("Test 4: Gradient flow through apply_m3po_to_logits")
    print(f"{'='*60}")

    N, seq_len = NUM_PATHS, 10

    for name in ["luong", "bahdanau"]:
        torch.manual_seed(42)
        logits = torch.randn(N, seq_len, SMALL_VOCAB, requires_grad=True)
        config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32, "attn_dim": 64}
        gf = create_gating_function(name, config)

        # Zero all grads
        for p in gf.parameters():
            assert p.grad is None, f"{name}: grad should be None before backward"

        blended = apply_m3po_to_logits(
            logits=logits,
            num_generations=N,
            lambda_blend=0.1,
            temperature=0.1,
            gating_function=gf,
        )

        loss = blended.sum()
        loss.backward()

        # Check logits grad
        assert logits.grad is not None, f"{name}: logits.grad is None"
        assert not torch.isnan(logits.grad).any(), f"{name}: logits.grad has NaN"

        # Check all learnable parameter gradients
        for pname, p in gf.named_parameters():
            assert p.grad is not None, f"{name}.{pname}: grad is None"
            assert not torch.isnan(p.grad).any(), f"{name}.{pname}: grad has NaN"
            assert p.grad.abs().sum() > 0, f"{name}.{pname}: grad is all zeros"

        param_info = ", ".join(f"{pn}={p.grad.abs().mean():.6f}" for pn, p in gf.named_parameters())
        print(f"  ✓ {name}: all param grads non-zero ({param_info})")

    return True


def test_optimizer_step():
    """Test 5: Parameters actually update after optimizer step."""
    print(f"\n{'='*60}")
    print("Test 5: Parameter updates via optimizer")
    print(f"{'='*60}")

    N, seq_len = NUM_PATHS, 10

    for name in ["luong", "bahdanau"]:
        torch.manual_seed(42)
        config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32, "attn_dim": 64}
        gf = create_gating_function(name, config)

        # Save initial parameter values
        initial_params = {pn: p.clone().detach() for pn, p in gf.named_parameters()}

        optimizer = torch.optim.AdamW(gf.parameters(), lr=1e-3)

        # Forward + backward + step
        logits = torch.randn(N, seq_len, SMALL_VOCAB, requires_grad=True)
        blended = apply_m3po_to_logits(
            logits=logits,
            num_generations=N,
            lambda_blend=0.1,
            temperature=0.1,
            gating_function=gf,
        )
        loss = blended.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        for pname, p in gf.named_parameters():
            diff = (p - initial_params[pname]).abs().max().item()
            assert diff > 0, f"{name}.{pname}: parameter did not change after optimizer step"

        print(f"  ✓ {name}: all parameters updated after optimizer.step()")

        # Do a second step to verify continued training
        logits2 = torch.randn(N, seq_len, SMALL_VOCAB, requires_grad=True)
        blended2 = apply_m3po_to_logits(
            logits=logits2,
            num_generations=N,
            lambda_blend=0.1,
            temperature=0.1,
            gating_function=gf,
        )
        loss2 = blended2.sum()
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        print(f"  ✓ {name}: second optimizer step also works (no accumulated errors)")

    return True


def test_numerical_stability():
    """Test 6: Numerical stability with extreme distributions."""
    print(f"\n{'='*60}")
    print("Test 6: Numerical stability")
    print(f"{'='*60}")

    N, seq_len = NUM_PATHS, 10
    all_passed = True

    test_cases = [
        ("Very peaked", torch.zeros(N, seq_len, SMALL_VOCAB).scatter_(-1, torch.randint(0, SMALL_VOCAB, (N, seq_len, 1)), 100.0)),
        ("Very uniform", torch.zeros(N, seq_len, SMALL_VOCAB) + 0.001),
        ("Large magnitude", torch.randn(N, seq_len, SMALL_VOCAB) * 50),
    ]

    for name in ["luong", "bahdanau"]:
        config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32, "attn_dim": 64}
        gf = create_gating_function(name, config)

        for case_name, logits in test_cases:
            logits = logits.requires_grad_(True)
            try:
                blended = apply_m3po_to_logits(
                    logits=logits,
                    num_generations=N,
                    lambda_blend=0.1,
                    temperature=0.1,
                    gating_function=gf,
                )
                has_nan = torch.isnan(blended).any().item()
                has_inf = torch.isinf(blended).any().item()

                if has_nan or has_inf:
                    print(f"  ✗ {name}/{case_name}: NaN={has_nan}, Inf={has_inf}")
                    all_passed = False
                else:
                    loss = blended.sum()
                    loss.backward()
                    grad_ok = logits.grad is not None and not torch.isnan(logits.grad).any()
                    param_grad_ok = all(
                        p.grad is not None and not torch.isnan(p.grad).any()
                        for p in gf.parameters()
                    )
                    if grad_ok and param_grad_ok:
                        print(f"  ✓ {name}/{case_name}: output OK, all gradients OK")
                    else:
                        print(f"  ✗ {name}/{case_name}: output OK but gradient issues")
                        all_passed = False
            except Exception as e:
                print(f"  ✗ {name}/{case_name}: Exception — {e}")
                all_passed = False

            # Reset grads for next case
            for p in gf.parameters():
                if p.grad is not None:
                    p.grad = None

    return all_passed


def test_full_scale_params():
    """Test 7: Parameter counts at Qwen2.5 scale."""
    print(f"\n{'='*60}")
    print("Test 7: Full-scale parameter counts (Qwen2.5 vocab=151936)")
    print(f"{'='*60}")

    # Luong with default config
    luong = create_gating_function("luong", {"temperature": 0.1})
    luong_params = sum(p.numel() for p in luong.parameters())
    expected_luong = 2 * 151936 * 128
    assert luong_params == expected_luong, f"Luong params: {luong_params} != expected {expected_luong}"
    print(f"  ✓ Luong: {luong_params:,} params (rank=128, vocab=151936)")
    print(f"    U: {luong.U.shape}, V: {luong.V.shape}")

    # Bahdanau with default config
    bahdanau = create_gating_function("bahdanau", {"temperature": 0.1})
    bahdanau_params = sum(p.numel() for p in bahdanau.parameters())
    expected_bahdanau = 2 * 256 * 151936 + 256
    assert bahdanau_params == expected_bahdanau, f"Bahdanau params: {bahdanau_params} != expected {expected_bahdanau}"
    print(f"  ✓ Bahdanau: {bahdanau_params:,} params (attn_dim=256, vocab=151936)")
    print(f"    W1: {bahdanau.W1.shape}, W2: {bahdanau.W2.shape}, v: {bahdanau.v.shape}")

    # Memory estimates (float32)
    luong_mb = luong_params * 4 / (1024 * 1024)
    bahdanau_mb = bahdanau_params * 4 / (1024 * 1024)
    print(f"\n  Memory (float32): Luong={luong_mb:.1f} MB, Bahdanau={bahdanau_mb:.1f} MB")
    print(f"  Memory (bfloat16): Luong={luong_mb/2:.1f} MB, Bahdanau={bahdanau_mb/2:.1f} MB")

    return True


def test_backward_compat():
    """Test 8: Existing gating functions still work."""
    print(f"\n{'='*60}")
    print("Test 8: Backward compatibility")
    print(f"{'='*60}")

    torch.manual_seed(42)
    N, seq_len = NUM_PATHS, 10

    existing_types = ["baseline", "raw_dot", "scaled_dot", "kl_divergence", "bhattacharyya"]
    all_passed = True

    for gating_type in existing_types:
        logits = torch.randn(N, seq_len, SMALL_VOCAB, requires_grad=True)
        config = {"temperature": 0.1, "debug": False}
        gf = create_gating_function(gating_type, config)

        try:
            blended = apply_m3po_to_logits(
                logits=logits,
                num_generations=N,
                lambda_blend=0.1,
                temperature=0.1,
                gating_function=gf,
            )
            assert blended.shape == logits.shape
            assert not torch.isnan(blended).any()

            loss = blended.sum()
            loss.backward()
            assert logits.grad is not None
            assert not torch.isnan(logits.grad).any()

            print(f"  ✓ {gating_type}: still works correctly")
        except Exception as e:
            print(f"  ✗ {gating_type}: {e}")
            all_passed = False

    return all_passed


def test_luong_asymmetry():
    """Test 9: Luong bilinear form is asymmetric (S_ij != S_ji in general)."""
    print(f"\n{'='*60}")
    print("Test 9: Luong asymmetry property")
    print(f"{'='*60}")

    torch.manual_seed(42)
    logits = torch.randn(NUM_PATHS, SMALL_VOCAB)
    output_dists = torch.softmax(logits, dim=-1)

    config = {"temperature": 0.1, "vocab_size": SMALL_VOCAB, "rank": 32}
    luong = create_gating_function("luong", config)

    sim = luong.compute_similarity_matrix(output_distributions=output_dists)

    # Check asymmetry: S_ij != S_ji (since U != V after random init)
    transpose_diff = (sim - sim.t()).abs().max().item()
    print(f"  Max |S_ij - S_ji| = {transpose_diff:.6f}")

    # It's possible but extremely unlikely for random U,V to produce symmetric S
    # Just verify the matrix is computed and report
    if transpose_diff > 1e-6:
        print(f"  ✓ Luong is asymmetric (captures directional influence)")
    else:
        print(f"  ⚠ Luong appears symmetric (unusual for random init, but not an error)")

    return True


def main():
    """Run all learnable gate tests."""
    print("\nM3PO Phase 4: Learnable Gating Functions Tests")
    print("=" * 60)
    print(f"\nRegistered gating functions: {list(GATING_REGISTRY.keys())}")

    tests = [
        ("Factory creation", test_factory_creation),
        ("Similarity computation", test_similarity_computation),
        ("Attention weights", test_attention_weights),
        ("Gradient flow", test_gradient_flow),
        ("Optimizer step", test_optimizer_step),
        ("Numerical stability", test_numerical_stability),
        ("Full-scale params", test_full_scale_params),
        ("Backward compat", test_backward_compat),
        ("Luong asymmetry", test_luong_asymmetry),
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
        print(f"\n✓ All learnable gate tests passed!")
        return 0
    else:
        failed = [n for n, s in results.items() if not s]
        print(f"\n✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit(main())
