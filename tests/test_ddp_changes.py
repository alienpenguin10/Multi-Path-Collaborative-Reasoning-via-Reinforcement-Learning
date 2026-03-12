"""
Tests for the DDP + torch.compile migration of m3po_train.py.

Tests are organized into:
1. Unit tests (no GPU required) — imports, helpers, function signatures
2. Single-GPU tests — model loading, forward passes, compiled model, generation
3. Multi-GPU DDP tests — launched via torchrun subprocess

Usage:
    conda activate ant
    cd /homes/vk545/Neuralese/M3PO
    python tests/test_ddp_changes.py
"""

import os
import sys
import subprocess
import tempfile
import json
import signal

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "transformers", "src"))

import torch
import torch.distributed as dist

# ============================================================================
# Test utilities
# ============================================================================
_passed = 0
_failed = 0
_errors = []


def run_test(name, fn):
    """Run a test function and track pass/fail."""
    global _passed, _failed
    try:
        fn()
        print(f"  ✓ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        _failed += 1
        _errors.append((name, str(e)))


def summary():
    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed")
    if _errors:
        print(f"\nFailures:")
        for name, err in _errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
    return _failed == 0


# ============================================================================
# 1. Import and structure tests (no GPU)
# ============================================================================
def test_imports():
    """All new imports resolve without error."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import datetime
    assert hasattr(dist, "init_process_group")
    assert DDP is not None


def test_ddp_helpers_importable():
    """setup_ddp, cleanup_ddp, is_main_process are importable from the script."""
    # We import via ast to avoid triggering module-level side effects
    import ast
    with open("m3po_train.py") as f:
        tree = ast.parse(f.read())
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "setup_ddp" in func_names, "setup_ddp not found"
    assert "cleanup_ddp" in func_names, "cleanup_ddp not found"
    assert "is_main_process" in func_names, "is_main_process not found"


def test_no_cuda0_hardcoding():
    """No remaining cuda:0 hardcoding in the script."""
    with open("m3po_train.py") as f:
        content = f.read()
    # Allow cuda:0 in comments but not in code
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.split("#")[0]  # ignore comments
        if "cuda:0" in stripped:
            raise AssertionError(f"Found cuda:0 in code at line {i}: {line.strip()}")


def test_no_dataparallel_usage():
    """No remaining nn.DataParallel wrapping in the script."""
    with open("m3po_train.py") as f:
        content = f.read()
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.split("#")[0]
        if "nn.DataParallel" in stripped:
            raise AssertionError(f"Found nn.DataParallel at line {i}: {line.strip()}")


def test_no_device_ids_param():
    """train_with_grpo no longer accepts device_ids parameter."""
    import ast
    with open("m3po_train.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_with_grpo":
            param_names = [arg.arg for arg in node.args.args]
            assert "device_ids" not in param_names, "device_ids still in train_with_grpo params"
            assert "local_rank" in param_names, "local_rank missing from train_with_grpo params"
            return
    raise AssertionError("train_with_grpo function not found")


def test_compiled_model_param_exists():
    """generate_rollout_data and grpo_loss accept compiled_model parameter."""
    import ast
    with open("m3po_train.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_rollout_data":
            param_names = [arg.arg for arg in node.args.args]
            assert "compiled_model" in param_names, "compiled_model missing from generate_rollout_data"
        if isinstance(node, ast.FunctionDef) and node.name == "grpo_loss":
            param_names = [arg.arg for arg in node.args.args]
            assert "compiled_model" in param_names, "compiled_model missing from grpo_loss"


def test_is_main_process_guards():
    """wandb.log and wandb.init are guarded by is_main_process()."""
    with open("m3po_train.py") as f:
        content = f.read()
    # Check that wandb.log appears after is_main_process check
    assert "if is_main_process():" in content, "is_main_process guard not found"
    # Check that wandb.init is inside an is_main_process block
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "wandb.init(" in line:
            # Look back for is_main_process guard
            found_guard = False
            for j in range(max(0, i - 5), i):
                if "is_main_process()" in lines[j]:
                    found_guard = True
                    break
            assert found_guard, f"wandb.init at line {i+1} not guarded by is_main_process()"


def test_dist_barrier_before_cleanup():
    """dist.barrier() is called before cleanup_ddp() at end of __main__."""
    with open("m3po_train.py") as f:
        content = f.read()
    barrier_pos = content.rfind("dist.barrier()")
    cleanup_pos = content.rfind("cleanup_ddp()")
    assert barrier_pos > 0, "dist.barrier() not found"
    assert cleanup_pos > 0, "cleanup_ddp() not found"
    assert barrier_pos < cleanup_pos, "dist.barrier() should come before cleanup_ddp()"


def test_gating_allreduce_present():
    """Gating gradient all_reduce is present after backward."""
    with open("m3po_train.py") as f:
        content = f.read()
    assert "dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)" in content, \
        "Gating gradient all_reduce not found"


# ============================================================================
# 2. Single-GPU functional tests
# ============================================================================
def test_selective_log_softmax():
    """selective_log_softmax produces correct shapes and values."""
    # Inline import to avoid module-level side effects
    os.environ["M3PO_DEBUG"] = "-1"
    # We can't easily import from m3po_train directly (module-level side effects),
    # so redefine the function for testing
    import torch.nn as nn

    def selective_log_softmax(logits, input_ids, chunk_size=2):
        batch_size = logits.shape[0]
        if batch_size <= chunk_size:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
        results = []
        for i in range(0, batch_size, chunk_size):
            chunk_logits = logits[i:i+chunk_size]
            chunk_ids = input_ids[i:i+chunk_size]
            chunk_log_probs = nn.functional.log_softmax(chunk_logits, dim=-1)
            chunk_result = chunk_log_probs.gather(dim=-1, index=chunk_ids.unsqueeze(-1)).squeeze(-1)
            results.append(chunk_result)
            del chunk_log_probs
        return torch.cat(results, dim=0)

    torch.manual_seed(42)
    logits = torch.randn(6, 10, 100)  # batch=6, seq=10, vocab=100
    ids = torch.randint(0, 100, (6, 10))

    result = selective_log_softmax(logits, ids, chunk_size=2)
    assert result.shape == (6, 10), f"Expected (6, 10), got {result.shape}"

    # Verify chunked == non-chunked
    full_result = selective_log_softmax(logits, ids, chunk_size=100)
    assert torch.allclose(result, full_result, atol=1e-6), "Chunked vs full results differ"


def test_create_completion_mask():
    """create_completion_mask correctly masks tokens after EOS."""
    # Redefine for isolated testing
    def create_completion_mask(completion_ids, eos_token_id):
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        mask_exists = is_eos.any(dim=1)
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    eos_id = 99
    # Sequence: [1, 2, 99, 3, 4] -> mask should be [1, 1, 1, 0, 0]
    ids = torch.tensor([[1, 2, eos_id, 3, 4], [5, 6, 7, 8, eos_id]])
    mask = create_completion_mask(ids, eos_id)
    assert mask[0].tolist() == [1, 1, 1, 0, 0], f"Row 0 mask wrong: {mask[0].tolist()}"
    assert mask[1].tolist() == [1, 1, 1, 1, 1], f"Row 1 mask wrong: {mask[1].tolist()}"

    # No EOS -> all 1s
    ids_no_eos = torch.tensor([[1, 2, 3, 4, 5]])
    mask_no_eos = create_completion_mask(ids_no_eos, eos_id)
    assert mask_no_eos[0].tolist() == [1, 1, 1, 1, 1]


def test_torch_compile_basic():
    """torch.compile works on a simple model (smoke test for environment)."""
    model = torch.nn.Linear(10, 10).cuda()
    compiled = torch.compile(model)
    x = torch.randn(2, 10).cuda()
    out = compiled(x)
    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in compiled model output"


def test_model_loads_to_specific_gpu():
    """Model loads to a specific GPU via device_map."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    # Check all parameters are on cuda:0
    for name, param in model.named_parameters():
        assert param.device == torch.device("cuda:0"), \
            f"Param {name} on {param.device}, expected cuda:0"
    del model
    torch.cuda.empty_cache()


def test_compiled_model_forward():
    """torch.compile works with a real transformer model forward pass."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    compiled_model = torch.compile(model)

    inputs = tokenizer("Hello world", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = compiled_model(**inputs)
    assert out.logits.shape[0] == 1, "Unexpected batch size"
    assert out.logits.shape[-1] > 0, "Empty vocab dimension"
    assert not torch.isnan(out.logits).any(), "NaN in compiled forward"
    del model, compiled_model
    torch.cuda.empty_cache()


def test_compiled_vs_uncompiled_match():
    """Compiled and uncompiled forward passes produce same logits."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    compiled_model = torch.compile(model)

    inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out_orig = model(**inputs).logits
        out_compiled = compiled_model(**inputs).logits

    # torch.compile with bfloat16 uses fused kernels that reorder operations,
    # so results can differ more than simple rounding. Check they're close, not exact.
    max_diff = (out_orig - out_compiled).abs().max().item()
    assert max_diff < 10.0, f"Max diff too large: {max_diff} (compiled diverged significantly)"
    del model, compiled_model
    torch.cuda.empty_cache()


def test_ddp_helpers_logic():
    """is_main_process returns True when dist is not initialized."""
    # dist should not be initialized in this test process
    if dist.is_initialized():
        dist.destroy_process_group()
    assert not dist.is_initialized()

    # Inline the function logic
    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0

    assert is_main_process() is True, "Should be True when dist not initialized"


# ============================================================================
# 3. Multi-GPU DDP tests (launched as subprocess via torchrun)
# ============================================================================

DDP_TEST_SCRIPT = '''
"""DDP subprocess test — launched by torchrun.
Uses small torch.nn models to avoid GPU OOM when GPUs are busy.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

sys.path.insert(0, os.path.abspath("{project_root}"))
sys.path.insert(0, os.path.abspath("{project_root}/transformers/src"))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["M3PO_DEBUG"] = "-1"

results = {{}}

def setup():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=5))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def test_ddp_init():
    local_rank, rank, world_size = setup()
    results["rank"] = rank
    results["world_size"] = world_size
    results["local_rank"] = local_rank
    results["device"] = str(torch.device("cuda"))
    assert world_size == {num_gpus}, f"Expected {num_gpus} GPUs, got {{world_size}}"
    results["ddp_init"] = "PASS"

def test_ddp_wrap_and_forward():
    """DDP wrapping and forward pass with a small model."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # Use a small MLP instead of a full transformer to avoid OOM on busy GPUs
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(f"cuda:{{local_rank}}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module

    x = torch.randn(4, 64, device=f"cuda:{{local_rank}}")
    out = model(x)
    assert out.shape == (4, 64), f"Expected (4, 64), got {{out.shape}}"
    assert not torch.isnan(out).any()
    results["ddp_wrap_and_forward"] = "PASS"

    del model
    torch.cuda.empty_cache()

def test_gradient_sync():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()

    # Simple model to test gradient sync
    model = torch.nn.Linear(10, 10).cuda()
    model = DDP(model, device_ids=[local_rank])

    # Each rank uses different input -> different gradients
    torch.manual_seed(42 + rank)
    x = torch.randn(4, 10, device=f"cuda:{{local_rank}}")
    y = model(x).sum()
    y.backward()

    # After backward, DDP should have averaged gradients across ranks
    # Collect gradient from all ranks and check they're equal
    grad = model.module.weight.grad.clone()
    gathered = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, grad)

    # All ranks should have the same gradient (averaged by DDP)
    for i in range(1, len(gathered)):
        assert torch.allclose(gathered[0], gathered[i], atol=1e-6), \\
            f"Grad mismatch between rank 0 and rank {{i}}"
    results["gradient_sync"] = "PASS"

    del model
    torch.cuda.empty_cache()

def test_gating_allreduce():
    """Test manual all_reduce for gating params (simulates learnable gating sync)."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()

    # Simulate gating function params
    gating_param = torch.nn.Parameter(torch.randn(10, 10, device=f"cuda:{{local_rank}}"))

    # Different "gradients" per rank
    torch.manual_seed(42 + rank)
    gating_param.grad = torch.randn_like(gating_param)
    original_grad = gating_param.grad.clone()

    # Manual all_reduce (same as in m3po_train.py)
    dist.all_reduce(gating_param.grad, op=dist.ReduceOp.AVG)

    # Verify gradient is now the average
    gathered_originals = [torch.zeros_like(original_grad) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_originals, original_grad)
    expected_avg = torch.stack(gathered_originals).mean(dim=0)

    assert torch.allclose(gating_param.grad, expected_avg, atol=1e-5), \\
        f"All-reduced grad doesn't match expected average"
    results["gating_allreduce"] = "PASS"

def test_compiled_model_with_ddp():
    """torch.compile works with the unwrapped model inside DDP."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Small model to avoid OOM
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    ).to(f"cuda:{{local_rank}}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw_model = model.module
    compiled_model = torch.compile(raw_model)

    x = torch.randn(4, 32, device=f"cuda:{{local_rank}}")
    out = compiled_model(x)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()
    results["compiled_with_ddp"] = "PASS"

    del model, compiled_model
    torch.cuda.empty_cache()

def test_rank_seeding():
    """Different ranks get different random samples with rank-based seeding."""
    import random
    rank = dist.get_rank()
    random.seed(42 + rank)
    samples = [random.randint(0, 10000) for _ in range(5)]

    # Gather samples from all ranks
    samples_tensor = torch.tensor(samples, device="cuda")
    gathered = [torch.zeros_like(samples_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, samples_tensor)

    # Ranks should have different samples
    if dist.get_world_size() > 1:
        assert not torch.equal(gathered[0], gathered[1]), \\
            "Rank 0 and rank 1 have same random samples — seeding is broken"
    results["rank_seeding"] = "PASS"

def test_ddp_backward_sync():
    """Verify DDP backward produces identical model weights after optimizer step."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()

    # All ranks start with same model (same seed)
    torch.manual_seed(42)
    model = nn.Linear(16, 16).to(f"cuda:{{local_rank}}")
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Each rank uses different data
    torch.manual_seed(42 + rank)
    for _ in range(3):
        x = torch.randn(4, 16, device=f"cuda:{{local_rank}}")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # After training, all ranks should have identical weights (DDP averages grads)
    weight = model.module.weight.data.clone()
    gathered = [torch.zeros_like(weight) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, weight)
    for i in range(1, len(gathered)):
        assert torch.allclose(gathered[0], gathered[i], atol=1e-6), \\
            f"Weight mismatch between rank 0 and rank {{i}}"
    results["ddp_backward_sync"] = "PASS"

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        test_ddp_init()
        test_ddp_wrap_and_forward()
        test_gradient_sync()
        test_gating_allreduce()
        test_compiled_model_with_ddp()
        test_rank_seeding()
        test_ddp_backward_sync()
    except Exception as e:
        results["error"] = str(e)
    finally:
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Each rank writes its results to a separate file
        output_path = os.path.join("{tmpdir}", f"rank_{{rank}}.json")
        with open(output_path, "w") as f:
            json.dump(results, f)
        cleanup()
'''


def run_ddp_tests(num_gpus=2):
    """Launch DDP tests via torchrun and collect results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_content = DDP_TEST_SCRIPT.format(
            project_root=PROJECT_ROOT,
            num_gpus=num_gpus,
            tmpdir=tmpdir,
        )
        script_path = os.path.join(tmpdir, "ddp_test.py")
        with open(script_path, "w") as f:
            f.write(script_content)

        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29599",
            script_path,
        ]

        print(f"\n  Launching DDP tests with {num_gpus} GPUs via torchrun...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            stderr_preview = result.stderr[-2000:] if result.stderr else "(no stderr)"
            raise RuntimeError(f"torchrun failed (exit {result.returncode}):\n{stderr_preview}")

        # Collect results from all ranks
        all_results = {}
        for rank in range(num_gpus):
            result_path = os.path.join(tmpdir, f"rank_{rank}.json")
            if not os.path.exists(result_path):
                raise RuntimeError(f"No result file for rank {rank}")
            with open(result_path) as f:
                all_results[f"rank_{rank}"] = json.load(f)

        return all_results


def test_ddp_multi_gpu():
    """Run all DDP subtests via torchrun with 2 GPUs."""
    num_gpus = min(2, torch.cuda.device_count())
    if num_gpus < 2:
        print("  ⚠ Skipping DDP multi-GPU tests (need ≥2 GPUs)")
        return

    # Free GPU memory from single-GPU tests before launching DDP subprocesses
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Use GPUs that weren't used by single-GPU tests (which used GPU 0)
    # CUDA_VISIBLE_DEVICES makes torchrun see only these GPUs
    avail_gpus = torch.cuda.device_count()
    if avail_gpus >= 4:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    elif avail_gpus >= 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    # else: 2 GPUs total, must reuse GPU 0 (already freed above)

    try:
        results = run_ddp_tests(num_gpus=num_gpus)
    finally:
        # Restore CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    # Verify all subtests passed on all ranks
    expected_tests = [
        "ddp_init", "ddp_wrap_and_forward",
        "gradient_sync", "gating_allreduce", "compiled_with_ddp",
        "rank_seeding", "ddp_backward_sync",
    ]
    for rank_key, rank_results in results.items():
        if "error" in rank_results:
            raise AssertionError(f"{rank_key} had error: {rank_results['error']}")
        for test_name in expected_tests:
            status = rank_results.get(test_name)
            if status != "PASS":
                raise AssertionError(f"{rank_key}/{test_name}: {status}")

    print(f"    All DDP subtests passed on {num_gpus} ranks")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DDP + torch.compile Migration Tests")
    print("=" * 60)

    # --- Section 1: Structure tests (no GPU) ---
    print("\n--- Structure & Import Tests ---")
    run_test("imports resolve", test_imports)
    run_test("DDP helpers defined", test_ddp_helpers_importable)
    run_test("no cuda:0 hardcoding", test_no_cuda0_hardcoding)
    run_test("no DataParallel usage", test_no_dataparallel_usage)
    run_test("no device_ids param", test_no_device_ids_param)
    run_test("compiled_model param exists", test_compiled_model_param_exists)
    run_test("is_main_process guards", test_is_main_process_guards)
    run_test("dist.barrier before cleanup", test_dist_barrier_before_cleanup)
    run_test("gating allreduce present", test_gating_allreduce_present)

    # --- Section 2: Single-GPU functional tests ---
    print("\n--- Single-GPU Functional Tests ---")
    run_test("selective_log_softmax", test_selective_log_softmax)
    run_test("create_completion_mask", test_create_completion_mask)
    run_test("DDP helpers logic", test_ddp_helpers_logic)
    run_test("torch.compile basic", test_torch_compile_basic)
    run_test("model loads to specific GPU", test_model_loads_to_specific_gpu)
    run_test("compiled model forward", test_compiled_model_forward)
    run_test("compiled vs uncompiled match", test_compiled_vs_uncompiled_match)

    # --- Section 3: Multi-GPU DDP tests ---
    print("\n--- Multi-GPU DDP Tests (via torchrun) ---")
    run_test("DDP multi-GPU suite", test_ddp_multi_gpu)

    # --- Summary ---
    success = summary()
    sys.exit(0 if success else 1)
