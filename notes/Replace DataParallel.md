# Plan: Replace DataParallel with DDP + torch.compile

## Context

`grpo_train_single.py` uses `nn.DataParallel` for multi-GPU training. DataParallel is slow because:
- Single-process with GIL contention
- Gather/scatter all outputs to GPU:0 every forward pass
- No overlap of gradient communication with backward pass

**Goal**: Replace with `DistributedDataParallel` (DDP) + `torch.compile` for significantly faster training. DDP uses one process per GPU with ring-allreduce gradient sync overlapped with backward. `torch.compile` fuses kernels for faster forward/backward.

**Constraint**: Keep changes minimal and localized. Don't restructure M3PO logic.

---

## Files to modify

- **`grpo_train_single.py`** — all changes are here (except one trivial util extraction already done)

No changes needed to `m3po_utils.py` or `utils.py` (no `cuda:0` hardcoding).

---

## Changes

### 1. Add imports (top of file, ~line 20)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
```

### 2. Add DDP helper functions (before `selective_log_softmax`, ~line 65)

```python
def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0
```

### 3. Replace all `cuda:0` hardcoding with `cuda` (current device)

After `torch.cuda.set_device(local_rank)`, `torch.device("cuda")` automatically uses the correct GPU.

Change at **5 locations** (lines 332, 427, 486, 632, 934):
```python
# Before:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# After:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 4. Modify `train_with_grpo` signature (~line 585)

- Remove `device_ids` parameter
- Add `local_rank=0` parameter

### 5. Replace DataParallel with DDP + torch.compile (~lines 655-666)

```python
# Before:
if device_ids is not None and len(device_ids) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)
    ...
raw_model = model.module if is_data_parallel else model

# After:
if dist.is_initialized():
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
raw_model = model.module if dist.is_initialized() else model

# torch.compile for faster forward/backward (skip generation via disable() context)
compiled_model = torch.compile(raw_model)
```

### 6. Use `compiled_model` for log prob computation, `raw_model` for generation

In `generate_rollout_data` (~line 401):
- Add `compiled_model=None` parameter
- Use `compiled_model or model` for `compute_log_probs` calls (lines 439-440)
- Keep `model` (raw_model) for `generate_completions`

In `grpo_loss` (~line 458):
- Add `compiled_model=None` parameter
- Use `compiled_model or model` for `compute_log_probs` / `compute_log_probs_with_m3po` calls

In `train_with_grpo` call sites:
- Pass `compiled_model=compiled_model` to `generate_rollout_data` and `grpo_loss`

### 7. Sync gating function gradients across ranks (~line 808, after backward)

```python
if has_learnable_gating and dist.is_initialized():
    for p in gating_function.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
```

### 8. Guard logging/printing behind `is_main_process()`

In the training loop (~lines 824-847):
- Wrap `wandb.log(...)` and `print(...)` with `if is_main_process():`

### 9. Restructure `__main__` block (~line 932+)

```python
if __name__ == "__main__":
    local_rank, rank, world_size = setup_ddp()
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map={"": local_rank}  # Each rank loads to its own GPU
    )
    ...
    # Rank-specific seed for different data sampling per rank
    set_random_seed(42 + rank)

    # Only rank 0 does wandb/eval/saving
    if is_main_process():
        wandb.init(...)

    model = train_with_grpo(..., local_rank=local_rank)
    # Remove device_ids from call

    if is_main_process():
        wandb.finish()
        evaluate_model(...)
        model.save_pretrained(...)
        # ... saving and push_to_hub

    cleanup_ddp()
```

Remove `reserve_gpu_memory()` call (unnecessary with DDP — each process owns its GPU).

Remove `device_ids` variable and logic.

### 10. Launch command

```bash
# Before:
python grpo_train_single.py

# After:
torchrun --nproc_per_node=NUM_GPUS grpo_train_single.py
```

---

## What stays unchanged

- All M3PO logic (generation, cross-path attention, gating)
- `generate_completions`, `compute_log_probs`, `compute_log_probs_with_m3po` core logic
- Reward functions, advantage computation, PPO clipping
- Two-phase gating warmup
- Gradient accumulation logic
- LR scheduler logic
- `utils.py`, `m3po_utils.py`

---

## Why this is safe

1. **Generation**: Already uses `raw_model` (unwrapped). Each DDP rank generates its own batches independently — same as before.
2. **M3PO cross-path**: All N paths for a question stay on the same GPU (same rank). No cross-rank interaction needed.
3. **Gradients**: DDP auto-syncs model gradients. Gating params synced via manual all-reduce (3 lines of code).
4. **Reference model**: Each rank deep-copies its own `raw_model`. Since DDP keeps weights in sync, all ref_models are identical.
5. **torch.compile**: Only applied to forward pass model, not generation. Falls back gracefully if issues arise.

---

## Verification

1. **Single GPU**: `torchrun --nproc_per_node=1 grpo_train_single.py` — should work identically to before
2. **Multi GPU**: `torchrun --nproc_per_node=4 grpo_train_single.py` — check wandb logs show only one run, loss decreases, rewards improve
3. **Ctrl+C**: Verify early stop still saves model correctly
4. **Gating**: Test with `gating_type='luong'` to verify learnable params train correctly across ranks
5. **torch.compile**: If issues, can be disabled by changing `compiled_model = torch.compile(raw_model)` to `compiled_model = raw_model`