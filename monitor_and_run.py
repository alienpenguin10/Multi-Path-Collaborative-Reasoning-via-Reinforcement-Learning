#!/usr/bin/env python3
"""
GPU Monitor Script - Automatically runs grpo_train.py when 4+ GPUs are free.

A GPU is considered "free" if:
- GPU utilization is 0%
- Memory usage is < 500 MiB (allows for base driver memory)

The script:
1. Polls nvidia-smi every 60 seconds
2. When enough GPUs are free, sets CUDA_VISIBLE_DEVICES to only those GPUs
3. Activates the conda environment and launches grpo_train.py
"""

import subprocess
import os
import time
import sys
from datetime import datetime


# ── Configuration ────────────────────────────────────────────────────────────
REQUIRED_FREE_GPUS = 2          # Minimum free GPUs needed to launch
CHECK_INTERVAL = 240             # Seconds between checks
MEMORY_THRESHOLD_MB = 500       # GPU is "free" if memory < this
CONDA_ENV = "ant"               # Conda environment name
CONDA_PYTHON = os.path.expanduser("~/Neuralese/miniconda3/envs/ant/bin/python")
CONDA_TORCHRUN = os.path.expanduser("~/Neuralese/miniconda3/envs/ant/bin/torchrun")
TARGET_GPUS = None               # Consider all GPUs (set to list like [3, 6, 7] to restrict)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_SCRIPT = os.path.join(SCRIPT_DIR, "run_m3po_experiment.py")
EXPERIMENT_ARGS = ["--gating_type", "luong", "--trial", "2"]
# ─────────────────────────────────────────────────────────────────────────────


def get_gpu_status():
    """
    Parse nvidia-smi output to get GPU status.
    Returns list of dicts with {gpu_id, util_percent, memory_used_mb, memory_total_mb}
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [x.strip() for x in line.split(',')]
                gpus.append({
                    'gpu_id': int(parts[0]),
                    'util_percent': int(parts[1]),
                    'memory_used_mb': int(parts[2]),
                    'memory_total_mb': int(parts[3])
                })
        return gpus
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}", file=sys.stderr)
        return []


def find_free_gpus(gpus, memory_threshold_mb=500):
    """Return list of GPU IDs that are free (0% util and < threshold memory)."""
    return [
        gpu['gpu_id'] for gpu in gpus
        if gpu['util_percent'] == 0 and gpu['memory_used_mb'] < memory_threshold_mb
    ]


def launch_training(free_gpu_ids):
    """
    Launch run_m3po_experiment.py with CUDA_VISIBLE_DEVICES set to only the free GPUs.

    The experiment script uses nn.DataParallel with device_ids=list(range(torch.cuda.device_count())),
    so setting CUDA_VISIBLE_DEVICES is the correct way to control which GPUs it uses.
    The script will see them as GPU 0, 1, 2, ... regardless of physical IDs.
    """
    gpu_str = ",".join(str(g) for g in free_gpu_ids)

    # Build environment: inherit current env + set CUDA_VISIBLE_DEVICES
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    cmd = [CONDA_TORCHRUN, f"--nproc_per_node={len(free_gpu_ids)}", EXPERIMENT_SCRIPT] + EXPERIMENT_ARGS
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_str}")
    print(f"Training will see {len(free_gpu_ids)} GPUs (remapped as 0..{len(free_gpu_ids)-1})")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Use the current Python interpreter (inherits the active conda env)
    subprocess.run(
        cmd,
        env=env,
        cwd=SCRIPT_DIR,
        check=True,
    )


def hold_gpus(gpu_ids):
    """
    Allocate large tensors on each GPU to reserve VRAM and prevent others from using them.
    Blocks until Ctrl+C is pressed.
    """
    if "torch" in sys.modules:
        # torch was already imported — CUDA_VISIBLE_DEVICES changes won't take effect.
        # Spawn a subprocess to hold GPUs with a clean torch import.
        gpu_str = ",".join(str(g) for g in gpu_ids)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_str
        script = (
            "import torch, time\n"
            f"holders = [torch.empty(20*1024*1024*1024//4, dtype=torch.float32, device=f'cuda:{{i}}') for i in range({len(gpu_ids)})]\n"
            f"print(f'\\nHolding {len(gpu_ids)} GPUs (physical IDs: {gpu_ids}). Press Ctrl+C to release.')\n"
            "try:\n"
            "    while True:\n"
            "        time.sleep(60)\n"
            "except KeyboardInterrupt:\n"
            "    print('\\nReleasing GPUs...')\n"
        )
        subprocess.run([CONDA_PYTHON, "-c", script], env=env)
        return

    import torch

    num_gpus = len(gpu_ids)
    holders = []
    for i in range(num_gpus):
        # Allocate ~20GB per GPU to reserve most of the VRAM
        t = torch.empty(20 * 1024 * 1024 * 1024 // 4, dtype=torch.float32, device=f"cuda:{i}")
        holders.append(t)

    print(f"\nHolding {num_gpus} GPUs (physical IDs: {gpu_ids}). Press Ctrl+C to release.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nReleasing GPUs...")
        del holders
        torch.cuda.empty_cache()


def main():
    print(f"GPU Monitor Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:    {sys.executable}")
    print(f"  Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'N/A')}")
    print(f"  Script:    {EXPERIMENT_SCRIPT}")
    print(f"  Args:      {EXPERIMENT_ARGS}")
    print(f"  Requires:  {REQUIRED_FREE_GPUS}+ free GPUs")
    print(f"  Free = 0% utilization AND < {MEMORY_THRESHOLD_MB} MiB memory")
    print(f"  Checking every {CHECK_INTERVAL} seconds...\n")

    while True:
        gpus = get_gpu_status()
        if not gpus:
            print("Warning: Could not get GPU status, retrying...")
            time.sleep(CHECK_INTERVAL)
            continue

        all_free = find_free_gpus(gpus, MEMORY_THRESHOLD_MB)
        free_gpu_ids = [g for g in all_free if g in TARGET_GPUS] if TARGET_GPUS else all_free
        num_free = len(free_gpu_ids)

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Free GPUs: {num_free}/{len(gpus)} - IDs: {free_gpu_ids if free_gpu_ids else 'none'}")

        if num_free >= REQUIRED_FREE_GPUS:
            print(f"\n{'='*60}")
            print(f"{num_free} GPUs are free! Launching training...")
            print(f"Using GPUs: {free_gpu_ids}")
            print(f"{'='*60}\n")

            # Set CUDA_VISIBLE_DEVICES in parent process before torch import (for hold_gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in free_gpu_ids)

            try:
                launch_training(free_gpu_ids)
                print("\nTraining completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"\nError: Training script failed with return code {e.returncode}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")

            # Always hold GPUs after training finishes/fails/is interrupted
            hold_gpus(free_gpu_ids)
            sys.exit(0)

        try:
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
