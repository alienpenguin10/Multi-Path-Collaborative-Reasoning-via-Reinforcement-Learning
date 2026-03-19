#!/usr/bin/env python3
"""Scan for free GPUs and hold them with dummy tensors until Ctrl+C."""

import subprocess
import signal
import sys
import time

import torch


def get_free_gpus(mem_threshold_mb=500, util_threshold=10):
    """Return list of GPU indices that are free (low memory and utilization)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    free = []
    for line in result.stdout.strip().split("\n"):
        idx, mem_used, util = [x.strip() for x in line.split(",")]
        if int(mem_used) < mem_threshold_mb and int(util) < util_threshold:
            free.append(int(idx))
    return free


def hold_gpus(gpu_ids):
    """Allocate tensors on each GPU to reserve them. Returns list of tensors."""
    holders = []
    for gid in gpu_ids:
        dev = torch.device(f"cuda:{gid}")
        # Allocate ~5 GB to mark the GPU as in-use
        # 1024*1024*5 = 5,242,880 floats (float32 = 4 bytes) ≈ 20.97MB, so we need 5*1024 MB / 4 = 1,280,000,000 floats
        # To allocate 5GB, use float32: 5 * 1024**3 / 4 = 1,342,177,280 elements
        # Let's use a shape that fits: e.g., (640, 1024, 1920) ~5GB (640*1024*1920*4 bytes ≈ 5GB)
        t = torch.zeros(640, 1024, 1920, device=dev)
        holders.append(t)
        print(f"  Holding GPU {gid}")
    return holders


def main():
    min_gpus = 2
    print(f"Scanning for >{min_gpus} free GPUs...")

    while True:
        free = get_free_gpus()
        if len(free) > min_gpus:
            print(f"Found {len(free)} free GPUs: {free}")
            break
        print(f"  Only {len(free)} free GPUs ({free}), waiting...", end="\r")
        time.sleep(5)

    holders = hold_gpus(free)
    print(f"\nHolding {len(free)} GPUs. Press Ctrl+C to release.")

    def release(sig, frame):
        print("\nReleasing GPUs...")
        holders.clear()
        torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGINT, release)
    signal.signal(signal.SIGTERM, release)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
