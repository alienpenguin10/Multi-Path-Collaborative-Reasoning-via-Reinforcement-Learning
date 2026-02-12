#!/usr/bin/env python3
"""
GPU Monitor Script - Automatically runs grpo_train.py when 4+ GPUs are free.

A GPU is considered "free" if:
- GPU utilization is 0%
- Memory usage is < 500 MiB (allows for base driver memory)
"""

import subprocess
import time
import re
import sys
from datetime import datetime


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


def count_free_gpus(gpus, memory_threshold_mb=500):
    """
    Count GPUs that are free (0% util and < threshold memory usage).
    """
    free_gpus = []
    for gpu in gpus:
        if gpu['util_percent'] == 0 and gpu['memory_used_mb'] < memory_threshold_mb:
            free_gpus.append(gpu['gpu_id'])
    return free_gpus


def main():
    check_interval = 60  # Check every 60 seconds
    required_free_gpus = 4
    memory_threshold_mb = 500  # Consider GPU free if < 500 MiB used

    print(f"GPU Monitor Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Waiting for {required_free_gpus}+ GPUs to be free...")
    print(f"(Free = 0% utilization and < {memory_threshold_mb} MiB memory used)")
    print(f"Checking every {check_interval} seconds...\n")

    while True:
        gpus = get_gpu_status()
        if not gpus:
            print("Warning: Could not get GPU status, retrying...")
            time.sleep(check_interval)
            continue

        free_gpu_ids = count_free_gpus(gpus, memory_threshold_mb)
        num_free = len(free_gpu_ids)

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Free GPUs: {num_free}/{len(gpus)} - IDs: {free_gpu_ids if free_gpu_ids else 'none'}")

        if num_free >= required_free_gpus:
            print(f"\n✓ {num_free} GPUs are now free!")
            print(f"Launching grpo_train.py...\n")

            # Launch the training script
            try:
                subprocess.run(['python', 'grpo_train.py'], check=True)
                print("\nTraining completed successfully!")
                sys.exit(0)
            except subprocess.CalledProcessError as e:
                print(f"\nError: Training script failed with return code {e.returncode}", file=sys.stderr)
                sys.exit(1)
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                sys.exit(130)

        # Wait before next check
        try:
            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            sys.exit(0)


if __name__ == "__main__":
    main()
