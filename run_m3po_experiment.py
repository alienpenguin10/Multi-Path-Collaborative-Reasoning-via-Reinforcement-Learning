#!/usr/bin/env python3
"""
M3PO Experiment Runner — Automates training + evaluation for gating function variants.

Runs via torchrun for DDP:
    torchrun --nproc_per_node=2 run_m3po_experiment.py --gating_type luong --trial 1
    torchrun --nproc_per_node=2 run_m3po_experiment.py --run_all --num_trials 3
    torchrun --nproc_per_node=2 run_m3po_experiment.py --gating_type none --trial 1
    torchrun --nproc_per_node=2 run_m3po_experiment.py --gating_type luong --trial 1 --eval_only
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

# Must be set before any CUDA operations to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import wandb

# Use the local transformers fork
sys.path.insert(0, os.path.abspath("transformers/src"))

os.environ["M3PO_DEBUG"] = "-1"

from dotenv import load_dotenv
load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "m3po-experiments")

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_random_seed, prepare_dataset, evaluate_model, combined_reward, get_next_trial_number
from m3po_train import train_with_grpo, optimize_model_memory, setup_ddp, cleanup_ddp, is_main_process

# ── Constants ────────────────────────────────────────────────────────────────

ALL_GATING_TYPES = [
    "none",           # No M3PO (control)
    "baseline",       # Original cosine similarity
    "raw_dot",        # Raw dot product
    "scaled_dot",     # Scaled dot product
    "kl_divergence",  # Jensen-Shannon divergence
    "bhattacharyya",  # Bhattacharyya coefficient
    "luong",          # Luong (bilinear) attention
    "bahdanau",       # Bahdanau (additive) attention
]

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_OUTPUT_DIR = "output"

# ── Core Experiment Logic ────────────────────────────────────────────────────


def build_training_config(gating_type, args, train_data_size):
    """
    Build training config dict matching m3po_train.py hyperparameters exactly.
    """
    config = {
        "num_iterations": 1,
        "num_steps": args.num_steps if args.num_steps is not None else math.ceil(train_data_size / args.batch_size),
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": 0.005,
        "learning_rate": 5e-6,
        "mu": 2,                               # 2 gradient updates per rollout
        "epsilon": 0.1,
        # M3PO-specific parameters (from paper Table 3)
        "lambda_blend": 0.1,
        "temperature_m3po": 0.1,
        # Gradient accumulation and LR schedule
        "gradient_accumulation_steps": 4,       # Paper Table 3
        "warmup_ratio": 0.1,                    # Paper Table 3: cosine schedule with warmup
    }

    if gating_type == "none":
        config["use_m3po"] = False
        config["gating_type"] = "baseline"
    else:
        config["use_m3po"] = True
        config["gating_type"] = gating_type

    config["gating_config"] = {
        "temperature": 0.1,
        "rank": args.gating_rank,
        "init_strategy": args.gating_init_strategy,
        "debug": False,
    }

    config["gating_warmup_steps"] = args.gating_warmup_steps
    config["gating_lr"] = args.gating_lr
    config["gating_grad_clip"] = args.gating_grad_clip

    return config


def get_output_dir(gating_type, trial):
    """Get the output directory path for an experiment."""
    return os.path.join(BASE_OUTPUT_DIR, gating_type, f"trial_{trial}")


def run_single_experiment(gating_type, trial, args, local_rank, rank):
    """
    Run a single training + evaluation experiment.
    """
    output_dir = get_output_dir(gating_type, trial)
    results_path = os.path.join(output_dir, "results.json")
    seed = args.seed_base + trial - 1

    # Resume-friendly: skip if already completed
    if os.path.exists(results_path) and not args.force:
        if is_main_process():
            print(f"\nSKIP: {gating_type} trial {trial} — results already exist at {results_path}")
            print(f"      Use --force to re-run.")
        with open(results_path) as f:
            return json.load(f)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: gating_type={gating_type}, trial={trial}, seed={seed}")
        print(f"OUTPUT: {output_dir}")
        print(f"{'='*60}\n")

    device = torch.device("cuda")

    if args.eval_only:
        if not os.path.exists(output_dir):
            if is_main_process():
                print(f"ERROR: No model found at {output_dir} for eval-only mode")
            return None
        if is_main_process():
            print(f"Eval-only mode: loading model from {output_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16, device_map={"": local_rank}
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Load model — same as m3po_train.py
        if is_main_process():
            print(f"Loading model from {MODEL_NAME}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
        )
        if is_main_process():
            print("Model loaded")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # Prepare data — same split as m3po_train.py
        train_data = prepare_dataset("train")
        eval_data = prepare_dataset("test")
        train_data = train_data[:len(train_data) // 2]
        eval_data = eval_data[:len(eval_data) // 2]
        random.shuffle(train_data)

        # Rank-specific seed so each GPU samples different batches
        set_random_seed(seed + rank)

        model = optimize_model_memory(model)

        training_config = build_training_config(gating_type, args, len(train_data))

        # Init wandb (rank 0 only)
        if is_main_process():
            if gating_type == "none":
                run_name = f"no-M3PO trial {trial}"
            else:
                run_name = f"M3PO {gating_type} trial {trial}"

            wandb.init(
                project=os.getenv("WANDB_PROJECT", "m3po-experiments"),
                name=run_name,
                config=training_config,
                tags=[gating_type, f"trial_{trial}"],
                reinit=True,
            )

        start_time = time.time()

        model = train_with_grpo(
            model=model,
            tokenizer=tokenizer,
            train_data=train_data,
            reward_function=combined_reward,
            local_rank=local_rank,
            **training_config,
        )

        train_duration = time.time() - start_time

        if is_main_process():
            wandb.finish()
            print(f"Training completed in {train_duration:.1f}s. Wandb run finished.")

    # Post-training: eval + save (rank 0 only)
    if is_main_process():
        print("\nFinal model evaluation after GRPO RL fine-tuning:")
        full_eval_data = prepare_dataset("test")
        accuracy = evaluate_model(model, tokenizer, full_eval_data, device)
        print(f"Accuracy: {accuracy:.2f}%")

        os.makedirs(output_dir, exist_ok=True)

        if not args.eval_only:
            print(f"Saving model to {output_dir}...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            serializable_config = {k: v for k, v in training_config.items()
                                   if isinstance(v, (int, float, str, bool, dict, list, type(None)))}
            serializable_config["trial"] = trial
            serializable_config["model_name"] = MODEL_NAME
            with open(os.path.join(output_dir, "training_config.json"), "w") as f:
                json.dump(serializable_config, f, indent=2)

        results = {
            "gating_type": gating_type,
            "trial": trial,
            "seed": seed,
            "accuracy": accuracy,
            "eval_size": len(full_eval_data),
            "timestamp": datetime.now().isoformat(),
        }
        if not args.eval_only:
            results["train_duration_s"] = round(train_duration, 1)

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

        return results

    # Non-rank-0 processes wait
    if dist.is_initialized():
        dist.barrier()
    return None


def run_all_experiments(args, local_rank, rank):
    """Run all gating types x all trials. Print summary at the end."""
    all_results = []

    for gating_type in ALL_GATING_TYPES:
        for trial in range(1, args.num_trials + 1):
            result = run_single_experiment(gating_type, trial, args, local_rank, rank)
            if result is not None:
                all_results.append(result)

    if is_main_process() and all_results:
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"{'Gating Type':<16} {'Trial':<8} {'Accuracy':<12} {'Duration':<12}")
        print(f"{'-'*16} {'-'*8} {'-'*12} {'-'*12}")

        for r in all_results:
            duration = f"{r.get('train_duration_s', 'N/A')}s" if "train_duration_s" in r else "N/A"
            print(f"{r['gating_type']:<16} {r['trial']:<8} {r['accuracy']:<12.2f} {duration:<12}")

        from collections import defaultdict
        import numpy as np

        by_type = defaultdict(list)
        for r in all_results:
            by_type[r["gating_type"]].append(r["accuracy"])

        print(f"\n{'='*50}")
        print("AVERAGES BY GATING TYPE")
        print(f"{'='*50}")
        sorted_types = sorted(by_type.keys(), key=lambda k: -np.mean(by_type[k]))
        for gtype in sorted_types:
            accs = by_type[gtype]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs) if len(accs) > 1 else 0.0
            print(f"{gtype:<16} {mean_acc:<12.2f} {std_acc:<12.2f} {len(accs)}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="M3PO Experiment Runner — Train and evaluate gating function variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  torchrun --nproc_per_node=2 run_m3po_experiment.py --gating_type luong --trial 1
  torchrun --nproc_per_node=2 run_m3po_experiment.py --run_all --num_trials 3
  torchrun --nproc_per_node=2 run_m3po_experiment.py --gating_type none --trial 1
        """,
    )

    # Experiment selection
    parser.add_argument("--gating_type", type=str, choices=ALL_GATING_TYPES, help="Gating function variant")
    parser.add_argument("--trial", type=int, default=1, help="Trial number (default: 1)")
    parser.add_argument("--run_all", action="store_true", help="Run all gating types x all trials")
    parser.add_argument("--num_trials", type=int, default=3, help="Trials per variant (default: 3)")

    # Training overrides (defaults match m3po_train.py)
    parser.add_argument("--num_steps", type=int, default=None, help="Training steps (default: full epoch)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--num_generations", type=int, default=4, help="Generations per prompt (default: 4)")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens (default: 512)")
    parser.add_argument("--seed_base", type=int, default=42, help="Base seed (default: 42)")

    # Learnable gating params (defaults match m3po_train.py)
    parser.add_argument("--gating_warmup_steps", type=int, default=50, help="Warmup steps for gating-only training (default: 50)")
    parser.add_argument("--gating_lr", type=float, default=5e-4, help="Learning rate for gating params (default: 5e-4)")
    parser.add_argument("--gating_grad_clip", type=float, default=1.0, help="Gradient clip for gating params (default: 1.0)")
    parser.add_argument("--gating_rank", type=int, default=32, help="Low-rank factorization rank (default: 32)")
    parser.add_argument("--gating_init_strategy", type=str, default="identity", choices=["identity", "xavier"], help="Init strategy (default: identity)")

    # Modes
    parser.add_argument("--eval_only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")

    return parser.parse_args()


def main():
    # DDP setup first — torchrun sets LOCAL_RANK, RANK, WORLD_SIZE
    local_rank, rank, world_size = setup_ddp()
    if is_main_process():
        print(f"DDP initialized: {world_size} GPU(s)")

    args = parse_args()

    if args.run_all:
        run_all_experiments(args, local_rank, rank)
    elif args.gating_type is not None:
        run_single_experiment(args.gating_type, args.trial, args, local_rank, rank)
    else:
        if is_main_process():
            print("Error: specify --gating_type or --run_all")
        sys.exit(1)

    # Clean up DDP
    if dist.is_initialized():
        dist.barrier()
    cleanup_ddp()


if __name__ == "__main__":
    main()
