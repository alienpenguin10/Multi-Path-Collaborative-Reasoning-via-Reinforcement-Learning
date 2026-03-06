#!/usr/bin/env python3
"""
M3PO Experiment Runner — Automates training + evaluation for gating function variants.

Usage:
    # Run a single experiment
    python run_m3po_experiment.py --gating_type raw_dot --trial 1

    # Run all 8 variants × 3 trials
    python run_m3po_experiment.py --run_all --num_trials 3

    # No-M3PO baseline (control)
    python run_m3po_experiment.py --gating_type none --trial 1

    # Evaluate an existing model only (skip training)
    python run_m3po_experiment.py --gating_type raw_dot --trial 1 --eval_only

    # Quick test with few steps
    python run_m3po_experiment.py --gating_type raw_dot --trial 1 --num_steps 5
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

# Must be set before any CUDA operations to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb

# Use the local transformers fork
sys.path.insert(0, os.path.abspath("transformers/src"))

os.environ["M3PO_DEBUG"] = "-1"

from dotenv import load_dotenv
load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "m3po-experiments")

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import set_random_seed, prepare_dataset, evaluate_model, combined_reward
from grpo_train import train_with_grpo, optimize_model_memory, reserve_gpu_memory

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
BASE_OUTPUT_DIR = "outputs"

# ── Core Experiment Logic ────────────────────────────────────────────────────


def build_training_config(gating_type, args):
    """Build training config dict for a given gating type and CLI args."""
    config = {
        "num_iterations": 1,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "beta": 0.005,
        "learning_rate": 5e-6,
        "mu": 1,
        "epsilon": 0.1,
        "lambda_blend": 0.1,
        "temperature_m3po": 0.1,
    }

    if gating_type == "none":
        config["use_m3po"] = False
        config["gating_type"] = "baseline"
    else:
        config["use_m3po"] = True
        config["gating_type"] = gating_type

    config["gating_config"] = {
        "temperature": 0.1,
        "debug": False,
    }

    # Learnable gating params (Luong/Bahdanau)
    if args.gating_rank is not None:
        config["gating_config"]["rank"] = args.gating_rank
    if args.gating_init_strategy is not None:
        config["gating_config"]["init_strategy"] = args.gating_init_strategy
    if args.gating_warmup_steps > 0:
        config["gating_warmup_steps"] = args.gating_warmup_steps
    if args.gating_lr is not None:
        config["gating_lr"] = args.gating_lr
    if args.gating_grad_clip is not None:
        config["gating_grad_clip"] = args.gating_grad_clip

    return config


def get_output_dir(gating_type, trial):
    """Get the output directory path for an experiment."""
    return os.path.join(BASE_OUTPUT_DIR, gating_type, f"trial_{trial}")


def load_model_and_tokenizer(model_name=MODEL_NAME):
    """Load the base model and tokenizer."""
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


def prepare_data(seed, eval_size):
    """Prepare train/eval split with deterministic seed."""
    set_random_seed(seed)
    all_data = prepare_dataset("test")
    random.shuffle(all_data)
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    return train_data, eval_data


def run_single_experiment(gating_type, trial, args):
    """
    Run a single training + evaluation experiment.

    Args:
        gating_type: One of ALL_GATING_TYPES
        trial: Trial number (1, 2, 3, ...)
        args: Parsed CLI arguments

    Returns:
        dict with results, or None if skipped
    """
    output_dir = get_output_dir(gating_type, trial)
    results_path = os.path.join(output_dir, "results.json")
    seed = args.seed_base + trial - 1

    # Resume-friendly: skip if already completed
    if os.path.exists(results_path) and not args.force:
        print(f"\n{'='*60}")
        print(f"SKIP: {gating_type} trial {trial} — results already exist at {results_path}")
        print(f"      Use --force to re-run.")
        print(f"{'='*60}")
        with open(results_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: gating_type={gating_type}, trial={trial}, seed={seed}")
    print(f"OUTPUT: {output_dir}")
    print(f"{'='*60}\n")

    # Set seed for reproducibility
    set_random_seed(seed)

    # Prepare data (same seed → same split across gating types for same trial)
    train_data, eval_data = prepare_data(seed, args.eval_size)
    print(f"Data prepared: {len(train_data)} train, {len(eval_data)} eval examples")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(num_gpus)) if num_gpus > 1 else None
    print(f"Using {num_gpus} GPUs, device_ids={device_ids}")

    if args.eval_only:
        # Eval-only mode: load from output_dir
        if not os.path.exists(output_dir):
            print(f"ERROR: No model found at {output_dir} for eval-only mode")
            return None
        print(f"Eval-only mode: loading model from {output_dir}")
        model = AutoModelForCausalLM.from_pretrained(
            output_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Full training mode
        model, tokenizer = load_model_and_tokenizer()
        model = optimize_model_memory(model)

        # Reserve remaining GPU memory AFTER model is loaded to avoid fragmentation
        print("Reserving GPU memory...")
        reserve_gpu_memory()

        training_config = build_training_config(gating_type, args)

        # Init wandb
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
            device_ids=device_ids,
            **training_config,
        )

        train_duration = time.time() - start_time

        wandb.finish()
        print(f"Training completed in {train_duration:.1f}s. Wandb run finished.")

    # Save model first (before eval, in case eval crashes)
    os.makedirs(output_dir, exist_ok=True)

    if not args.eval_only:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(training_config, f, indent=2)

    # Full evaluation on the entire GSM8K test set
    print("\nRunning full evaluation on entire GSM8K test set...")
    full_eval_data = prepare_dataset("test")
    print(f"Full eval set size: {len(full_eval_data)}")
    accuracy = evaluate_model(model, tokenizer, full_eval_data, device)
    print(f"Full eval accuracy: {accuracy:.2f}%")

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
        results["num_steps"] = args.num_steps
        results["batch_size"] = args.batch_size

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def run_all_experiments(args):
    """Run all gating types × all trials. Print summary at the end."""
    all_results = []

    for gating_type in ALL_GATING_TYPES:
        for trial in range(1, args.num_trials + 1):
            result = run_single_experiment(gating_type, trial, args)
            if result is not None:
                all_results.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Gating Type':<16} {'Trial':<8} {'Accuracy':<12} {'Duration':<12}")
    print(f"{'-'*16} {'-'*8} {'-'*12} {'-'*12}")

    for r in all_results:
        duration = f"{r.get('train_duration_s', 'N/A')}s" if "train_duration_s" in r else "N/A"
        print(f"{r['gating_type']:<16} {r['trial']:<8} {r['accuracy']:<12.2f} {duration:<12}")

    # Per-variant averages
    print(f"\n{'='*50}")
    print("AVERAGES BY GATING TYPE")
    print(f"{'='*50}")
    print(f"{'Gating Type':<16} {'Mean Acc':<12} {'Std':<12} {'N':<6}")
    print(f"{'-'*16} {'-'*12} {'-'*12} {'-'*6}")

    from collections import defaultdict
    import numpy as np

    by_type = defaultdict(list)
    for r in all_results:
        by_type[r["gating_type"]].append(r["accuracy"])

    sorted_types = sorted(by_type.keys(), key=lambda k: -np.mean(by_type[k]))
    for gtype in sorted_types:
        accs = by_type[gtype]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) > 1 else 0.0
        print(f"{gtype:<16} {mean_acc:<12.2f} {std_acc:<12.2f} {len(accs):<6}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="M3PO Experiment Runner — Train and evaluate gating function variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_m3po_experiment.py --gating_type raw_dot --trial 1
  python run_m3po_experiment.py --run_all --num_trials 3
  python run_m3po_experiment.py --gating_type none --trial 1
  python run_m3po_experiment.py --gating_type kl_divergence --trial 1 --eval_only
  python run_m3po_experiment.py --gating_type raw_dot --trial 1 --num_steps 5
        """,
    )

    # Experiment selection
    parser.add_argument(
        "--gating_type",
        type=str,
        choices=ALL_GATING_TYPES,
        help="Gating function variant to run",
    )
    parser.add_argument("--trial", type=int, default=1, help="Trial number (default: 1)")
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all gating types × all trials",
    )
    parser.add_argument(
        "--num_trials", type=int, default=3, help="Number of trials per variant (default: 3)"
    )

    # Training overrides
    parser.add_argument("--num_steps", type=int, default=500, help="Training steps (default: 500)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size (default: 5)")
    parser.add_argument("--num_generations", type=int, default=4, help="Generations per prompt (default: 4)")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens (default: 512)")
    parser.add_argument("--eval_size", type=int, default=30, help="Number of eval examples (default: 30)")
    parser.add_argument("--seed_base", type=int, default=42, help="Base seed; trial seed = base + trial - 1 (default: 42)")

    # Learnable gating params (Luong/Bahdanau)
    parser.add_argument("--gating_warmup_steps", type=int, default=0, help="Steps to train only gating params before unfreezing model (default: 0)")
    parser.add_argument("--gating_lr", type=float, default=None, help="Learning rate for gating params (default: 100x model LR)")
    parser.add_argument("--gating_grad_clip", type=float, default=None, help="Gradient clip norm for gating params (default: 1.0)")
    parser.add_argument("--gating_rank", type=int, default=None, help="Low-rank factorization rank for Luong/Bahdanau (default: 2048)")
    parser.add_argument("--gating_init_strategy", type=str, default=None, choices=["identity", "xavier"], help="Init strategy for learnable gating (default: identity)")

    # Modes
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, only evaluate existing model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if results already exist",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_all:
        run_all_experiments(args)
    elif args.gating_type is not None:
        run_single_experiment(args.gating_type, args.trial, args)
    else:
        print("Error: specify --gating_type or --run_all")
        print("Run with --help for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
