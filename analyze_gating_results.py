#!/usr/bin/env python3
"""
M3PO Gating Results Analyzer — Compare gating function variants.

Reads results from outputs/{gating_type}/trial_{n}/results.json,
generates comparison plots, summary tables, and statistical significance tests.

Usage:
    python analyze_gating_results.py
    python analyze_gating_results.py --variants raw_dot kl_divergence baseline
    python analyze_gating_results.py --plot_dir plots/ --results_dir outputs/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

# ── Data Loading ─────────────────────────────────────────────────────────────


def load_all_results(results_dir="outputs"):
    """
    Load all experiment results from the output directory structure.

    Expects: results_dir/{gating_type}/trial_{n}/results.json

    Returns:
        pandas DataFrame with columns: gating_type, trial, accuracy, seed, etc.
    """
    records = []

    if not os.path.exists(results_dir):
        print(f"Warning: results directory '{results_dir}' does not exist")
        return pd.DataFrame()

    for gating_type in sorted(os.listdir(results_dir)):
        gating_dir = os.path.join(results_dir, gating_type)
        if not os.path.isdir(gating_dir):
            continue

        for trial_dir_name in sorted(os.listdir(gating_dir)):
            if not trial_dir_name.startswith("trial_"):
                continue

            results_path = os.path.join(gating_dir, trial_dir_name, "results.json")
            if not os.path.exists(results_path):
                continue

            with open(results_path) as f:
                result = json.load(f)

            # Ensure required fields
            result.setdefault("gating_type", gating_type)
            result.setdefault("trial", int(trial_dir_name.split("_")[1]))
            records.append(result)

    if not records:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} experiment results across {df['gating_type'].nunique()} gating types")
    return df


def load_wandb_history(project=None):
    """
    Load training curves from wandb (optional).

    Returns dict: {run_name: DataFrame with step, loss, reward columns}
    """
    if project is None:
        project = os.getenv("WANDB_PROJECT", "m3po-experiments")

    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(project)
    except Exception as e:
        print(f"Could not load wandb data: {e}")
        return {}

    histories = {}
    for run in runs:
        try:
            history = run.history(keys=["loss", "average_reward", "m3po/attention_entropy_mean"])
            if not history.empty:
                histories[run.name] = history
        except Exception:
            continue

    print(f"Loaded wandb history for {len(histories)} runs")
    return histories


# ── Summary Table ────────────────────────────────────────────────────────────


def print_summary_table(df):
    """Print and return a summary table: mean accuracy +/- std per gating type."""
    summary = (
        df.groupby("gating_type")["accuracy"]
        .agg(["mean", "std", "count", "min", "max"])
        .round(2)
    )
    summary.columns = ["Mean Acc %", "Std", "N Trials", "Min", "Max"]
    summary = summary.sort_values("Mean Acc %", ascending=False)

    # Include duration if available
    if "train_duration_s" in df.columns:
        duration_summary = (
            df.groupby("gating_type")["train_duration_s"]
            .agg(["mean"])
            .round(0)
        )
        duration_summary.columns = ["Mean Duration (s)"]
        summary = summary.join(duration_summary)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(summary.to_string())
    print("=" * 70)

    return summary


# ── Statistical Analysis ─────────────────────────────────────────────────────


def compute_pairwise_significance(df, alpha=0.05):
    """
    Compute pairwise statistical significance between all gating type pairs.

    Uses Welch's t-test (unequal variance t-test).

    Returns:
        DataFrame with p-values for each pair, and a summary of significant differences.
    """
    try:
        from scipy import stats
    except ImportError:
        print("scipy not installed — skipping statistical significance testing.")
        print("Install with: pip install scipy")
        return None, None

    gating_types = sorted(df["gating_type"].unique())
    n = len(gating_types)

    p_values = pd.DataFrame(np.ones((n, n)), index=gating_types, columns=gating_types)
    effect_sizes = pd.DataFrame(np.zeros((n, n)), index=gating_types, columns=gating_types)

    significant_pairs = []

    for g1, g2 in combinations(gating_types, 2):
        acc1 = df[df["gating_type"] == g1]["accuracy"].values
        acc2 = df[df["gating_type"] == g2]["accuracy"].values

        if len(acc1) < 2 or len(acc2) < 2:
            continue

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(acc1, acc2, equal_var=False)
        p_values.loc[g1, g2] = p_val
        p_values.loc[g2, g1] = p_val

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(acc1) ** 2 + np.std(acc2) ** 2) / 2)
        if pooled_std > 0:
            d = (np.mean(acc1) - np.mean(acc2)) / pooled_std
        else:
            d = 0.0
        effect_sizes.loc[g1, g2] = d
        effect_sizes.loc[g2, g1] = -d

        if p_val < alpha:
            winner = g1 if np.mean(acc1) > np.mean(acc2) else g2
            significant_pairs.append({
                "pair": f"{g1} vs {g2}",
                "p_value": round(p_val, 4),
                "effect_size_d": round(d, 3),
                "winner": winner,
            })

    # Print significant differences
    print(f"\n{'='*70}")
    print(f"STATISTICAL SIGNIFICANCE (alpha={alpha})")
    print(f"{'='*70}")
    if significant_pairs:
        sig_df = pd.DataFrame(significant_pairs)
        print(sig_df.to_string(index=False))
    else:
        print("No statistically significant differences found.")
    print(f"{'='*70}")

    return p_values, effect_sizes


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_accuracy_comparison(df, plot_dir):
    """Bar plot: mean accuracy by gating type with error bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = df.groupby("gating_type")["accuracy"].agg(["mean", "std"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(summary))
    bars = ax.bar(x, summary["mean"], yerr=summary["std"], capsize=5, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("M3PO Gating Function Comparison — Accuracy")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{row['mean']:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(plot_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_accuracy_boxplot(df, plot_dir):
    """Box plot: accuracy distribution per variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort by median accuracy
    gating_order = (
        df.groupby("gating_type")["accuracy"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [df[df["gating_type"] == g]["accuracy"].values for g in gating_order]
    bp = ax.boxplot(data, tick_labels=gating_order, patch_artist=True)

    colors = plt.cm.Set3(np.linspace(0, 1, len(gating_order)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("M3PO Gating Function Comparison — Accuracy Distribution")
    ax.set_xticklabels(gating_order, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "accuracy_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_significance_heatmap(p_values, plot_dir):
    """Heatmap of pairwise p-values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use log scale for better visibility
    display_vals = -np.log10(p_values.values.clip(min=1e-10))
    im = ax.imshow(display_vals, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(p_values.columns)))
    ax.set_yticks(range(len(p_values.index)))
    ax.set_xticklabels(p_values.columns, rotation=45, ha="right")
    ax.set_yticklabels(p_values.index)

    # Annotate with actual p-values
    for i in range(len(p_values.index)):
        for j in range(len(p_values.columns)):
            val = p_values.iloc[i, j]
            text = f"{val:.3f}" if val >= 0.001 else f"{val:.1e}"
            color = "white" if display_vals[i, j] > 1.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Pairwise Significance (-log10 p-value)\nHigher = more significant")
    plt.colorbar(im, label="-log10(p-value)")
    plt.tight_layout()
    path = os.path.join(plot_dir, "significance_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_convergence_curves(wandb_histories, plot_dir):
    """Plot loss and reward curves from wandb history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not wandb_histories:
        print("No wandb history available, skipping convergence plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run_name, history in wandb_histories.items():
        if "loss" in history.columns:
            axes[0].plot(history.index, history["loss"], label=run_name, alpha=0.7)
        if "average_reward" in history.columns:
            axes[1].plot(history.index, history["average_reward"], label=run_name, alpha=0.7)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss by Gating Function")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Average Reward by Gating Function")
    axes[1].legend(fontsize=7, loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "convergence_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze M3PO gating experiment results",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs",
        help="Base directory containing experiment outputs (default: outputs/)",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots/)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        type=str,
        default=None,
        help="Only analyze these gating types (default: all found)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project for convergence curves (default: from env)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Skip loading wandb data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load results
    df = load_all_results(args.results_dir)
    if df.empty:
        print("No results to analyze. Run experiments first with run_m3po_experiment.py")
        sys.exit(1)

    # Filter to requested variants
    if args.variants:
        df = df[df["gating_type"].isin(args.variants)]
        if df.empty:
            print(f"No results found for variants: {args.variants}")
            sys.exit(1)

    # Summary table
    summary = print_summary_table(df)

    # Save summary CSV
    os.makedirs(args.plot_dir, exist_ok=True)
    csv_path = os.path.join(args.plot_dir, "results_summary.csv")
    summary.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    # Statistical significance (need ≥2 trials per variant)
    variants_with_multiple = df.groupby("gating_type").filter(lambda x: len(x) >= 2)
    if len(variants_with_multiple["gating_type"].unique()) >= 2:
        p_values, effect_sizes = compute_pairwise_significance(variants_with_multiple)
    else:
        print("\nNeed ≥2 trials per variant for significance testing. Skipping.")
        p_values = None

    # Plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(df, args.plot_dir)

    if df.groupby("gating_type")["accuracy"].count().max() > 1:
        plot_accuracy_boxplot(df, args.plot_dir)
    else:
        print("Only 1 trial per variant — skipping box plot")

    if p_values is not None:
        plot_significance_heatmap(p_values, args.plot_dir)

    # Convergence curves from wandb (optional)
    if not args.no_wandb:
        wandb_histories = load_wandb_history(args.wandb_project)
        plot_convergence_curves(wandb_histories, args.plot_dir)

    print(f"\nAnalysis complete. All outputs saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()
