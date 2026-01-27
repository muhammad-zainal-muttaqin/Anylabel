"""
Compare Real Depth (A.2) vs Synthetic Depth (A.4a)

Analyze and visualize performance differences between real depth from
RealSense sensor and synthetic depth from Depth-Anything-V2.

Usage:
    python compare_real_vs_synthetic.py

Requirements:
    - A.2 results (real depth)
    - A.4a results (synthetic depth)
    - Both evaluated on test set

Author: Research Team
Date: 2026-01-21
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_DIR / "Experiments" / "runs" / "detect"
OUTPUT_DIR = PROJECT_DIR / "Experiments" / "eda_output" / "depth_comparison"

# Expected experiment directories
A2_DIRS = [RUNS_DIR / "exp_a2_depth_seed42", RUNS_DIR / "exp_a2_depth_seed123"]

A4A_DIRS = [
    RUNS_DIR / "exp_a4a_synthetic_depth_seed_42",
    RUNS_DIR / "exp_a4a_synthetic_depth_seed_123",
]


def load_results_csv(exp_dir):
    """
    Load results.csv from experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        pd.DataFrame or None
    """
    results_file = exp_dir / "results.csv"

    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return None

    try:
        df = pd.read_csv(results_file)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def extract_final_metrics(exp_dirs, exp_name):
    """
    Extract final metrics from multiple runs.

    Args:
        exp_dirs: List of experiment directories
        exp_name: Experiment name

    Returns:
        dict: Metrics summary
    """
    metrics = []

    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} not found, skipping")
            continue

        df = load_results_csv(exp_dir)

        if df is None:
            continue

        # Get final row (last epoch)
        final = df.iloc[-1]

        metrics.append(
            {
                "exp_name": exp_name,
                "seed": exp_dir.stem.split("_")[-1],  # Extract seed from dir name
                "mAP50": final["metrics/mAP50(B)"]
                if "metrics/mAP50(B)" in final
                else None,
                "mAP50-95": final["metrics/mAP50-95(B)"]
                if "metrics/mAP50-95(B)" in final
                else None,
                "precision": final["metrics/precision(B)"]
                if "metrics/precision(B)" in final
                else None,
                "recall": final["metrics/recall(B)"]
                if "metrics/recall(B)" in final
                else None,
                "box_loss": final["train/box_loss"]
                if "train/box_loss" in final
                else None,
                "cls_loss": final["train/cls_loss"]
                if "train/cls_loss" in final
                else None,
            }
        )

    if not metrics:
        return None

    # Calculate mean and std
    df_metrics = pd.DataFrame(metrics)

    summary = {
        "exp_name": exp_name,
        "n_runs": len(metrics),
        "mAP50_mean": df_metrics["mAP50"].mean(),
        "mAP50_std": df_metrics["mAP50"].std(),
        "mAP50-95_mean": df_metrics["mAP50-95"].mean(),
        "mAP50-95_std": df_metrics["mAP50-95"].std(),
        "precision_mean": df_metrics["precision"].mean(),
        "recall_mean": df_metrics["recall"].mean(),
    }

    return summary


def plot_comparison(a2_summary, a4a_summary, output_dir):
    """
    Plot comparison between A.2 and A.4a.

    Args:
        a2_summary: A.2 metrics summary
        a4a_summary: A.4a metrics summary
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for plotting
    experiments = ["A.2\n(Real Depth)", "A.4a\n(Synthetic Depth)"]
    map50_means = [a2_summary["mAP50_mean"], a4a_summary["mAP50_mean"]]
    map50_stds = [a2_summary["mAP50_std"], a4a_summary["mAP50_std"]]
    map50_95_means = [a2_summary["mAP50-95_mean"], a4a_summary["mAP50-95_mean"]]
    map50_95_stds = [a2_summary["mAP50-95_std"], a4a_summary["mAP50-95_std"]]

    # Plot mAP comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # mAP50
    ax1.bar(
        experiments,
        map50_means,
        yerr=map50_stds,
        capsize=5,
        color=["#2ecc71", "#3498db"],
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_ylabel("mAP50", fontsize=12)
    ax1.set_title("mAP50 Comparison\n(Test Set)", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # Add values on bars
    for i, (mean, std) in enumerate(zip(map50_means, map50_stds)):
        ax1.text(
            i,
            mean + std + 0.02,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # mAP50-95
    ax2.bar(
        experiments,
        map50_95_means,
        yerr=map50_95_stds,
        capsize=5,
        color=["#2ecc71", "#3498db"],
        alpha=0.7,
        edgecolor="black",
    )
    ax2.set_ylabel("mAP50-95", fontsize=12)
    ax2.set_title("mAP50-95 Comparison\n(Test Set)", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, max(map50_95_means) * 1.3)
    ax2.grid(axis="y", alpha=0.3)

    # Add values on bars
    for i, (mean, std) in enumerate(zip(map50_95_means, map50_95_stds)):
        ax2.text(
            i,
            mean + std + 0.01,
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "map_comparison.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'map_comparison.png'}")
    plt.close()

    # Plot Precision-Recall comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    precision_means = [a2_summary["precision_mean"], a4a_summary["precision_mean"]]
    recall_means = [a2_summary["recall_mean"], a4a_summary["recall_mean"]]

    x = range(len(experiments))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        precision_means,
        width,
        label="Precision",
        color="#e74c3c",
        alpha=0.7,
        edgecolor="black",
    )
    ax.bar(
        [i + width / 2 for i in x],
        recall_means,
        width,
        label="Recall",
        color="#9b59b6",
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Precision & Recall Comparison\n(Test Set)", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "precision_recall_comparison.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {output_dir / 'precision_recall_comparison.png'}")
    plt.close()


def generate_report(a2_summary, a4a_summary, output_dir):
    """
    Generate text report.

    Args:
        a2_summary: A.2 metrics summary
        a4a_summary: A.4a metrics summary
        output_dir: Output directory
    """
    report_path = output_dir / "comparison_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Real Depth vs Synthetic Depth Comparison\n")
        f.write("=" * 60 + "\n\n")

        f.write("A.2 - Real Depth (RealSense Sensor)\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"  mAP50:     {a2_summary['mAP50_mean']:.3f} ± {a2_summary['mAP50_std']:.3f}\n"
        )
        f.write(
            f"  mAP50-95:  {a2_summary['mAP50-95_mean']:.3f} ± {a2_summary['mAP50-95_std']:.3f}\n"
        )
        f.write(f"  Precision: {a2_summary['precision_mean']:.3f}\n")
        f.write(f"  Recall:    {a2_summary['recall_mean']:.3f}\n")
        f.write(f"  Runs:      {a2_summary['n_runs']}\n\n")

        f.write("A.4a - Synthetic Depth (Depth-Anything-V2)\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"  mAP50:     {a4a_summary['mAP50_mean']:.3f} ± {a4a_summary['mAP50_std']:.3f}\n"
        )
        f.write(
            f"  mAP50-95:  {a4a_summary['mAP50-95_mean']:.3f} ± {a4a_summary['mAP50-95_std']:.3f}\n"
        )
        f.write(f"  Precision: {a4a_summary['precision_mean']:.3f}\n")
        f.write(f"  Recall:    {a4a_summary['recall_mean']:.3f}\n")
        f.write(f"  Runs:      {a4a_summary['n_runs']}\n\n")

        f.write("Performance Gap\n")
        f.write("-" * 60 + "\n")
        map50_gap = a2_summary["mAP50_mean"] - a4a_summary["mAP50_mean"]
        map50_95_gap = a2_summary["mAP50-95_mean"] - a4a_summary["mAP50-95_mean"]

        f.write(
            f"  mAP50 gap:     {map50_gap:+.3f} ({map50_gap / a2_summary['mAP50_mean'] * 100:+.1f}%)\n"
        )
        f.write(
            f"  mAP50-95 gap:  {map50_95_gap:+.3f} ({map50_95_gap / a2_summary['mAP50-95_mean'] * 100:+.1f}%)\n\n"
        )

        # Interpretation
        f.write("Interpretation\n")
        f.write("-" * 60 + "\n")

        if map50_95_gap > 0.05:
            f.write("  Real depth significantly outperforms synthetic depth.\n")
            f.write("  Recommendation: Use real depth sensor if available.\n")
        elif map50_95_gap > 0.02:
            f.write("  Real depth moderately better than synthetic depth.\n")
            f.write("  Recommendation: Real depth preferred, but synthetic viable.\n")
        else:
            f.write("  Synthetic depth performs comparably to real depth.\n")
            f.write("  Recommendation: Synthetic depth is a viable alternative!\n")

        percentage_retained = (
            a4a_summary["mAP50-95_mean"] / a2_summary["mAP50-95_mean"]
        ) * 100
        f.write(
            f"\n  Synthetic depth retains {percentage_retained:.1f}% of real depth performance.\n"
        )

    print(f"Report saved: {report_path}")


def main():
    print("=" * 60)
    print("Compare Real Depth (A.2) vs Synthetic Depth (A.4a)")
    print("=" * 60)

    # Extract metrics
    print("\nExtracting A.2 metrics...")
    a2_summary = extract_final_metrics(A2_DIRS, "A.2")

    print("\nExtracting A.4a metrics...")
    a4a_summary = extract_final_metrics(A4A_DIRS, "A.4a")

    if a2_summary is None:
        print("\n✗ A.2 results not found. Please train A.2 first.")
        return

    if a4a_summary is None:
        print("\n✗ A.4a results not found. Please train A.4a first.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating comparison plots...")
    plot_comparison(a2_summary, a4a_summary, OUTPUT_DIR)

    # Generate report
    print("\nGenerating comparison report...")
    generate_report(a2_summary, a4a_summary, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nFiles generated:")
    print("  - map_comparison.png")
    print("  - precision_recall_comparison.png")
    print("  - comparison_report.txt")

    print("\nQuick Summary:")
    print(
        f"  A.2 mAP50-95:  {a2_summary['mAP50-95_mean']:.3f} ± {a2_summary['mAP50-95_std']:.3f}"
    )
    print(
        f"  A.4a mAP50-95: {a4a_summary['mAP50-95_mean']:.3f} ± {a4a_summary['mAP50-95_std']:.3f}"
    )

    gap = a2_summary["mAP50-95_mean"] - a4a_summary["mAP50-95_mean"]
    print(f"  Gap: {gap:+.3f} ({gap / a2_summary['mAP50-95_mean'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
