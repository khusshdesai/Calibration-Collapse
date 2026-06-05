"""
SATS Analysis Pipeline
=======================
End-to-end script that:
1. Loads saved logits from DistilBERT and RoBERTa experiments
2. Computes logit statistics for each model/ratio
3. Fits SATS on DistilBERT data (5 points)
4. Validates SATS on RoBERTa data (cross-architecture generalization)
5. Generates comparison tables, scatter plots, and saves results

Run AFTER both experiment.py and experiment_roberta.py have completed.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sats import SATSCalibrator
from calibration import compute_ece


# ── Config ────────────────────────────────────────────────────────────────────
LOGITS_DIR      = r"C:\Users\Lenovo\Downloads\gen ai\results\logits"
RESULTS_DIR     = r"C:\Users\Lenovo\Downloads\gen ai\results"
FIGURES_DIR     = r"C:\Users\Lenovo\Downloads\gen ai\results\figures"

DISTILBERT_CSV  = os.path.join(RESULTS_DIR, "experiment_results.csv")
ROBERTA_CSV     = os.path.join(RESULTS_DIR, "roberta_experiment_results.csv")
SATS_CSV        = os.path.join(RESULTS_DIR, "sats_results.csv")

RATIOS = [0, 25, 50, 75, 100]
MODELS = {
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa",
}


def load_logits(model_key: str, ratio: int) -> dict:
    """Load saved logits and labels for a given model and synthetic ratio."""
    test_logits = np.load(os.path.join(LOGITS_DIR, f"{model_key}_{ratio}_test_logits.npy"))
    test_labels = np.load(os.path.join(LOGITS_DIR, f"{model_key}_{ratio}_test_labels.npy"))
    val_logits  = np.load(os.path.join(LOGITS_DIR, f"{model_key}_{ratio}_val_logits.npy"))
    val_labels  = np.load(os.path.join(LOGITS_DIR, f"{model_key}_{ratio}_val_labels.npy"))
    return {
        "test_logits": test_logits,
        "test_labels": test_labels,
        "val_logits": val_logits,
        "val_labels": val_labels,
    }


def main():
    print(f"\n{'='*60}")
    print("  SATS Analysis Pipeline")
    print(f"{'='*60}\n")

    # ── Step 1: Load existing results ─────────────────────────────────────
    distilbert_df = pd.read_csv(DISTILBERT_CSV)
    roberta_df = pd.read_csv(ROBERTA_CSV)

    print("Loaded experiment results:")
    print(f"  DistilBERT: {len(distilbert_df)} experiments")
    print(f"  RoBERTa:    {len(roberta_df)} experiments\n")

    # ── Step 2: Compute logit statistics for all experiments ──────────────
    all_stats = []

    for model_key, model_name in MODELS.items():
        csv_df = distilbert_df if model_key == "distilbert" else roberta_df

        for i, ratio in enumerate(RATIOS):
            print(f"  Computing stats for {model_name} @ {ratio}% synthetic...")
            data = load_logits(model_key, ratio)
            stats = SATSCalibrator.compute_logit_stats(data["test_logits"])

            stats["model"] = model_name
            stats["ratio"] = ratio
            stats["optimal_T"] = float(csv_df.iloc[i]["temperature"])
            stats["ece_before"] = float(csv_df.iloc[i]["ece_before"])
            stats["accuracy"] = float(csv_df.iloc[i]["accuracy"])

            all_stats.append(stats)

    stats_df = pd.DataFrame(all_stats)
    print(f"\n  Logit statistics computed for {len(stats_df)} experiments.\n")

    # ── Step 3: Fit SATS on DistilBERT ────────────────────────────────────
    print(f"{'='*60}")
    print("  Fitting SATS on DistilBERT data...")
    print(f"{'='*60}\n")

    sats = SATSCalibrator()

    db_stats = [s for s in all_stats if s["model"] == "DistilBERT"]
    db_temps = [s["optimal_T"] for s in db_stats]

    fit_result = sats.fit(db_stats, db_temps, feature_key="mean_logit_magnitude")
    print(sats.summary())
    print(f"  α = {fit_result['alpha']:.6f}")
    print(f"  β = {fit_result['beta']:.6f}")
    print(f"  R² = {fit_result['r_squared']:.4f}\n")

    # ── Step 4: Evaluate SATS on ALL experiments ──────────────────────────
    print(f"{'='*60}")
    print("  Evaluating SATS across all experiments...")
    print(f"{'='*60}\n")

    sats_results = []

    for model_key, model_name in MODELS.items():
        csv_df = distilbert_df if model_key == "distilbert" else roberta_df

        for i, ratio in enumerate(RATIOS):
            data = load_logits(model_key, ratio)
            T_standard = float(csv_df.iloc[i]["temperature"])

            eval_result = sats.evaluate(
                logits=data["test_logits"],
                labels=data["test_labels"],
                T_standard=T_standard,
            )
            eval_result["model"] = model_name
            eval_result["ratio"] = f"{ratio}%"
            sats_results.append(eval_result)

            print(f"  {model_name} @ {ratio}%: "
                  f"T_std={eval_result['T_standard']:.3f} | "
                  f"T_sats={eval_result['T_sats']:.3f} | "
                  f"ECE_std={eval_result['ece_standard_ts']:.4f} | "
                  f"ECE_sats={eval_result['ece_sats']:.4f}")

    sats_df = pd.DataFrame(sats_results)
    sats_df.to_csv(SATS_CSV, index=False)
    print(f"\n  SATS results saved to: {SATS_CSV}\n")

    # ── Step 5: Generate SATS Scatter Plot ────────────────────────────────
    print("  Generating SATS scatter plot...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot DistilBERT points (used for fitting)
    db_x = [s["mean_logit_magnitude"] for s in all_stats if s["model"] == "DistilBERT"]
    db_y = [s["optimal_T"] for s in all_stats if s["model"] == "DistilBERT"]
    ax.scatter(db_x, db_y, c="steelblue", s=120, zorder=5,
               label="DistilBERT (fit data)", edgecolors="black", linewidth=1)

    # Plot RoBERTa points (validation)
    rb_x = [s["mean_logit_magnitude"] for s in all_stats if s["model"] == "RoBERTa"]
    rb_y = [s["optimal_T"] for s in all_stats if s["model"] == "RoBERTa"]
    ax.scatter(rb_x, rb_y, c="coral", s=120, zorder=5, marker="^",
               label="RoBERTa (validation)", edgecolors="black", linewidth=1)

    # Plot SATS fit line
    all_x = db_x + rb_x
    x_range = np.linspace(min(all_x) * 0.9, max(all_x) * 1.1, 100)
    y_line = sats.alpha * x_range + sats.beta
    ax.plot(x_range, y_line, "k--", linewidth=2, alpha=0.7,
            label=f"SATS: T = {sats.alpha:.3f}×μ + {sats.beta:.3f} (R²={sats.r_squared:.3f})")

    # Annotate ratios
    for s in all_stats:
        offset = (5, 5) if s["model"] == "DistilBERT" else (5, -15)
        ax.annotate(f'{s["ratio"]}%', (s["mean_logit_magnitude"], s["optimal_T"]),
                    textcoords="offset points", xytext=offset, fontsize=9, alpha=0.8)

    ax.set_xlabel("Mean Logit Magnitude (μ_z)", fontsize=13)
    ax.set_ylabel("Optimal Temperature T", fontsize=13)
    ax.set_title("SATS: Logit Magnitude vs Optimal Temperature\nAcross Models and Synthetic Ratios", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    scatter_path = os.path.join(FIGURES_DIR, "sats_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"  Scatter plot saved to: {scatter_path}")

    # ── Step 6: Generate ECE Comparison Bar Chart ─────────────────────────
    print("  Generating ECE comparison chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (model_key, model_name) in enumerate(MODELS.items()):
        model_results = [r for r in sats_results if r["model"] == model_name]
        ratios_str = [r["ratio"] for r in model_results]
        ece_uncal = [r["ece_uncalibrated"] * 100 for r in model_results]
        ece_std = [r["ece_standard_ts"] * 100 for r in model_results]
        ece_sats_vals = [r["ece_sats"] * 100 for r in model_results]

        x = np.arange(len(ratios_str))
        width = 0.25

        axes[idx].bar(x - width, ece_uncal, width, label="Uncalibrated", color="salmon", alpha=0.8)
        axes[idx].bar(x, ece_std, width, label="Standard TS", color="steelblue", alpha=0.8)
        axes[idx].bar(x + width, ece_sats_vals, width, label="SATS (ours)", color="seagreen", alpha=0.8)

        axes[idx].set_xlabel("Synthetic Ratio", fontsize=12)
        axes[idx].set_ylabel("ECE (%)", fontsize=12)
        axes[idx].set_title(f"{model_name}", fontsize=14)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(ratios_str)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.suptitle("ECE Comparison: Uncalibrated vs Standard TS vs SATS", fontsize=15, fontweight="bold")
    plt.tight_layout()

    bar_path = os.path.join(FIGURES_DIR, "sats_ece_comparison.png")
    plt.savefig(bar_path, dpi=150)
    print(f"  ECE comparison chart saved to: {bar_path}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SATS ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"\n  SATS Formula: T = {sats.alpha:.4f} × mean_logit_magnitude + {sats.beta:.4f}")
    print(f"  R² = {sats.r_squared:.4f}")
    print(f"\n  Mean |T_standard - T_sats| across all experiments:")

    for model_name in MODELS.values():
        model_results = [r for r in sats_results if r["model"] == model_name]
        mean_diff = np.mean([r["T_difference"] for r in model_results])
        mean_ece_diff = np.mean([abs(r["ece_standard_ts"] - r["ece_sats"]) for r in model_results])
        print(f"    {model_name}: ΔT = {mean_diff:.4f}, ΔECE = {mean_ece_diff:.4f}")

    print(f"\n  Files saved:")
    print(f"    {SATS_CSV}")
    print(f"    {scatter_path}")
    print(f"    {bar_path}")


if __name__ == "__main__":
    main()
