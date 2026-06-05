"""
Calibration Metrics: Expected Calibration Error (ECE)
======================================================
Computes ECE from model logits/probabilities and true labels.
Also plots a reliability diagram showing confidence vs accuracy.

Usage:
    python calibration.py

Or import and call directly:
    from calibration import compute_ece, plot_reliability_diagram
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# ── ECE Core ──────────────────────────────────────────────────────────────────
def compute_ece(
    logits_or_probs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
    verbose: bool = True,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        logits_or_probs : Shape [N, num_classes]. Raw logits OR softmax probs.
        labels          : Shape [N]. Integer ground-truth class indices.
        n_bins          : Number of equally-spaced confidence bins (default: 10).
        verbose         : If True, print bin-wise stats.

    Returns:
        ece (float): The ECE value in [0, 1]. Lower is better.
    """
    # --- Convert to numpy
    if isinstance(logits_or_probs, torch.Tensor):
        logits_or_probs = logits_or_probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # --- Softmax if needed (detect logits by checking if any row doesn't sum ~1)
    row_sums = logits_or_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        # Apply softmax manually
        exp = np.exp(logits_or_probs - logits_or_probs.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    else:
        probs = logits_or_probs

    confidences = probs.max(axis=1)      # max prob per sample
    predictions = probs.argmax(axis=1)   # predicted class
    correct     = (predictions == labels).astype(float)

    # --- Bin samples by confidence
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0
    n_samples = len(labels)

    if verbose:
        print(f"\n{'Bin':>10} {'N':>6} {'Conf':>8} {'Acc':>8} {'|Conf-Acc|':>12}")
        print("-" * 50)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)

        # Include the lowest boundary in the first bin
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        gap      = abs(avg_conf - avg_acc)
        ece     += (n_bin / n_samples) * gap

        if verbose:
            print(f"  ({lo:.1f},{hi:.1f}] {n_bin:>6}   {avg_conf:.4f}   {avg_acc:.4f}      {gap:.4f}")

    if verbose:
        print(f"\n  ECE = {ece:.4f}  ({ece * 100:.2f}%)\n")

    return ece


# ── Reliability Diagram ───────────────────────────────────────────────────────
def plot_reliability_diagram(
    logits_or_probs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
) -> None:
    """
    Plot a reliability diagram (confidence vs accuracy) with a gap histogram.

    Args:
        logits_or_probs : Shape [N, num_classes].
        labels          : Shape [N].
        n_bins          : Number of bins (default: 10).
        title           : Plot title.
        save_path       : If given, saves figure to this path instead of showing it.
    """
    if isinstance(logits_or_probs, torch.Tensor):
        logits_or_probs = logits_or_probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Softmax
    row_sums = logits_or_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        exp   = np.exp(logits_or_probs - logits_or_probs.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    else:
        probs = logits_or_probs

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_edges     = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs      = []
    bin_confs     = []
    bin_centers   = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
        bin_centers.append((lo + hi) / 2)

    ece = compute_ece(probs, labels, n_bins=n_bins, verbose=False)

    # --- Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    width = 1.0 / n_bins

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Accuracy bars
    ax.bar(bin_centers, bin_accs, width=width * 0.9, alpha=0.7,
           color="steelblue", label="Accuracy", align="center")

    # Gap bars (overconfidence / underconfidence)
    gap_vals = [c - a for c, a in zip(bin_confs, bin_accs)]
    ax.bar(bin_centers, gap_vals, width=width * 0.9, alpha=0.5,
           color="red", bottom=bin_accs, label="Gap", align="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Reliability diagram saved to: {save_path}")
    else:
        plt.show()


# ── Quick self-test with dummy data ───────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N, C = 500, 4

    # Simulate overconfident logits
    logits = np.random.randn(N, C) * 3.0
    labels = np.random.randint(0, C, size=N)

    print("=== ECE Calibration Test (random logits) ===")
    ece = compute_ece(logits, labels, n_bins=10, verbose=True)

    plot_reliability_diagram(logits, labels, n_bins=10,
                             title="Reliability Diagram (Random Logits)",
                             save_path="reliability_diagram.png")
