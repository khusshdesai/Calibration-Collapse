"""
SATS: Synthetic-Aware Temperature Scaling
==========================================
A novel validation-free calibration method that predicts the optimal
Temperature parameter T directly from model logit statistics, without
requiring any labeled validation data.

Core Idea:
    Synthetic training data causes predictable logit inflation.
    The mean logit magnitude correlates linearly with the optimal T.
    Therefore: T_SATS = α × mean_logit_magnitude + β

Usage:
    from sats import SATSCalibrator
    sats = SATSCalibrator()
    sats.fit(logit_stats, optimal_temperatures)  # from known experiments
    T_predicted = sats.predict(new_logits)        # for unseen model
"""

import numpy as np
from typing import Dict, List, Tuple


class SATSCalibrator:
    """
    Synthetic-Aware Temperature Scaling.

    Learns a linear mapping from logit statistics to optimal temperature T,
    enabling validation-free calibration for models trained on synthetic data.
    """

    def __init__(self):
        self.alpha = None   # slope
        self.beta = None    # intercept
        self.r_squared = None
        self.is_fitted = False

    @staticmethod
    def compute_logit_stats(logits: np.ndarray) -> Dict[str, float]:
        """
        Extract calibration-relevant statistics from raw logits.

        Args:
            logits: Shape [N, num_classes]. Raw model logits (before softmax).

        Returns:
            Dictionary of logit statistics.
        """
        # Apply softmax for confidence stats
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        confidences = probs.max(axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        return {
            "mean_logit_magnitude": float(np.mean(np.abs(logits))),
            "max_logit_magnitude": float(np.mean(np.max(np.abs(logits), axis=1))),
            "logit_std": float(np.std(logits)),
            "logit_range": float(np.mean(np.max(logits, axis=1) - np.min(logits, axis=1))),
            "mean_confidence": float(np.mean(confidences)),
            "mean_entropy": float(np.mean(entropy)),
        }

    def fit(
        self,
        logit_stats_list: List[Dict[str, float]],
        optimal_temperatures: List[float],
        feature_key: str = "mean_logit_magnitude",
    ) -> Dict[str, float]:
        """
        Fit the SATS linear mapping: T = α × feature + β

        Args:
            logit_stats_list: List of stat dicts from compute_logit_stats()
            optimal_temperatures: Corresponding optimal T values from standard TS
            feature_key: Which logit statistic to use as the predictor

        Returns:
            Dictionary with alpha, beta, and r_squared
        """
        X = np.array([s[feature_key] for s in logit_stats_list])
        Y = np.array(optimal_temperatures)

        # Simple linear regression: Y = αX + β
        n = len(X)
        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        numerator = np.sum((X - x_mean) * (Y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        self.alpha = float(numerator / denominator) if denominator > 0 else 0.0
        self.beta = float(y_mean - self.alpha * x_mean)

        # R-squared
        y_pred = self.alpha * X + self.beta
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - y_mean) ** 2)
        self.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.is_fitted = True
        self.feature_key = feature_key

        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "r_squared": self.r_squared,
            "feature_used": feature_key,
        }

    def predict_temperature(self, logits: np.ndarray) -> float:
        """
        Predict optimal T from raw logits without any labeled validation data.

        Args:
            logits: Shape [N, num_classes]. Raw model logits.

        Returns:
            Predicted optimal temperature T.
        """
        if not self.is_fitted:
            raise RuntimeError("SATS not fitted. Call fit() first.")

        stats = self.compute_logit_stats(logits)
        feature_value = stats[self.feature_key]
        T_predicted = self.alpha * feature_value + self.beta

        # Clamp to reasonable range
        T_predicted = max(0.5, min(T_predicted, 5.0))

        return float(T_predicted)

    def evaluate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        T_standard: float,
    ) -> Dict[str, float]:
        """
        Compare SATS-predicted T against standard Temperature Scaling T.

        Args:
            logits: Raw test logits
            labels: True labels
            T_standard: Optimal T from standard Temperature Scaling
        """
        from calibration import compute_ece

        T_sats = self.predict_temperature(logits)

        # ECE with standard TS
        scaled_standard = logits / T_standard
        ece_standard = compute_ece(scaled_standard, labels, n_bins=10, verbose=False)

        # ECE with SATS
        scaled_sats = logits / T_sats
        ece_sats = compute_ece(scaled_sats, labels, n_bins=10, verbose=False)

        # ECE uncalibrated
        ece_uncalibrated = compute_ece(logits, labels, n_bins=10, verbose=False)

        return {
            "T_standard": T_standard,
            "T_sats": T_sats,
            "T_difference": abs(T_standard - T_sats),
            "ece_uncalibrated": ece_uncalibrated,
            "ece_standard_ts": ece_standard,
            "ece_sats": ece_sats,
            "ece_improvement_standard": ece_uncalibrated - ece_standard,
            "ece_improvement_sats": ece_uncalibrated - ece_sats,
        }

    def summary(self) -> str:
        """Print a formatted summary of the fitted SATS model."""
        if not self.is_fitted:
            return "SATS model not fitted yet."

        return (
            f"\n{'='*50}\n"
            f"  SATS Model Summary\n"
            f"{'='*50}\n"
            f"  Formula: T = {self.alpha:.4f} × {self.feature_key} + {self.beta:.4f}\n"
            f"  R² = {self.r_squared:.4f}\n"
            f"{'='*50}\n"
        )
