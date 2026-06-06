"""Split conformal prediction intervals for regression."""

from __future__ import annotations

import numpy as np

from src.utils.logger import setup_logger

log = setup_logger(__name__)


def split_conformal(
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute conformal prediction intervals.

    Uses the calibration set (validation) residuals to set interval width.

    Parameters
    ----------
    y_cal : true values on calibration (validation) set.
    pred_cal : predictions on calibration set.
    pred_test : predictions on test set.
    alpha : miscoverage rate (0.10 => 90% coverage target).

    Returns
    -------
    lower : ndarray, lower bounds for test predictions.
    upper : ndarray, upper bounds for test predictions.
    q_hat : the conformal quantile used for the intervals.
    """
    residuals = np.abs(y_cal - pred_cal)
    n = len(residuals)

    # Quantile with finite-sample correction
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(residuals, q_level)

    lower = pred_test - q_hat
    upper = pred_test + q_hat

    log.info("Conformal: alpha=%.2f, q_hat=%.6f, interval_width=%.6f", alpha, q_hat, 2 * q_hat)

    return lower, upper, q_hat


def evaluate_coverage(
    y_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float = 0.90,
) -> dict:
    """Evaluate the empirical coverage of conformal intervals.

    Returns
    -------
    dict with coverage, avg_width, and whether coverage meets target.
    """
    covered = (y_test >= lower) & (y_test <= upper)
    coverage = covered.mean()
    avg_width = (upper - lower).mean()

    log.info(
        "Conformal coverage: %.3f (target %.3f), avg width: %.6f",
        coverage, target_coverage, avg_width,
    )

    return {
        "empirical_coverage": coverage,
        "target_coverage": target_coverage,
        "avg_interval_width": avg_width,
        "meets_target": coverage >= target_coverage,
    }
