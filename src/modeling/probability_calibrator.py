"""Out-of-fold probability calibration (Platt & Isotonic) and calibration metrics."""

from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.core.churn_config import ValidationConfig


def _validate_probability_vectors(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary labels and probability arrays without mutating inputs."""
    labels = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probs.ndim != 1 or len(labels) != len(probs):
        raise ValueError("y_true and y_prob must be 1D arrays of equal length.")
    if len(labels) == 0:
        raise ValueError("Cannot compute calibration metrics on empty arrays.")
    clipped = np.clip(probs, 0.0, 1.0)
    return labels, clipped


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute quantile-free uniform-bin Expected Calibration Error (ECE).

    Inputs:
        y_true: Binary target labels ``{0, 1}``.
        y_prob: Predicted positive-class probabilities in ``[0, 1]``.
        n_bins: Number of uniform probability intervals (default 10).

    Returns:
        Scalar ECE weighted by bin sample fraction.

    Raises:
        ValueError: When arrays are empty, unequal length, or ``n_bins < 2``.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    labels, probs = _validate_probability_vectors(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = float(len(labels))
    for idx in range(n_bins):
        low, high = edges[idx], edges[idx + 1]
        in_bin = (probs >= low) & (probs <= high if idx == n_bins - 1 else probs < high)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        bin_acc = float(np.mean(labels[in_bin]))
        bin_conf = float(np.mean(probs[in_bin]))
        ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute quantile reliability diagram coordinates ``(mean_pred, frac_pos)``.

    Inputs:
        y_true: Binary target labels.
        y_prob: Predicted probabilities.
        n_bins: Target quantile bin count.

    Returns:
        Tuple ``(prob_pred, prob_true)`` for plotting reliability curves.
    """
    labels, probs = _validate_probability_vectors(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def _fit_platt_oof(
    labels: np.ndarray,
    logits: np.ndarray,
    val_config: ValidationConfig,
) -> np.ndarray:
    """Generate strict out-of-fold Platt sigmoid calibrated probabilities."""
    splitter = StratifiedKFold(
        n_splits=val_config.n_splits,
        shuffle=True,
        random_state=val_config.random_seed,
    )
    calibrated = np.zeros(len(labels), dtype=float)
    for train_idx, val_idx in splitter.split(logits.reshape(-1, 1), labels):
        calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
        calibrator.fit(logits[train_idx].reshape(-1, 1), labels[train_idx])
        calibrated[val_idx] = calibrator.predict_proba(logits[val_idx].reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def _fit_isotonic_oof(
    labels: np.ndarray,
    probs: np.ndarray,
    val_config: ValidationConfig,
) -> np.ndarray:
    """Generate strict out-of-fold Isotonic calibrated probabilities."""
    splitter = StratifiedKFold(
        n_splits=val_config.n_splits,
        shuffle=True,
        random_state=val_config.random_seed,
    )
    calibrated = np.zeros(len(labels), dtype=float)
    for train_idx, val_idx in splitter.split(probs.reshape(-1, 1), labels):
        calibrator = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        calibrator.fit(probs[train_idx], labels[train_idx])
        calibrated[val_idx] = calibrator.transform(probs[val_idx])
    return np.clip(calibrated, 1e-6, 1.0 - 1e-6)


def calibrate_oof_probabilities(
    y_true: np.ndarray,
    raw_oof_probs: np.ndarray,
    val_config: ValidationConfig,
) -> tuple[np.ndarray, str, float, float]:
    """Compare OOF Platt vs Isotonic calibration and select lower-Brier probabilities.

    Inputs:
        y_true: Ground-truth binary labels.
        raw_oof_probs: Out-of-fold raw model probabilities.
        val_config: Cross-validation split and bin settings.

    Returns:
        ``(calibrated_probs, chosen_method, calibrated_brier, calibrated_ece)``.
    """
    labels, probs = _validate_probability_vectors(y_true, raw_oof_probs)
    safe_probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    platt_probs = _fit_platt_oof(labels, logit(safe_probs), val_config)
    iso_probs = _fit_isotonic_oof(labels, safe_probs, val_config)
    raw_brier = float(brier_score_loss(labels, probs))
    platt_brier = float(brier_score_loss(labels, platt_probs))
    iso_brier = float(brier_score_loss(labels, iso_probs))

    best_probs, method_name, best_brier = probs, "uncalibrated", raw_brier
    if platt_brier < best_brier:
        best_probs, method_name, best_brier = platt_probs, "platt_sigmoid", platt_brier
    if iso_brier < best_brier:
        best_probs, method_name, best_brier = iso_probs, "isotonic", iso_brier

    best_ece = compute_expected_calibration_error(labels, best_probs, val_config.calibration_bins)
    return best_probs, method_name, float(best_brier), float(best_ece)
