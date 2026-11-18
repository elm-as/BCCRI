"""Offline unit tests for OOF calibration, SHAP lever partition, CRM segments, and ROI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.churn_config import ChurnPipelineConfig, RetentionEconomicsConfig, ValidationConfig
from src.evaluation.retention_roi_simulator import (
    estimate_customer_clv_eur,
    simulate_retention_strategies,
)
from src.evaluation.shap_lever_segmenter import assign_crm_segments, compute_oof_shap_profile
from src.modeling.probability_calibrator import (
    calibrate_oof_probabilities,
    compute_expected_calibration_error,
)


def _make_clean_synthetic_cohort(n_rows: int = 120, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    """Build deterministic 12-feature clean cohort with realistic churn signal."""
    rng = np.random.default_rng(seed)
    is_active = rng.choice([0, 1], size=n_rows)
    num_prod = rng.choice([1, 2, 3], size=n_rows, p=[0.45, 0.45, 0.10])
    balance = np.where(rng.random(n_rows) < 0.35, 0.0, rng.uniform(25000.0, 200000.0, size=n_rows))
    age = rng.integers(22, 70, size=n_rows)
    logits = -1.3 + 0.04 * (age - 38) - 1.1 * is_active + 1.4 * (num_prod == 3)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((rng.random(n_rows) < probs).astype(int), name="Exited")
    x_df = pd.DataFrame(
        {
            "IsActiveMember": is_active,
            "NumOfProducts": num_prod,
            "Balance": balance,
            "HasCrCard": rng.choice([0, 1], size=n_rows),
            "Card Type": rng.choice(["DIAMOND", "GOLD", "SILVER", "PLATINUM"], size=n_rows),
            "Point Earned": rng.integers(200, 950, size=n_rows),
            "CreditScore": rng.integers(480, 820, size=n_rows),
            "Geography": rng.choice(["France", "Germany", "Spain"], size=n_rows),
            "Gender": rng.choice(["Male", "Female"], size=n_rows),
            "Age": age,
            "Tenure": rng.integers(0, 11, size=n_rows),
            "EstimatedSalary": rng.uniform(25000.0, 175000.0, size=n_rows),
        }
    )
    return x_df, y


def test_ece_and_oof_calibration_bounds() -> None:
    """Perfect calibration has zero ECE and calibrated OOF probabilities remain in [0, 1]."""
    y_perfect = np.array([0, 0, 1, 1], dtype=int)
    p_perfect = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)
    assert compute_expected_calibration_error(y_perfect, p_perfect, n_bins=5) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="equal length"):
        compute_expected_calibration_error(np.array([0, 1]), np.array([0.5]), n_bins=5)

    rng = np.random.default_rng(42)
    y_syn = rng.choice([0, 1], size=120, p=[0.75, 0.25])
    raw_p = np.clip(0.15 + 0.55 * y_syn + rng.normal(0.0, 0.15, size=120), 0.02, 0.98)
    cal_p, method, brier, ece = calibrate_oof_probabilities(y_syn, raw_p, ValidationConfig(n_splits=3))
    assert method in {"uncalibrated", "platt_sigmoid", "isotonic"}
    assert np.all((cal_p >= 0.0) & (cal_p <= 1.0))
    assert 0.0 <= brier <= 0.5
    assert 0.0 <= ece <= 1.0


def test_oof_shap_partition_sums_to_one_and_segments_assign_deterministically() -> None:
    """OOF SHAP matrix aligns 1-to-1 with 12 features and CRM segments map deterministically."""
    x_df, y_ser = _make_clean_synthetic_cohort(n_rows=120)
    cfg = ChurnPipelineConfig.create_default(Path("."))
    fast_cfg = ChurnPipelineConfig(
        paths=cfg.paths,
        partition=cfg.partition,
        validation=ValidationConfig(n_splits=3, random_seed=42),
        hyperparams=cfg.hyperparams,
        economics=cfg.economics,
    )
    shap_profile = compute_oof_shap_profile(x_df, y_ser, fast_cfg)
    assert shap_profile.shap_matrix.shape == (120, 12)
    assert shap_profile.actionable_share + shap_profile.static_share == pytest.approx(1.0)
    assert 0.0 < shap_profile.actionable_share < 1.0

    probs = np.where(np.arange(120) < 60, 0.15, 0.72)
    seg_df = assign_crm_segments(x_df, (probs, shap_profile), fast_cfg.validation)
    assert (seg_df.loc[seg_df["p_churn"] < 0.35, "crm_segment"] == "Healthy / Low-Risk Monitor").all()
    allowed_segments = {name for name, _, _ in cfg.economics.segment_economics}
    assert set(seg_df["crm_segment"].unique()).issubset(allowed_segments)


def test_clv_floor_and_roi_simulation_beats_random_and_avoids_zero_balance_trap() -> None:
    """EV-targeted outreach outperforms pure risk-ranking when zero-balance traps exist."""
    econ = RetentionEconomicsConfig()
    df_trap = pd.DataFrame(
        {
            "Balance": [0.0, 0.0, 180000.0, 180000.0, 15000.0, 15000.0],
            "EstimatedSalary": [30000.0, 30000.0, 140000.0, 140000.0, 50000.0, 50000.0],
            "p_churn": [0.92, 0.90, 0.78, 0.76, 0.10, 0.08],
            "y_true": [1, 1, 1, 1, 0, 0],
            "crm_segment": [
                "Structural Demographic Risk",
                "Structural Demographic Risk",
                "Dormancy Reactivation",
                "Dormancy Reactivation",
                "Healthy / Low-Risk Monitor",
                "Healthy / Low-Risk Monitor",
            ],
        }
    )
    clvs = estimate_customer_clv_eur(df_trap, econ)
    assert np.all(clvs >= econ.min_clv_eur)
    assert clvs[2] > clvs[0]

    summary = simulate_retention_strategies(df_trap, econ)
    assert summary.optimal_ev_point.net_roi_eur > summary.optimal_risk_point.net_roi_eur
    assert summary.optimal_ev_point.net_roi_eur > 0.0
    assert len(summary.sensitivity_grid) == 49
