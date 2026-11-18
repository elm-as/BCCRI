"""Offline unit tests for schema validation, leakage audit, and feature purity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.churn_config import FeaturePartitionConfig, ValidationConfig
from src.core.churn_contracts import ChurnDatasetBundle
from src.features.churn_feature_pipeline import (
    build_ablation_feature_matrix,
    build_clean_feature_matrix,
    compute_domain_diagnostic_flags,
    encode_categorical_fold,
)
from src.features.leakage_auditor import audit_target_leakage, summarize_satisfaction_alignment
from src.ingestion.kaggle_churn_loader import has_valid_churn_schema, load_and_validate_churn_csv


def _make_synthetic_raw_df(n_rows: int = 80, seed: int = 7) -> pd.DataFrame:
    """Create a deterministic synthetic 18-column DataFrame matching Kaggle schema."""
    rng = np.random.default_rng(seed)
    exited = (rng.random(n_rows) < 0.25).astype(int)
    complain = exited.copy()
    complain[0] = 1 - exited[0]
    return pd.DataFrame(
        {
            "RowNumber": np.arange(1, n_rows + 1),
            "CustomerId": np.arange(15600001, 15600001 + n_rows),
            "Surname": [f"Client_{i}" for i in range(n_rows)],
            "CreditScore": rng.integers(450, 820, size=n_rows),
            "Geography": rng.choice(["France", "Germany", "Spain"], size=n_rows),
            "Gender": rng.choice(["Male", "Female"], size=n_rows),
            "Age": rng.integers(21, 72, size=n_rows),
            "Tenure": rng.integers(0, 11, size=n_rows),
            "Balance": np.where(rng.random(n_rows) < 0.35, 0.0, rng.uniform(5000, 180000, size=n_rows)),
            "NumOfProducts": rng.choice([1, 2, 3, 4], size=n_rows, p=[0.45, 0.45, 0.07, 0.03]),
            "HasCrCard": rng.choice([0, 1], size=n_rows),
            "IsActiveMember": rng.choice([0, 1], size=n_rows),
            "EstimatedSalary": rng.uniform(20000, 180000, size=n_rows),
            "Exited": exited,
            "Complain": complain,
            "Satisfaction Score": rng.integers(1, 6, size=n_rows),
            "Card Type": rng.choice(["DIAMOND", "GOLD", "SILVER", "PLATINUM"], size=n_rows),
            "Point Earned": rng.integers(150, 950, size=n_rows),
        }
    )


def test_schema_validation_rejects_missing_column(tmp_path: Path) -> None:
    """Dropping Balance fails schema validation and raises ValueError on load."""
    cfg = FeaturePartitionConfig()
    df = _make_synthetic_raw_df()
    assert has_valid_churn_schema(df, cfg) is True

    broken = df.drop(columns=["Balance"])
    assert has_valid_churn_schema(broken, cfg) is False

    bad_csv = tmp_path / "broken.csv"
    broken.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="schema"):
        load_and_validate_churn_csv(bad_csv, cfg)


def test_leakage_auditor_flags_complain_and_audits_satisfaction() -> None:
    """Near-unit Complain correlation is flagged while Satisfaction Score is audit-ablated."""
    df = _make_synthetic_raw_df(n_rows=100)
    bundle = ChurnDatasetBundle(
        raw_df=df,
        n_rows=len(df),
        n_cols=df.shape[1],
        churn_rate=float(df["Exited"].mean()),
        zero_balance_rate=float((df["Balance"] == 0.0).mean()),
    )
    rows = audit_target_leakage(bundle, FeaturePartitionConfig(), ValidationConfig())
    by_name = {row.feature_name: row for row in rows}
    assert by_name["Complain"].is_flagged_leakage is True
    assert by_name["Complain"].pearson_corr > 0.90
    assert by_name["Complain"].disposition == "MANDATORY_EXCLUDE_POST_EVENT_LEAKAGE"
    assert by_name["Satisfaction Score"].disposition == "AUDIT_ABLATED_EXCLUDE_PRIMARY"

    sat_summary = summarize_satisfaction_alignment(df, FeaturePartitionConfig())
    assert 1.0 <= sat_summary["mean_overall"] <= 5.0


def test_clean_feature_matrix_excludes_leakage_and_preserves_input() -> None:
    """Clean matrix keeps exact 12 pre-event columns without mutating raw DataFrame."""
    cfg = FeaturePartitionConfig()
    raw_df = _make_synthetic_raw_df()
    snapshot = raw_df.copy(deep=True)
    x_clean, y_clean = build_clean_feature_matrix(raw_df, cfg)

    pd.testing.assert_frame_equal(raw_df, snapshot)
    assert list(x_clean.columns) == list(cfg.ordered_clean_columns())
    assert x_clean.shape[1] == 12
    for forbidden in ("RowNumber", "CustomerId", "Surname", "Complain", "Satisfaction Score", "Exited"):
        assert forbidden not in x_clean.columns
    assert set(y_clean.unique()).issubset({0, 1})


def test_ablation_matrix_and_fold_encoder_behavior() -> None:
    """Controlled ablation appends audit columns cleanly and rejects ID/target injection."""
    cfg = FeaturePartitionConfig()
    raw_df = _make_synthetic_raw_df()
    x_sat, _ = build_ablation_feature_matrix(raw_df, cfg, ("Satisfaction Score",))
    x_leak, _ = build_ablation_feature_matrix(raw_df, cfg, ("Complain", "Satisfaction Score"))
    assert x_sat.shape[1] == 13
    assert x_leak.shape[1] == 14

    with pytest.raises(ValueError, match="ID or target"):
        build_ablation_feature_matrix(raw_df, cfg, ("CustomerId",))

    diag = compute_domain_diagnostic_flags(x_sat.iloc[:, :12])
    assert {"is_zero_balance", "balance_to_salary_ratio", "has_multi_product_risk"}.issubset(diag.columns)

    x_tr, x_va = encode_categorical_fold((x_sat.iloc[:60], x_sat.iloc[60:]), cfg.categorical_columns)
    assert list(x_tr.columns) == list(x_va.columns)
    assert pd.api.types.is_numeric_dtype(x_va["Geography"])
