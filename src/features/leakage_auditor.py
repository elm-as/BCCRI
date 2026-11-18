"""Automated target-leakage audit and post-event survey diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.core.churn_config import FeaturePartitionConfig, ValidationConfig
from src.core.churn_contracts import ChurnDatasetBundle, LeakageAuditRow


def _encode_series_for_audit(series: pd.Series) -> np.ndarray:
    """Convert numeric or categorical series into finite numeric array without mutating."""
    if pd.api.types.is_numeric_dtype(series):
        arr = series.to_numpy(dtype=float, na_value=np.nan, copy=True)
        finite_mask = np.isfinite(arr)
        if np.all(finite_mask):
            return arr
        fill_value = float(np.median(arr[finite_mask])) if np.any(finite_mask) else 0.0
        return np.where(finite_mask, arr, fill_value)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return codes.astype(float)


def _compute_univariate_cv_auc(
    values: np.ndarray,
    target: np.ndarray,
    val_config: ValidationConfig,
) -> float:
    """Compute 5-fold stratified univariate ROC-AUC using raw rank direction."""
    splitter = StratifiedKFold(
        n_splits=val_config.n_splits,
        shuffle=True,
        random_state=val_config.random_seed,
    )
    fold_aucs: list[float] = []
    for train_idx, val_idx in splitter.split(values, target):
        corr = float(np.corrcoef(values[train_idx], target[train_idx])[0, 1])
        sign = -1.0 if np.isfinite(corr) and corr < 0.0 else 1.0
        auc = float(roc_auc_score(target[val_idx], sign * values[val_idx]))
        fold_aucs.append(auc)
    return float(np.mean(fold_aucs))


def _classify_feature_disposition(
    feature_name: str,
    metrics_pair: tuple[float, float],
    configs_pair: tuple[FeaturePartitionConfig, ValidationConfig],
) -> tuple[bool, str]:
    """Assign leakage flag and disposition label from univariate diagnostics."""
    abs_corr, uni_auc = metrics_pair
    partition_config, val_config = configs_pair
    is_post_event_leak = (
        feature_name in partition_config.mandatory_leakage_columns
        or abs_corr >= val_config.leakage_corr_threshold
        or uni_auc >= val_config.leakage_auc_threshold
    )
    if is_post_event_leak:
        return True, "MANDATORY_EXCLUDE_POST_EVENT_LEAKAGE"
    if feature_name in partition_config.audit_candidate_columns:
        return False, "AUDIT_ABLATED_EXCLUDE_PRIMARY"
    if feature_name in partition_config.actionable_lever_columns:
        return False, "INCLUDE_ACTIONABLE_LEVER"
    return False, "INCLUDE_STATIC_CONTEXT"


def audit_target_leakage(
    bundle: ChurnDatasetBundle,
    partition_config: FeaturePartitionConfig,
    val_config: ValidationConfig,
) -> tuple[LeakageAuditRow, ...]:
    """Compute Pearson r, mutual information, and univariate AUC across candidate columns.

    Inputs:
        bundle: Validated churn dataset snapshot.
        partition_config: Column role definitions (IDs, leakage, audit, actionable, static).
        val_config: Correlation/AUC leakage thresholds and CV seed.

    Returns:
        Tuple of ``LeakageAuditRow`` sorted descending by absolute Pearson correlation.

    Edge Cases:
        Zero-variance features yield ``pearson_corr == 0.0`` and ``univariate_auc == 0.5``.
        Input DataFrame is never mutated.
    """
    df = bundle.raw_df.copy()
    target = df[partition_config.target_column].to_numpy(dtype=int)
    excluded = set(partition_config.id_columns) | {partition_config.target_column}
    candidates = [col for col in df.columns if col not in excluded]

    rows: list[LeakageAuditRow] = []
    for col in candidates:
        values = _encode_series_for_audit(df[col])
        std_val = float(np.std(values))
        is_constant = std_val == 0.0 or not np.isfinite(std_val)
        pearson_r = 0.0 if is_constant else float(np.corrcoef(values, target)[0, 1])
        mi_score = (
            0.0
            if is_constant
            else float(
                mutual_info_classif(
                    values.reshape(-1, 1),
                    target,
                    random_state=val_config.random_seed,
                )[0]
            )
        )
        uni_auc = 0.5 if is_constant else _compute_univariate_cv_auc(values, target, val_config)
        flagged, disposition = _classify_feature_disposition(
            col,
            (abs(pearson_r), uni_auc),
            (partition_config, val_config),
        )
        rows.append(
            LeakageAuditRow(
                feature_name=col,
                pearson_corr=pearson_r,
                mutual_info=mi_score,
                univariate_auc=uni_auc,
                is_flagged_leakage=flagged,
                disposition=disposition,
            )
        )
    rows.sort(key=lambda item: abs(item.pearson_corr), reverse=True)
    return tuple(rows)


def summarize_satisfaction_alignment(
    df: pd.DataFrame,
    partition_config: FeaturePartitionConfig,
) -> dict[str, float]:
    """Quantify distribution and churn-rate alignment of ``Satisfaction Score``.

    Inputs:
        df: Raw churn DataFrame (copied internally, never mutated).
        partition_config: Schema config providing target and complaint names.

    Returns:
        Dictionary with mean satisfaction overall, among churners vs retained,
        min/max churn rate across scores 1..5, and Pearson correlation with ``Exited``.
    """
    frame = df.copy()
    score_col = "Satisfaction Score"
    target_col = partition_config.target_column
    if score_col not in frame.columns or target_col not in frame.columns:
        raise KeyError("DataFrame missing Satisfaction Score or target column.")
    by_score = frame.groupby(score_col)[target_col].mean()
    corr_val = float(frame[score_col].corr(frame[target_col]))
    return {
        "mean_overall": float(frame[score_col].mean()),
        "mean_churned": float(frame.loc[frame[target_col] == 1, score_col].mean()),
        "mean_retained": float(frame.loc[frame[target_col] == 0, score_col].mean()),
        "min_score_churn_rate": float(by_score.min()),
        "max_score_churn_rate": float(by_score.max()),
        "pearson_r_exited": corr_val,
    }
