"""Out-of-fold TreeExplainer attribution and actionable CRM segment assignment."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.core.churn_config import ChurnPipelineConfig, FeaturePartitionConfig, ValidationConfig
from src.core.churn_contracts import ShapLeverProfile
from src.features.churn_feature_pipeline import encode_categorical_fold

SEGMENT_ACTIONS = MappingProxyType(
    {
        "Dormancy Reactivation": "Omnichannel engagement nudge & loyalty point bonus",
        "Product-Depth Reset": "Portfolio consolidation review to 2-product sweet spot",
        "Deposit & Card Retention": "Targeted deposit yield match or credit-card fee waiver",
        "Structural Demographic Risk": "Low-cost automated digital self-service retention",
        "Healthy / Low-Risk Monitor": "Passive monitoring (no costly incentive)",
    }
)


def _normalize_binary_shap_matrix(raw_shap: object) -> np.ndarray:
    """Normalize shap.TreeExplainer outputs across versions into (N, F) positive-class matrix."""
    if isinstance(raw_shap, list):
        return np.asarray(raw_shap[1], dtype=float)
    arr = np.asarray(raw_shap, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _select_top_actionable_lever(
    row_shap: np.ndarray,
    actionable_indices: tuple[int, ...],
    feature_names: tuple[str, ...],
) -> str:
    """Identify the dominant actionable churn driver for a single customer row."""
    actionable_vals = row_shap[list(actionable_indices)]
    best_pos_idx = int(np.argmax(actionable_vals))
    if actionable_vals[best_pos_idx] > 0.0:
        return feature_names[actionable_indices[best_pos_idx]]
    best_abs_idx = int(np.argmax(np.abs(actionable_vals)))
    return feature_names[actionable_indices[best_abs_idx]]


def _classify_single_client_segment(
    client_metrics: tuple[float, float, float],
    top_lever: str,
    high_risk_threshold: float,
) -> str:
    """Apply deterministic CRM segmentation rules for one client."""
    p_churn, act_abs, stat_abs = client_metrics
    if p_churn < high_risk_threshold:
        return "Healthy / Low-Risk Monitor"
    if stat_abs > 1.5 * act_abs:
        return "Structural Demographic Risk"
    if top_lever in ("IsActiveMember", "Point Earned"):
        return "Dormancy Reactivation"
    if top_lever == "NumOfProducts":
        return "Product-Depth Reset"
    return "Deposit & Card Retention"


def _resolve_recommended_action(segment: str, has_credit_card: int) -> str:
    """Tailor card-tier action when HasCrCard == 0 to avoid synthetic Card Type artifact."""
    if segment != "Deposit & Card Retention":
        return SEGMENT_ACTIONS[segment]
    if has_credit_card == 1:
        return "Targeted deposit yield match & credit-card tier fee waiver"
    return "Targeted deposit yield match & zero-fee credit-card onboarding"


def compute_oof_shap_profile(
    features_df: pd.DataFrame,
    target: pd.Series,
    pipeline_config: ChurnPipelineConfig,
) -> ShapLeverProfile:
    """Compute strictly out-of-fold TreeExplainer SHAP partitioned into actionable vs static.

    Inputs:
        features_df: Clean 12-feature matrix (copied internally).
        target: Binary ``Exited`` target series.
        pipeline_config: Master configuration with CV splits and lever partitions.

    Returns:
        Frozen ``ShapLeverProfile`` where ``actionable_share + static_share == 1.0``.

    Edge Cases:
        Normalizes ``shap==0.49.1`` list/3D/2D outputs and handles zero-total-SHAP safely.
    """
    x_df = features_df.copy().reset_index(drop=True)
    y_arr = target.to_numpy(dtype=int)
    val_cfg = pipeline_config.validation
    part_cfg = pipeline_config.partition
    hp = pipeline_config.hyperparams
    splitter = StratifiedKFold(n_splits=val_cfg.n_splits, shuffle=True, random_state=val_cfg.random_seed)
    cat_cols = tuple(c for c in part_cfg.categorical_columns if c in x_df.columns)
    shap_matrix = np.zeros(x_df.shape, dtype=float)

    for train_idx, val_idx in splitter.split(x_df, y_arr):
        x_tr, x_va = encode_categorical_fold((x_df.iloc[train_idx], x_df.iloc[val_idx]), cat_cols)
        clf = LGBMClassifier(
            n_estimators=hp.lgbm_n_estimators,
            learning_rate=hp.lgbm_learning_rate,
            num_leaves=hp.lgbm_num_leaves,
            max_depth=hp.lgbm_max_depth,
            min_child_samples=hp.lgbm_min_child_samples,
            subsample=hp.lgbm_subsample,
            colsample_bytree=hp.lgbm_colsample_bytree,
            reg_alpha=hp.lgbm_reg_alpha,
            reg_lambda=hp.lgbm_reg_lambda,
            random_state=val_cfg.random_seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(x_tr, y_arr[train_idx])
        explainer = shap.TreeExplainer(clf)
        shap_matrix[val_idx] = _normalize_binary_shap_matrix(explainer.shap_values(x_va))

    feature_names = tuple(x_df.columns.tolist())
    mean_abs_vals = np.mean(np.abs(shap_matrix), axis=0)
    mean_abs_map = {name: float(val) for name, val in zip(feature_names, mean_abs_vals)}
    act_set = set(part_cfg.actionable_lever_columns)
    act_indices = tuple(i for i, name in enumerate(feature_names) if name in act_set)
    act_sum = float(sum(val for name, val in mean_abs_map.items() if name in act_set))
    stat_sum = float(sum(val for name, val in mean_abs_map.items() if name not in act_set))
    total = act_sum + stat_sum
    act_share = 0.5 if total == 0.0 else act_sum / total
    stat_share = 1.0 - act_share
    top_levers = tuple(
        _select_top_actionable_lever(shap_matrix[i], act_indices, feature_names)
        for i in range(len(x_df))
    )
    return ShapLeverProfile(
        feature_names=feature_names,
        mean_abs_shap=mean_abs_map,
        actionable_share=float(act_share),
        static_share=float(stat_share),
        shap_matrix=shap_matrix,
        top_actionable_levers=top_levers,
    )


def assign_crm_segments(
    features_df: pd.DataFrame,
    model_outputs: tuple[np.ndarray, ShapLeverProfile],
    val_config: ValidationConfig,
) -> pd.DataFrame:
    """Assign each customer to a deterministic actionable CRM retention segment.

    Inputs:
        features_df: Clean 12-feature DataFrame (copied internally).
        model_outputs: Tuple ``(calibrated_probs, shap_profile)``.
        val_config: Validation thresholds including ``high_risk_threshold``.

    Returns:
        New DataFrame augmented with ``p_churn``, ``actionable_shap_abs``,
        ``static_shap_abs``, ``top_actionable_lever``, ``crm_segment``, and
        ``recommended_action``.
    """
    calibrated_probs, shap_profile = model_outputs
    probs = np.asarray(calibrated_probs, dtype=float)
    if len(probs) != len(features_df):
        raise ValueError("calibrated_probs length must match features_df row count.")

    out_df = features_df.copy().reset_index(drop=True)
    actionable_cols = frozenset(FeaturePartitionConfig().actionable_lever_columns)
    act_mask = np.array(
        [name in actionable_cols for name in shap_profile.feature_names],
        dtype=bool,
    )
    abs_shap = np.abs(shap_profile.shap_matrix)
    act_abs = np.sum(abs_shap[:, act_mask], axis=1)
    stat_abs = np.sum(abs_shap[:, ~act_mask], axis=1)

    segments = [
        _classify_single_client_segment(
            (float(probs[i]), float(act_abs[i]), float(stat_abs[i])),
            shap_profile.top_actionable_levers[i],
            val_config.high_risk_threshold,
        )
        for i in range(len(out_df))
    ]
    has_card = out_df["HasCrCard"].to_numpy(dtype=int)
    actions = [_resolve_recommended_action(segments[i], int(has_card[i])) for i in range(len(out_df))]

    out_df["p_churn"] = probs
    out_df["actionable_shap_abs"] = act_abs
    out_df["static_shap_abs"] = stat_abs
    out_df["top_actionable_lever"] = list(shap_profile.top_actionable_levers)
    out_df["crm_segment"] = segments
    out_df["recommended_action"] = actions
    return out_df
