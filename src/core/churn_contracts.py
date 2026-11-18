"""Immutable domain value objects shared across BCCRI pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChurnDatasetBundle:
    """Validated raw customer churn dataset snapshot."""

    raw_df: pd.DataFrame
    n_rows: int
    n_cols: int
    churn_rate: float
    zero_balance_rate: float


@dataclass(frozen=True)
class LeakageAuditRow:
    """Univariate leakage diagnostic for one non-ID candidate feature."""

    feature_name: str
    pearson_corr: float
    mutual_info: float
    univariate_auc: float
    is_flagged_leakage: bool
    disposition: str


@dataclass(frozen=True)
class LeakageAblationRow:
    """Cross-validated metric comparison under controlled feature inclusion scenarios."""

    scenario_name: str
    model_name: str
    feature_count: int
    roc_auc: float
    pr_auc: float
    f1_score: float
    brier_score: float


@dataclass(frozen=True)
class ModelBenchmarkRow:
    """Stratified CV and OOF probability calibration metrics for one model family."""

    model_name: str
    fold_aucs: tuple[float, ...]
    oof_roc_auc: float
    oof_pr_auc: float
    oof_f1_default: float
    oof_f1_optimal: float
    optimal_threshold: float
    oof_brier: float
    oof_ece: float
    calibration_method: str
    calibrated_brier: float
    calibrated_ece: float
    oof_raw_probs: np.ndarray
    oof_calibrated_probs: np.ndarray


@dataclass(frozen=True)
class ShapLeverProfile:
    """Out-of-fold TreeExplainer attribution partitioned into actionable vs static drivers."""

    feature_names: tuple[str, ...]
    mean_abs_shap: dict[str, float]
    actionable_share: float
    static_share: float
    shap_matrix: np.ndarray
    top_actionable_levers: tuple[str, ...]


@dataclass(frozen=True)
class CrmSegmentSummary:
    """Aggregated risk, economic exposure, and recommended retention play for a CRM segment."""

    segment_name: str
    client_count: int
    client_share: float
    mean_churn_prob: float
    actual_churn_rate: float
    mean_clv_eur: float
    total_at_risk_clv_eur: float
    recommended_action: str


@dataclass(frozen=True)
class RetentionPolicyPoint:
    """Simulated retention campaign outcome at a given outreach budget fraction."""

    strategy_name: str
    outreach_fraction: float
    min_score_cutoff: float
    targeted_clients: int
    expected_saved_clients: float
    gross_clv_saved_eur: float
    campaign_cost_eur: float
    net_roi_eur: float
    roi_pct: float


@dataclass(frozen=True)
class SensitivityCell:
    """Break-even sensitivity grid point over save rate and offer cost multipliers."""

    base_save_rate: float
    offer_cost_eur: float
    optimal_net_roi_eur: float
    optimal_fraction: float


@dataclass(frozen=True)
class RetentionSimulationSummary:
    """Complete retention simulation outputs across strategies, budgets, and sensitivity."""

    optimal_ev_point: RetentionPolicyPoint
    optimal_risk_point: RetentionPolicyPoint
    random_at_optimal_fraction_point: RetentionPolicyPoint
    policy_curve: tuple[RetentionPolicyPoint, ...]
    sensitivity_grid: tuple[SensitivityCell, ...]
    segment_summaries: tuple[CrmSegmentSummary, ...]


@dataclass(frozen=True)
class DashboardPayload:
    """Single immutable payload passed to Matplotlib and Plotly visualization builders."""

    dataset_bundle: ChurnDatasetBundle
    leakage_audit_rows: tuple[LeakageAuditRow, ...]
    ablation_rows: tuple[LeakageAblationRow, ...]
    benchmark_rows: tuple[ModelBenchmarkRow, ...]
    shap_profile: ShapLeverProfile
    simulation_summary: RetentionSimulationSummary
    satisfaction_summary: dict[str, float]
    best_model_name: str
