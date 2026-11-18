"""Immutable configuration dataclasses for the BCCRI churn & retention pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChurnPathConfig:
    """Filesystem locations for raw data, outputs, visuals, and Kaggle config."""

    project_root: Path
    raw_csv_path: Path
    processed_dir: Path
    artifacts_dir: Path
    images_dir: Path
    dashboard_path: Path
    results_json_path: Path
    kaggle_config_dir: Path

    @classmethod
    def from_root(cls, root_dir: Path) -> "ChurnPathConfig":
        """Build default project paths from the BCCRI repository root."""
        resolved_root = root_dir.resolve()
        env_kaggle = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
        kaggle_dir = (
            Path(env_kaggle).resolve()
            if env_kaggle
            else (resolved_root.parent / "PCCF").resolve()
        )
        return cls(
            project_root=resolved_root,
            raw_csv_path=resolved_root / "data" / "raw" / "Customer-Churn-Records.csv",
            processed_dir=resolved_root / "data" / "processed",
            artifacts_dir=resolved_root / "artifacts",
            images_dir=resolved_root / "images",
            dashboard_path=resolved_root / "artifacts" / "dashboard.html",
            results_json_path=resolved_root / "artifacts" / "results.json",
            kaggle_config_dir=kaggle_dir,
        )


@dataclass(frozen=True)
class FeaturePartitionConfig:
    """Column partition rules separating target, IDs, leakage, levers, and static context."""

    target_column: str = "Exited"
    id_columns: tuple[str, ...] = ("RowNumber", "CustomerId", "Surname")
    mandatory_leakage_columns: tuple[str, ...] = ("Complain",)
    audit_candidate_columns: tuple[str, ...] = ("Satisfaction Score",)
    actionable_lever_columns: tuple[str, ...] = (
        "IsActiveMember",
        "NumOfProducts",
        "Balance",
        "HasCrCard",
        "Card Type",
        "Point Earned",
    )
    static_context_columns: tuple[str, ...] = (
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "EstimatedSalary",
    )
    categorical_columns: tuple[str, ...] = ("Geography", "Gender", "Card Type")

    def ordered_clean_columns(self) -> tuple[str, ...]:
        """Return deterministic 12-column pre-event modeling feature order."""
        return self.actionable_lever_columns + self.static_context_columns

    def expected_raw_columns(self) -> tuple[str, ...]:
        """Return all 18 expected raw CSV columns."""
        return (
            self.id_columns
            + self.static_context_columns
            + self.actionable_lever_columns
            + self.mandatory_leakage_columns
            + self.audit_candidate_columns
            + (self.target_column,)
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Stratified cross-validation, leakage audit, and calibration parameters."""

    n_splits: int = 5
    random_seed: int = 42
    calibration_bins: int = 10
    leakage_corr_threshold: float = 0.50
    leakage_auc_threshold: float = 0.90
    high_risk_threshold: float = 0.35


@dataclass(frozen=True)
class ModelHyperparamsConfig:
    """Regularized hyperparameters across the four benchmark model families."""

    lr_c: float = 1.0
    lr_max_iter: int = 1000
    lgbm_n_estimators: int = 350
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 15
    lgbm_max_depth: int = 4
    lgbm_min_child_samples: int = 40
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_alpha: float = 1.0
    lgbm_reg_lambda: float = 5.0
    xgb_n_estimators: int = 350
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 4
    xgb_min_child_weight: float = 10.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 1.0
    xgb_reg_lambda: float = 5.0
    cat_iterations: int = 500
    cat_learning_rate: float = 0.04
    cat_depth: int = 5
    cat_l2_leaf_reg: float = 6.0


@dataclass(frozen=True)
class RetentionEconomicsConfig:
    """Unit-economics assumptions for CLV estimation and retention ROI simulation."""

    nim_annual_rate: float = 0.015
    salary_fee_proxy_rate: float = 0.0025
    horizon_years: int = 2
    discount_rate: float = 0.08
    min_clv_eur: float = 250.0
    base_save_rate_anchor: float = 0.25
    base_offer_cost_eur: float = 75.0
    segment_economics: tuple[tuple[str, float, float], ...] = (
        ("Dormancy Reactivation", 0.28, 65.0),
        ("Product-Depth Reset", 0.22, 85.0),
        ("Deposit & Card Retention", 0.25, 95.0),
        ("Structural Demographic Risk", 0.08, 45.0),
        ("Healthy / Low-Risk Monitor", 0.05, 35.0),
    )
    outreach_fractions: tuple[float, ...] = tuple(
        round(0.05 * idx, 2) for idx in range(1, 21)
    )
    sensitivity_save_rates: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    sensitivity_offer_costs: tuple[float, ...] = (30.0, 45.0, 60.0, 75.0, 90.0, 120.0, 150.0)


@dataclass(frozen=True)
class ChurnPipelineConfig:
    """Master immutable configuration bundling all BCCRI domain settings."""

    paths: ChurnPathConfig
    partition: FeaturePartitionConfig
    validation: ValidationConfig
    hyperparams: ModelHyperparamsConfig
    economics: RetentionEconomicsConfig

    @classmethod
    def create_default(cls, root_dir: Path) -> "ChurnPipelineConfig":
        """Create the canonical production configuration for BCCRI."""
        return cls(
            paths=ChurnPathConfig.from_root(root_dir),
            partition=FeaturePartitionConfig(),
            validation=ValidationConfig(),
            hyperparams=ModelHyperparamsConfig(),
            economics=RetentionEconomicsConfig(),
        )
