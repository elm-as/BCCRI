"""Pure feature matrix construction and inside-fold categorical encoding."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.core.churn_config import FeaturePartitionConfig


def build_clean_feature_matrix(
    raw_df: pd.DataFrame,
    partition_config: FeaturePartitionConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the strictly leak-free 12-feature matrix and binary target series.

    Inputs:
        raw_df: Validated raw bank churn DataFrame.
        partition_config: Column role definitions specifying IDs, leakage, and clean inputs.

    Returns:
        ``(features_df, target_series)`` as defensive copies in deterministic 12-column order.

    Raises:
        KeyError: If any clean feature or target column is missing.
    """
    frame = raw_df.copy()
    clean_cols = list(partition_config.ordered_clean_columns())
    missing = [col for col in clean_cols + [partition_config.target_column] if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns for clean feature matrix: {missing}")
    features_df = frame[clean_cols].copy()
    target_series = frame[partition_config.target_column].astype(int).copy()
    return features_df, target_series


def build_ablation_feature_matrix(
    raw_df: pd.DataFrame,
    partition_config: FeaturePartitionConfig,
    extra_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct a feature matrix appending specific audit/leakage columns for ablation.

    Inputs:
        raw_df: Validated raw bank churn DataFrame.
        partition_config: Column partition config.
        extra_columns: Controlled audit columns (e.g., ``("Satisfaction Score",)`` or
            ``("Complain", "Satisfaction Score")``).

    Returns:
        ``(ablation_df, target_series)`` as defensive copies without mutating ``raw_df``.

    Raises:
        KeyError: If requested extra columns do not exist in ``raw_df``.
        ValueError: If any ID or target column is requested inside ``extra_columns``.
    """
    frame = raw_df.copy()
    forbidden = set(partition_config.id_columns) | {partition_config.target_column}
    invalid_extra = [col for col in extra_columns if col in forbidden]
    if invalid_extra:
        raise ValueError(f"ID or target columns cannot be added to ablation matrix: {invalid_extra}")
    clean_df, target_series = build_clean_feature_matrix(frame, partition_config)
    missing = [col for col in extra_columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing requested ablation columns: {missing}")
    ordered_extra = [col for col in extra_columns if col not in clean_df.columns]
    if not ordered_extra:
        return clean_df, target_series
    combined_df = pd.concat([clean_df, frame[ordered_extra].copy()], axis=1)
    return combined_df, target_series


def compute_domain_diagnostic_flags(
    clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pure banking domain diagnostic indicators for EDA and linear baselines.

    Inputs:
        clean_df: Clean 12-feature DataFrame (never mutated).

    Returns:
        New DataFrame with ``is_zero_balance``, ``balance_to_salary_ratio``, and
        ``has_multi_product_risk`` (``NumOfProducts >= 3``) appended.
    """
    enriched = clean_df.copy()
    enriched["is_zero_balance"] = (enriched["Balance"] == 0.0).astype(int)
    safe_salary = enriched["EstimatedSalary"].clip(lower=1.0)
    enriched["balance_to_salary_ratio"] = (enriched["Balance"] / safe_salary).round(6)
    enriched["has_multi_product_risk"] = (enriched["NumOfProducts"] >= 3).astype(int)
    return enriched


def encode_categorical_fold(
    train_val_pair: tuple[pd.DataFrame, pd.DataFrame],
    categorical_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit an OrdinalEncoder strictly on training fold and transform train and validation.

    Inputs:
        train_val_pair: Tuple ``(x_train, x_val)`` for one CV split.
        categorical_cols: Categorical column names to encode.

    Returns:
        New ``(x_train_encoded, x_val_encoded)`` copies preserving 1-to-1 column alignment.
    """
    x_train, x_val = train_val_pair
    train_enc = x_train.copy()
    val_enc = x_val.copy()
    active_cats = [col for col in categorical_cols if col in train_enc.columns]
    if not active_cats:
        return train_enc, val_enc
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=float,
    )
    train_enc[active_cats] = encoder.fit_transform(train_enc[active_cats].astype(str))
    val_enc[active_cats] = encoder.transform(val_enc[active_cats].astype(str))
    return train_enc, val_enc
