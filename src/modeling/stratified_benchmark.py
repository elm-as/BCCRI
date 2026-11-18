"""Stratified K-Fold benchmark across 4 model families and 3-way leakage ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.core.churn_config import ChurnPipelineConfig, ModelHyperparamsConfig
from src.core.churn_contracts import ChurnDatasetBundle, LeakageAblationRow, ModelBenchmarkRow
from src.features.churn_feature_pipeline import (
    build_ablation_feature_matrix,
    compute_domain_diagnostic_flags,
    encode_categorical_fold,
)
from src.modeling.probability_calibrator import (
    calibrate_oof_probabilities,
    compute_expected_calibration_error,
)

BENCHMARK_MODELS: tuple[str, ...] = ("LogisticRegression", "LightGBM", "XGBoost", "CatBoost")


def _fit_predict_linear_fold(
    train_pair: tuple[pd.DataFrame, np.ndarray],
    x_val: pd.DataFrame,
    pipeline_config: ChurnPipelineConfig,
) -> np.ndarray:
    """Fit regularized LogisticRegression with inside-fold domain flags and OneHotEncoder."""
    x_train, y_train = train_pair
    train_diag = compute_domain_diagnostic_flags(x_train)
    val_diag = compute_domain_diagnostic_flags(x_val)
    cat_cols = [
        c for c in (*pipeline_config.partition.categorical_columns, "NumOfProducts")
        if c in train_diag.columns
    ]
    num_cols = [c for c in train_diag.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    hp = pipeline_config.hyperparams
    clf = LogisticRegression(
        C=hp.lr_c,
        max_iter=hp.lr_max_iter,
        random_state=pipeline_config.validation.random_seed,
    )
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(train_diag, y_train)
    return pipe.predict_proba(val_diag)[:, 1]


def _build_gbdt_estimator(model_name: str, hp: ModelHyperparamsConfig, seed: int) -> object:
    """Instantiate LightGBM or XGBoost classifier from frozen hyperparameters."""
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=hp.lgbm_n_estimators,
            learning_rate=hp.lgbm_learning_rate,
            num_leaves=hp.lgbm_num_leaves,
            max_depth=hp.lgbm_max_depth,
            min_child_samples=hp.lgbm_min_child_samples,
            subsample=hp.lgbm_subsample,
            colsample_bytree=hp.lgbm_colsample_bytree,
            reg_alpha=hp.lgbm_reg_alpha,
            reg_lambda=hp.lgbm_reg_lambda,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    return XGBClassifier(
        n_estimators=hp.xgb_n_estimators,
        learning_rate=hp.xgb_learning_rate,
        max_depth=hp.xgb_max_depth,
        min_child_weight=hp.xgb_min_child_weight,
        subsample=hp.xgb_subsample,
        colsample_bytree=hp.xgb_colsample_bytree,
        reg_alpha=hp.xgb_reg_alpha,
        reg_lambda=hp.xgb_reg_lambda,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )


def _fit_predict_gbdt_fold(
    model_name: str,
    fold_matrices: tuple[pd.DataFrame, pd.DataFrame, np.ndarray],
    pipeline_config: ChurnPipelineConfig,
) -> np.ndarray:
    """Fit LightGBM, XGBoost, or CatBoost on one CV split without leaking fold encoders."""
    x_train, x_val, y_train = fold_matrices
    cat_cols = tuple(c for c in pipeline_config.partition.categorical_columns if c in x_train.columns)
    hp = pipeline_config.hyperparams
    seed = pipeline_config.validation.random_seed
    if model_name == "CatBoost":
        train_cb, val_cb = x_train.copy(), x_val.copy()
        for col in cat_cols:
            train_cb[col] = train_cb[col].astype(str)
            val_cb[col] = val_cb[col].astype(str)
        clf = CatBoostClassifier(
            iterations=hp.cat_iterations,
            learning_rate=hp.cat_learning_rate,
            depth=hp.cat_depth,
            l2_leaf_reg=hp.cat_l2_leaf_reg,
            cat_features=list(cat_cols),
            random_seed=seed,
            thread_count=1,
            verbose=0,
            allow_writing_files=False,
        )
        clf.fit(train_cb, y_train)
        return clf.predict_proba(val_cb)[:, 1]

    x_train_enc, x_val_enc = encode_categorical_fold((x_train, x_val), cat_cols)
    clf = _build_gbdt_estimator(model_name, hp, seed)
    clf.fit(x_train_enc, y_train)
    return clf.predict_proba(x_val_enc)[:, 1]


def _find_optimal_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Search probability grid in [0.10, 0.85] for maximum F1 score."""
    best_thr, best_f1 = 0.50, float(f1_score(y_true, (probs >= 0.50).astype(int)))
    for thr in np.linspace(0.10, 0.85, 76):
        score = float(f1_score(y_true, (probs >= thr).astype(int)))
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)
    return float(best_thr), float(best_f1)


def _evaluate_single_model_cv(
    model_name: str,
    data_pair: tuple[pd.DataFrame, pd.Series],
    pipeline_config: ChurnPipelineConfig,
) -> ModelBenchmarkRow:
    """Run Stratified K-Fold CV and OOF probability calibration for one model family."""
    features_df, target_series = data_pair
    x_df = features_df.copy().reset_index(drop=True)
    y_arr = target_series.to_numpy(dtype=int)
    val_cfg = pipeline_config.validation
    splitter = StratifiedKFold(n_splits=val_cfg.n_splits, shuffle=True, random_state=val_cfg.random_seed)
    oof_raw = np.zeros(len(x_df), dtype=float)
    fold_aucs: list[float] = []

    for train_idx, val_idx in splitter.split(x_df, y_arr):
        x_tr, x_va = x_df.iloc[train_idx], x_df.iloc[val_idx]
        y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
        if model_name == "LogisticRegression":
            preds = _fit_predict_linear_fold((x_tr, y_tr), x_va, pipeline_config)
        else:
            preds = _fit_predict_gbdt_fold(model_name, (x_tr, x_va, y_tr), pipeline_config)
        oof_raw[val_idx] = preds
        fold_aucs.append(float(roc_auc_score(y_va, preds)))

    opt_thr, opt_f1 = _find_optimal_f1_threshold(y_arr, oof_raw)
    cal_probs, cal_method, cal_brier, cal_ece = calibrate_oof_probabilities(y_arr, oof_raw, val_cfg)
    return ModelBenchmarkRow(
        model_name=model_name,
        fold_aucs=tuple(fold_aucs),
        oof_roc_auc=float(roc_auc_score(y_arr, oof_raw)),
        oof_pr_auc=float(average_precision_score(y_arr, oof_raw)),
        oof_f1_default=float(f1_score(y_arr, (oof_raw >= 0.50).astype(int))),
        oof_f1_optimal=opt_f1,
        optimal_threshold=opt_thr,
        oof_brier=float(brier_score_loss(y_arr, oof_raw)),
        oof_ece=compute_expected_calibration_error(y_arr, oof_raw, val_cfg.calibration_bins),
        calibration_method=cal_method,
        calibrated_brier=cal_brier,
        calibrated_ece=cal_ece,
        oof_raw_probs=oof_raw,
        oof_calibrated_probs=cal_probs,
    )


def run_stratified_cv_benchmark(
    features_df: pd.DataFrame,
    target: pd.Series,
    pipeline_config: ChurnPipelineConfig,
) -> tuple[ModelBenchmarkRow, ...]:
    """Benchmark all four model families using Stratified K-Fold CV on leak-free features.

    Inputs:
        features_df: Clean 12-column pre-event feature DataFrame (copied internally).
        target: Binary churn target ``Exited``.
        pipeline_config: Frozen pipeline configuration.

    Returns:
        Tuple of ``ModelBenchmarkRow`` sorted descending by ``oof_roc_auc``.
    """
    rows = [
        _evaluate_single_model_cv(name, (features_df, target), pipeline_config)
        for name in BENCHMARK_MODELS
    ]
    rows.sort(key=lambda item: item.oof_roc_auc, reverse=True)
    return tuple(rows)


def run_leakage_ablation_suite(
    bundle: ChurnDatasetBundle,
    pipeline_config: ChurnPipelineConfig,
) -> tuple[LeakageAblationRow, ...]:
    """Quantify CV inflation under leaky ``Complain`` and ``Satisfaction Score`` inclusion.

    Inputs:
        bundle: Raw churn dataset bundle.
        pipeline_config: Master frozen configuration.

    Returns:
        Tuple of ``LeakageAblationRow`` covering ``leaky_with_complain``,
        ``no_complain_with_satisfaction``, and ``strict_leak_free`` across
        ``LogisticRegression`` and ``LightGBM``.
    """
    scenarios: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("leaky_with_complain", ("Complain", "Satisfaction Score")),
        ("no_complain_with_satisfaction", ("Satisfaction Score",)),
        ("strict_leak_free", ()),
    )
    results: list[LeakageAblationRow] = []
    for scenario_name, extra_cols in scenarios:
        x_mat, y_ser = build_ablation_feature_matrix(bundle.raw_df, pipeline_config.partition, extra_cols)
        for model_name in ("LogisticRegression", "LightGBM"):
            bench = _evaluate_single_model_cv(model_name, (x_mat, y_ser), pipeline_config)
            results.append(
                LeakageAblationRow(
                    scenario_name=scenario_name,
                    model_name=model_name,
                    feature_count=int(x_mat.shape[1]),
                    roc_auc=bench.oof_roc_auc,
                    pr_auc=bench.oof_pr_auc,
                    f1_score=bench.oof_f1_optimal,
                    brier_score=bench.oof_brier,
                )
            )
    return tuple(results)
