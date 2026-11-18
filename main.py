"""End-to-end CLI orchestrator for BCCRI: Bank Customer Churn & Retention Intelligence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

from src.core.churn_config import ChurnPipelineConfig
from src.core.churn_contracts import DashboardPayload
from src.evaluation.retention_roi_simulator import simulate_retention_strategies
from src.evaluation.shap_lever_segmenter import assign_crm_segments, compute_oof_shap_profile
from src.features.churn_feature_pipeline import build_clean_feature_matrix
from src.features.leakage_auditor import audit_target_leakage, summarize_satisfaction_alignment
from src.ingestion.kaggle_churn_loader import download_churn_dataset, load_and_validate_churn_csv
from src.modeling.stratified_benchmark import run_leakage_ablation_suite, run_stratified_cv_benchmark
from src.visualization.churn_matplotlib_plots import render_bccri_figures
from src.visualization.churn_plotly_dashboard import generate_churn_dashboard


def _build_results_dictionary(payload: DashboardPayload, pipeline_config: ChurnPipelineConfig) -> dict:
    """Serialize frozen domain evaluation artifacts into JSON-safe Python primitives."""
    audit_map = {row.feature_name: row for row in payload.leakage_audit_rows}
    opt_ev = payload.simulation_summary.optimal_ev_point
    opt_risk = payload.simulation_summary.optimal_risk_point
    rand_pt = payload.simulation_summary.random_at_optimal_fraction_point
    return {
        "dataset": {
            "n_rows": payload.dataset_bundle.n_rows,
            "n_cols": payload.dataset_bundle.n_cols,
            "churn_rate": round(payload.dataset_bundle.churn_rate, 6),
            "zero_balance_rate": round(payload.dataset_bundle.zero_balance_rate, 6),
        },
        "leakage_audit": {
            "excluded_ids": list(pipeline_config.partition.id_columns),
            "mandatory_excluded": list(pipeline_config.partition.mandatory_leakage_columns),
            "audit_ablated": list(pipeline_config.partition.audit_candidate_columns),
            "complain_pearson_r": round(audit_map["Complain"].pearson_corr, 6),
            "satisfaction_pearson_r": round(audit_map["Satisfaction Score"].pearson_corr, 6),
            "satisfaction_summary": payload.satisfaction_summary,
            "audit_table": [asdict(row) for row in payload.leakage_audit_rows],
            "ablation_table": [asdict(row) for row in payload.ablation_rows],
        },
        "best_model_name": payload.best_model_name,
        "model_benchmark": [
            {
                "model_name": row.model_name,
                "fold_aucs": [round(float(x), 6) for x in row.fold_aucs],
                "oof_roc_auc": round(row.oof_roc_auc, 6),
                "oof_pr_auc": round(row.oof_pr_auc, 6),
                "oof_f1_default": round(row.oof_f1_default, 6),
                "oof_f1_optimal": round(row.oof_f1_optimal, 6),
                "optimal_threshold": round(row.optimal_threshold, 4),
                "oof_brier": round(row.oof_brier, 6),
                "oof_ece": round(row.oof_ece, 6),
                "calibration_method": row.calibration_method,
                "calibrated_brier": round(row.calibrated_brier, 6),
                "calibrated_ece": round(row.calibrated_ece, 6),
            }
            for row in payload.benchmark_rows
        ],
        "shap_lever_partition": {
            "actionable_share": round(payload.shap_profile.actionable_share, 6),
            "static_share": round(payload.shap_profile.static_share, 6),
            "feature_mean_abs_shap": {
                k: round(float(v), 6) for k, v in payload.shap_profile.mean_abs_shap.items()
            },
            "crm_segments": [asdict(seg) for seg in payload.simulation_summary.segment_summaries],
        },
        "retention_roi": {
            "optimal_strategy": opt_ev.strategy_name,
            "optimal_outreach_fraction": opt_ev.outreach_fraction,
            "optimal_targeted_clients": opt_ev.targeted_clients,
            "optimal_expected_saved_clients": round(opt_ev.expected_saved_clients, 2),
            "optimal_gross_clv_saved_eur": round(opt_ev.gross_clv_saved_eur, 2),
            "optimal_campaign_cost_eur": round(opt_ev.campaign_cost_eur, 2),
            "optimal_net_roi_eur": round(opt_ev.net_roi_eur, 2),
            "optimal_roi_pct": round(opt_ev.roi_pct, 2),
            "risk_ranked_peak_net_roi_eur": round(opt_risk.net_roi_eur, 2),
            "risk_ranked_peak_fraction": opt_risk.outreach_fraction,
            "random_at_optimal_fraction_net_roi_eur": round(rand_pt.net_roi_eur, 2),
            "ev_gain_over_risk_ranked_eur": round(opt_ev.net_roi_eur - opt_risk.net_roi_eur, 2),
            "policy_curve": [asdict(pt) for pt in payload.simulation_summary.policy_curve],
            "sensitivity_grid": [asdict(cell) for cell in payload.simulation_summary.sensitivity_grid],
        },
        "hyperparameters": asdict(pipeline_config.hyperparams),
        "figures": [
            "bccri_leakage_audit_shap.png",
            "bccri_actionable_segments.png",
            "bccri_retention_roi_simulation.png",
        ],
    }


def _apply_smoke_overrides(config: ChurnPipelineConfig) -> ChurnPipelineConfig:
    """Reduce CV splits and boosting trees for rapid end-to-end smoke checks."""
    fast_val = replace(config.validation, n_splits=3)
    fast_hp = replace(
        config.hyperparams,
        lgbm_n_estimators=60,
        xgb_n_estimators=60,
        cat_iterations=80,
    )
    return replace(config, validation=fast_val, hyperparams=fast_hp)


def execute_bccri_pipeline(config: ChurnPipelineConfig) -> dict:
    """Run all six BCCRI pipeline stages and write JSON, PNG, and HTML outputs.

    Inputs:
        config: Master frozen ``ChurnPipelineConfig``.

    Returns:
        Serialized results summary dictionary written to ``artifacts/results.json``.
    """
    print("[1/6] Acquiring and validating Kaggle Bank Customer Churn dataset...")
    csv_path = download_churn_dataset(config.paths)
    bundle = load_and_validate_churn_csv(csv_path, config.partition)

    print("[2/6] Running target leakage audit & 3-way ablation...")
    audit_rows = audit_target_leakage(bundle, config.partition, config.validation)
    sat_summary = summarize_satisfaction_alignment(bundle.raw_df, config.partition)
    ablation_rows = run_leakage_ablation_suite(bundle, config)

    print("[3/6] Running leak-free Stratified CV benchmark & OOF calibration...")
    clean_x, target_y = build_clean_feature_matrix(bundle.raw_df, config.partition)
    benchmark_rows = run_stratified_cv_benchmark(clean_x, target_y, config)
    best_row = benchmark_rows[0]

    print("[4/6] Computing OOF TreeExplainer SHAP actionable vs static levers...")
    shap_profile = compute_oof_shap_profile(clean_x, target_y, config)
    segmented_df = assign_crm_segments(
        clean_x,
        (best_row.oof_calibrated_probs, shap_profile),
        config.validation,
    )
    segmented_df["y_true"] = target_y.to_numpy(dtype=int)

    print("[5/6] Simulating retention ROI across outreach strategies...")
    sim_summary = simulate_retention_strategies(segmented_df, config.economics)

    print("[6/6] Exporting 300 DPI figures, Plotly dashboard, and results.json...")
    payload = DashboardPayload(
        dataset_bundle=bundle,
        leakage_audit_rows=audit_rows,
        ablation_rows=ablation_rows,
        benchmark_rows=benchmark_rows,
        shap_profile=shap_profile,
        simulation_summary=sim_summary,
        satisfaction_summary=sat_summary,
        best_model_name=best_row.model_name,
    )
    render_bccri_figures(payload, config.paths.images_dir)
    generate_churn_dashboard(payload, config.paths.dashboard_path)
    results_dict = _build_results_dictionary(payload, config)
    config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.paths.results_json_path.write_text(
        json.dumps(results_dict, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return results_dict


def main() -> None:
    """Parse CLI arguments and run the BCCRI churn & retention intelligence study."""
    parser = argparse.ArgumentParser(description="BCCRI: Bank Customer Churn & Retention Intelligence")
    parser.add_argument("--fast-smoke", action="store_true", help="Run reduced 3-fold smoke configuration.")
    args = parser.parse_args()
    config = ChurnPipelineConfig.create_default(Path(__file__).resolve().parent)
    if args.fast_smoke:
        config = _apply_smoke_overrides(config)
    summary = execute_bccri_pipeline(config)
    roi = summary["retention_roi"]
    best = summary["model_benchmark"][0]
    print(
        f"Completed BCCRI | Best={best['model_name']} ROC-AUC={best['oof_roc_auc']:.4f} "
        f"| Actionable SHAP={summary['shap_lever_partition']['actionable_share']*100:.1f}% "
        f"| Optimal Net ROI=+{roi['optimal_net_roi_eur']:,.0f} EUR @ f*={roi['optimal_outreach_fraction']*100:.0f}%"
    )


if __name__ == "__main__":
    main()
