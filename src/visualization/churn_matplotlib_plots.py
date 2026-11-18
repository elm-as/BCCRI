"""Export 300 DPI retail-banking figures for leakage audit, SHAP segments, and ROI."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.churn_config import FeaturePartitionConfig
from src.core.churn_contracts import DashboardPayload
from src.modeling.probability_calibrator import compute_reliability_bins

PALETTE_NAVY = "#0F2942"
PALETTE_TEAL = "#0F766E"
PALETTE_AMBER = "#B45309"
PALETTE_CRIMSON = "#B91C1C"
PALETTE_SLATE = "#64748B"
PALETTE_BG = "#F8FAFC"
PALETTE_INK = "#0F172A"
PALETTE_GRID = "#CBD5E1"

ACTIONABLE_SET = frozenset(FeaturePartitionConfig().actionable_lever_columns)


def _apply_axis_style(ax: plt.Axes, title: str) -> None:
    """Apply clean institutional banking styling to a Matplotlib axis."""
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE_GRID)
    ax.spines["bottom"].set_color(PALETTE_GRID)
    ax.grid(axis="both", linestyle="--", linewidth=0.6, alpha=0.55, color=PALETTE_GRID)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=PALETTE_INK, pad=10)
    ax.tick_params(colors=PALETTE_INK, labelsize=9)


def _render_leakage_and_shap_figure(payload: DashboardPayload, output_path: Path) -> Path:
    """Render Figure 1: 3-Way Leakage Ablation & Actionable vs Static OOF SHAP."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(PALETTE_BG)

    scenario_order = ("leaky_with_complain", "no_complain_with_satisfaction", "strict_leak_free")
    scenario_labels = ("With Complain\n(Leaky)", "No Complain +\nSatisfaction", "Strict\nLeak-Free")
    lgbm_rows = {r.scenario_name: r for r in payload.ablation_rows if r.model_name == "LightGBM"}
    lr_rows = {r.scenario_name: r for r in payload.ablation_rows if r.model_name == "LogisticRegression"}

    x_pos = np.arange(len(scenario_order))
    width = 0.28
    lgbm_aucs = [lgbm_rows[s].roc_auc for s in scenario_order]
    lr_aucs = [lr_rows[s].roc_auc for s in scenario_order]
    lgbm_prs = [lgbm_rows[s].pr_auc for s in scenario_order]

    b1 = ax_left.bar(x_pos - width, lgbm_aucs, width, label="LightGBM ROC-AUC", color=PALETTE_NAVY)
    b2 = ax_left.bar(x_pos, lr_aucs, width, label="LogReg ROC-AUC", color=PALETTE_SLATE)
    b3 = ax_left.bar(x_pos + width, lgbm_prs, width, label="LightGBM PR-AUC", color=PALETTE_AMBER)
    for bar_group in (b1, b2, b3):
        ax_left.bar_label(bar_group, fmt="%.3f", padding=3, fontsize=8, color=PALETTE_INK)

    complain_row = next((r for r in payload.leakage_audit_rows if r.feature_name == "Complain"), None)
    r_val = complain_row.pearson_corr if complain_row else 0.996
    ax_left.annotate(
        f"Complain post-event leak\nPearson r = {r_val:.4f}",
        xy=(0 - width, lgbm_aucs[0]),
        xytext=(0.55, 0.93),
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE_CRIMSON,
        arrowprops={"arrowstyle": "->", "color": PALETTE_CRIMSON, "lw": 1.2},
    )
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(scenario_labels)
    ax_left.set_ylim(0.40, 1.09)
    ax_left.set_ylabel("5-Fold Stratified OOF Score", color=PALETTE_INK, fontsize=10)
    ax_left.legend(loc="lower left", fontsize=8.5, frameon=True)
    _apply_axis_style(ax_left, "A. 3-Way Leakage Ablation (Complain & Satisfaction Audit)")

    shap_items = sorted(payload.shap_profile.mean_abs_shap.items(), key=lambda kv: kv[1])
    names = [item[0] for item in shap_items]
    vals = [item[1] for item in shap_items]
    colors = [PALETTE_TEAL if name in ACTIONABLE_SET else PALETTE_SLATE for name in names]
    bars = ax_right.barh(names, vals, color=colors, height=0.68)
    ax_right.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, color=PALETTE_INK)
    ax_right.set_xlim(0.0, max(vals) * 1.20)
    ax_right.set_xlabel("Mean |SHAP| (Log-Odds Contribution)", color=PALETTE_INK, fontsize=10)
    act_pct = payload.shap_profile.actionable_share * 100.0
    stat_pct = payload.shap_profile.static_share * 100.0
    _apply_axis_style(
        ax_right,
        f"B. OOF TreeExplainer Partition (Actionable {act_pct:.1f}% vs Static {stat_pct:.1f}%)",
    )
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_segments_and_calibration_figure(payload: DashboardPayload, output_path: Path) -> Path:
    """Render Figure 2: CRM Actionable Segments & 4-Model Calibrated Reliability."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(PALETTE_BG)

    segs = list(reversed(payload.simulation_summary.segment_summaries))
    seg_names = [s.segment_name for s in segs]
    clv_risk_k = [s.total_at_risk_clv_eur / 1000.0 for s in segs]
    seg_colors = [
        PALETTE_SLATE if "Healthy" in s.segment_name
        else (PALETTE_AMBER if "Structural" in s.segment_name else PALETTE_TEAL)
        for s in segs
    ]
    bars = ax_left.barh(seg_names, clv_risk_k, color=seg_colors, height=0.62)
    labels = [f"{s.total_at_risk_clv_eur/1000.0:.0f}k EUR | n={s.client_count:,} | {s.actual_churn_rate*100:.1f}% churn" for s in segs]
    ax_left.bar_label(bars, labels=labels, padding=4, fontsize=8, color=PALETTE_INK)
    ax_left.set_xlim(0.0, max(clv_risk_k) * 1.48)
    ax_left.set_xlabel("Expected At-Risk 2-Year CLV (Thousand EUR)", color=PALETTE_INK, fontsize=10)
    _apply_axis_style(ax_left, "A. Actionable CRM Segments by At-Risk CLV & Empirical Churn")

    y_true = payload.dataset_bundle.raw_df["Exited"].to_numpy(dtype=int)
    model_colors = {
        "LightGBM": PALETTE_TEAL,
        "CatBoost": PALETTE_NAVY,
        "XGBoost": PALETTE_AMBER,
        "LogisticRegression": PALETTE_SLATE,
    }
    ax_right.plot([0, 1], [0, 1], linestyle="--", color=PALETTE_CRIMSON, lw=1.2, label="Perfect Calibration")
    for row in payload.benchmark_rows:
        probs = row.oof_calibrated_probs if row.model_name == payload.best_model_name else row.oof_raw_probs
        prob_pred, prob_true = compute_reliability_bins(y_true, probs, 10)
        suffix = f" ({row.calibration_method})" if row.model_name == payload.best_model_name else ""
        brier_val = row.calibrated_brier if row.model_name == payload.best_model_name else row.oof_brier
        ax_right.plot(
            prob_pred,
            prob_true,
            marker="o",
            lw=1.9,
            markersize=4.5,
            color=model_colors.get(row.model_name, PALETTE_NAVY),
            label=f"{row.model_name}{suffix} | AUC={row.oof_roc_auc:.3f}, Brier={brier_val:.4f}",
        )
    ax_right.set_xlabel("Predicted Churn Probability", color=PALETTE_INK, fontsize=10)
    ax_right.set_ylabel("Observed Churn Frequency", color=PALETTE_INK, fontsize=10)
    ax_right.legend(loc="upper left", fontsize=8, frameon=True)
    _apply_axis_style(ax_right, "B. 5-Fold OOF Model Calibration & Discrimination")
    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_retention_roi_figure(payload: DashboardPayload, output_path: Path) -> Path:
    """Render Figure 3: 3-Strategy Retention ROI Curve & 2D Break-Even Sensitivity."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=300)
    fig.patch.set_facecolor(PALETTE_BG)

    curve = payload.simulation_summary.policy_curve
    styles = (
        ("ev_lever_targeted_outreach", "EV / Lever-Targeted", PALETTE_TEAL, "-", 2.4),
        ("risk_ranked_outreach", "Pure Risk-Ranked", PALETTE_NAVY, "--", 2.0),
        ("random_outreach", "Random Outreach", PALETTE_SLATE, ":", 1.8),
    )
    for key, label, color, lstyle, lw in styles:
        pts = [pt for pt in curve if pt.strategy_name == key]
        xs = [pt.outreach_fraction * 100.0 for pt in pts]
        ys = [pt.net_roi_eur / 1000.0 for pt in pts]
        ax_left.plot(xs, ys, label=label, color=color, linestyle=lstyle, linewidth=lw, marker="o", markersize=3.5)

    opt = payload.simulation_summary.optimal_ev_point
    ax_left.axhline(0.0, color=PALETTE_CRIMSON, linewidth=1.0, linestyle="-.")
    ax_left.scatter([opt.outreach_fraction * 100.0], [opt.net_roi_eur / 1000.0], color=PALETTE_AMBER, s=75, zorder=5)
    ax_left.annotate(
        f"Optimal f*={opt.outreach_fraction*100:.0f}%\n+{opt.net_roi_eur/1000.0:.1f}k EUR ({opt.roi_pct:.1f}% ROI)",
        xy=(opt.outreach_fraction * 100.0, opt.net_roi_eur / 1000.0),
        xytext=(opt.outreach_fraction * 100.0 + 18.0, opt.net_roi_eur / 1000.0 * 0.54),
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE_INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFFFFF", "edgecolor": PALETTE_AMBER, "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": PALETTE_AMBER, "lw": 1.3},
    )
    ax_left.set_xlabel("Outreach Budget Fraction (% of Portfolio)", color=PALETTE_INK, fontsize=10)
    ax_left.set_ylabel("Net Retention ROI (Thousand EUR)", color=PALETTE_INK, fontsize=10)
    ax_left.legend(loc="lower right", fontsize=8.5, frameon=True)
    _apply_axis_style(ax_left, "A. Retention Campaign Net ROI by Targeting Strategy")

    grid = payload.simulation_summary.sensitivity_grid
    rates = sorted({c.base_save_rate for c in grid})
    costs = sorted({c.offer_cost_eur for c in grid})
    matrix = np.zeros((len(rates), len(costs)), dtype=float)
    lookup = {(c.base_save_rate, c.offer_cost_eur): c.optimal_net_roi_eur / 1000.0 for c in grid}
    for i, r_val in enumerate(rates):
        for j, c_val in enumerate(costs):
            matrix[i, j] = lookup[(r_val, c_val)]

    im = ax_right.imshow(matrix, cmap="RdYlGn", aspect="auto", origin="lower")
    ax_right.set_xticks(range(len(costs)))
    ax_right.set_xticklabels([f"{c:.0f}" for c in costs])
    ax_right.set_yticks(range(len(rates)))
    ax_right.set_yticklabels([f"{r*100:.0f}%" for r in rates])
    ax_right.set_xlabel("Offer Cost per Client (EUR)", color=PALETTE_INK, fontsize=10)
    ax_right.set_ylabel("Anchor Save Rate", color=PALETTE_INK, fontsize=10)
    for i in range(len(rates)):
        for j in range(len(costs)):
            ax_right.text(j, i, f"{matrix[i, j]:+.0f}k", ha="center", va="center", fontsize=7.5, color=PALETTE_INK)
    fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04, label="Peak Net ROI (k EUR)")
    ax_right.set_title("B. Break-Even Sensitivity (Save Rate x Offer Cost)", fontsize=11.5, fontweight="bold", color=PALETTE_INK, pad=10)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_bccri_figures(payload: DashboardPayload, images_dir: Path) -> tuple[Path, Path, Path]:
    """Export all three 300 DPI BCCRI visual artifacts to ``images_dir``.

    Inputs:
        payload: Complete immutable ``DashboardPayload`` from pipeline evaluation.
        images_dir: Target folder (``BCCRI/images``).

    Returns:
        Tuple of the three generated PNG file paths.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    fig1 = _render_leakage_and_shap_figure(payload, images_dir / "bccri_leakage_audit_shap.png")
    fig2 = _render_segments_and_calibration_figure(payload, images_dir / "bccri_actionable_segments.png")
    fig3 = _render_retention_roi_figure(payload, images_dir / "bccri_retention_roi_simulation.png")
    return fig1, fig2, fig3
