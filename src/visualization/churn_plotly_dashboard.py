"""Standalone interactive Plotly HTML5 executive report for BCCRI."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from src.core.churn_contracts import DashboardPayload

NAVY = "#0F2942"
TEAL = "#0F766E"
AMBER = "#B45309"
CRIMSON = "#B91C1C"
SLATE = "#64748B"

ACTIONABLE_SET = frozenset(
    {"IsActiveMember", "NumOfProducts", "Balance", "HasCrCard", "Card Type", "Point Earned"}
)


def _build_plotly_ablation_and_shap(payload: DashboardPayload) -> str:
    """Build interactive Plotly Figure 1: Leakage Ablation & OOF SHAP Partition."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "3-Way Leakage Ablation (LightGBM OOF)",
            "OOF TreeExplainer Mean |SHAP| Partition",
        ),
        horizontal_spacing=0.14,
    )
    scenarios = ("leaky_with_complain", "no_complain_with_satisfaction", "strict_leak_free")
    labels = ("With Complain (Leaky)", "No Complain + Satisfaction", "Strict Leak-Free")
    lgbm_map = {r.scenario_name: r for r in payload.ablation_rows if r.model_name == "LightGBM"}
    fig.add_trace(
        go.Bar(name="ROC-AUC", x=labels, y=[lgbm_map[s].roc_auc for s in scenarios], marker_color=NAVY),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="PR-AUC", x=labels, y=[lgbm_map[s].pr_auc for s in scenarios], marker_color=AMBER),
        row=1,
        col=1,
    )
    shap_items = sorted(payload.shap_profile.mean_abs_shap.items(), key=lambda kv: kv[1])
    bar_colors = [TEAL if k in ACTIONABLE_SET else SLATE for k, _ in shap_items]
    fig.add_trace(
        go.Bar(
            name="Mean |SHAP|",
            x=[v for _, v in shap_items],
            y=[k for k, _ in shap_items],
            orientation="h",
            marker_color=bar_colors,
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(template="plotly_white", height=420, margin={"t": 50, "b": 40, "l": 120, "r": 30})
    return to_html(fig, full_html=False, include_plotlyjs="cdn")


def _build_plotly_roi_and_sensitivity(payload: DashboardPayload) -> str:
    """Build interactive Plotly Figure 2: Retention ROI Frontier & Sensitivity Heatmap."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Net Retention ROI by Outreach Strategy",
            "Break-Even Sensitivity (Net ROI EUR)",
        ),
        horizontal_spacing=0.12,
    )
    curve = payload.simulation_summary.policy_curve
    series_spec = (
        ("ev_lever_targeted_outreach", "EV / Lever-Targeted", TEAL, "solid"),
        ("risk_ranked_outreach", "Pure Risk-Ranked", NAVY, "dash"),
        ("random_outreach", "Random Outreach", SLATE, "dot"),
    )
    for key, label, color, dash_style in series_spec:
        pts = [pt for pt in curve if pt.strategy_name == key]
        fig.add_trace(
            go.Scatter(
                name=label,
                x=[pt.outreach_fraction * 100.0 for pt in pts],
                y=[pt.net_roi_eur for pt in pts],
                mode="lines+markers",
                line={"color": color, "dash": dash_style, "width": 2.5},
            ),
            row=1,
            col=1,
        )
    grid = payload.simulation_summary.sensitivity_grid
    rates = sorted({c.base_save_rate for c in grid})
    costs = sorted({c.offer_cost_eur for c in grid})
    lookup = {(c.base_save_rate, c.offer_cost_eur): c.optimal_net_roi_eur for c in grid}
    z_matrix = [[lookup[(r, c)] for c in costs] for r in rates]
    fig.add_trace(
        go.Heatmap(
            z=z_matrix,
            x=[f"{c:.0f} EUR" for c in costs],
            y=[f"{r*100:.0f}%" for r in rates],
            colorscale=[[0.0, CRIMSON], [0.45, "#FEF3C7"], [1.0, TEAL]],
            colorbar={"title": "EUR"},
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Outreach Fraction (%)", row=1, col=1)
    fig.update_yaxes(title_text="Net ROI (EUR)", row=1, col=1)
    fig.update_xaxes(title_text="Offer Cost (EUR)", row=1, col=2)
    fig.update_yaxes(title_text="Anchor Save Rate", row=1, col=2)
    fig.update_layout(template="plotly_white", height=440, margin={"t": 50, "b": 50, "l": 60, "r": 30})
    return to_html(fig, full_html=False, include_plotlyjs=False)


def _format_segment_rows_html(payload: DashboardPayload) -> str:
    """Render CRM segment unit-economics rows with tabular-nums alignment."""
    rows_html: list[str] = []
    for seg in payload.simulation_summary.segment_summaries:
        rows_html.append(
            "<tr>"
            f"<td><strong>{seg.segment_name}</strong></td>"
            f"<td class='num'>{seg.client_count:,} ({seg.client_share*100:.1f}%)</td>"
            f"<td class='num'>{seg.mean_churn_prob*100:.1f}%</td>"
            f"<td class='num'>{seg.actual_churn_rate*100:.1f}%</td>"
            f"<td class='num'>{seg.mean_clv_eur:,.0f} EUR</td>"
            f"<td class='num'>{seg.total_at_risk_clv_eur:,.0f} EUR</td>"
            f"<td>{seg.recommended_action}</td>"
            "</tr>"
        )
    return "\n".join(rows_html)


def _format_benchmark_rows_html(payload: DashboardPayload) -> str:
    """Render 5-fold CV model benchmark rows with tabular-nums alignment."""
    rows_html: list[str] = []
    for row in payload.benchmark_rows:
        rows_html.append(
            "<tr>"
            f"<td><strong>{row.model_name}</strong></td>"
            f"<td class='num'>{row.oof_roc_auc:.4f}</td>"
            f"<td class='num'>{row.oof_pr_auc:.4f}</td>"
            f"<td class='num'>{row.oof_f1_optimal:.4f}</td>"
            f"<td class='num'>{row.oof_brier:.4f}</td>"
            f"<td class='num'>{row.calibrated_brier:.4f} ({row.calibration_method})</td>"
            f"<td class='num'>{row.calibrated_ece:.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows_html)


def generate_churn_dashboard(payload: DashboardPayload, output_path: Path) -> Path:
    """Build standalone HTML5 Plotly dashboard with retail-banking styling and tabular-nums.

    Inputs:
        payload: Immutable ``DashboardPayload`` with CV, SHAP, segment, and ROI results.
        output_path: Destination path (``BCCRI/artifacts/dashboard.html``).

    Returns:
        Written HTML file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_model = next(r for r in payload.benchmark_rows if r.model_name == payload.best_model_name)
    opt_ev = payload.simulation_summary.optimal_ev_point
    opt_risk = payload.simulation_summary.optimal_risk_point
    ev_lift = opt_ev.net_roi_eur - opt_risk.net_roi_eur

    chart_one = _build_plotly_ablation_and_shap(payload)
    chart_two = _build_plotly_roi_and_sensitivity(payload)
    seg_rows = _format_segment_rows_html(payload)
    bench_rows = _format_benchmark_rows_html(payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BCCRI — Bank Customer Churn &amp; Retention Intelligence</title>
<style>
  :root {{
    --bg: #F8FAFC; --surface: #FFFFFF; --border: #CBD5E1; --ink: #0F172A;
    --muted: #475569; --navy: #0F2942; --teal: #0F766E; --amber: #B45309; --crimson: #B91C1C;
  }}
  body {{
    margin: 0; padding: 24px 32px; background: var(--bg); color: var(--ink);
    font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
    font-variant-numeric: tabular-nums;
  }}
  header {{ border-bottom: 2px solid var(--navy); padding-bottom: 14px; margin-bottom: 22px; }}
  h1 {{ margin: 0; font-size: 1.55rem; color: var(--navy); letter-spacing: -0.01em; }}
  .subtitle {{ margin-top: 4px; color: var(--muted); font-size: 0.92rem; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi-card {{ background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--navy); padding: 12px 14px; border-radius: 4px; }}
  .kpi-card.teal {{ border-top-color: var(--teal); }}
  .kpi-card.crimson {{ border-top-color: var(--crimson); }}
  .kpi-value {{ font-size: 1.35rem; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; }}
  .kpi-label {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.03em; }}
  .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 16px; margin-bottom: 20px; }}
  h2 {{ margin: 0 0 12px 0; font-size: 1.05rem; color: var(--navy); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; }}
  th, td {{ padding: 9px 10px; border-bottom: 1px solid #E2E8F0; text-align: left; }}
  th {{ background: #F1F5F9; color: var(--navy); font-weight: 600; }}
  td.num {{ text-align: right; font-family: "JetBrains Mono", Consolas, monospace; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
<header>
  <h1>BCCRI — Bank Customer Churn &amp; Retention Intelligence</h1>
  <div class="subtitle">Leak-Free 5-Fold Stratified CV, OOF TreeExplainer Lever Partition &amp; Unit-Economics Retention ROI Simulator</div>
</header>
<section class="kpi-grid">
  <div class="kpi-card"><div class="kpi-value">{payload.dataset_bundle.n_rows:,}</div><div class="kpi-label">Retail Accounts</div></div>
  <div class="kpi-card"><div class="kpi-value">{payload.dataset_bundle.churn_rate*100:.2f}%</div><div class="kpi-label">Observed Churn Rate</div></div>
  <div class="kpi-card"><div class="kpi-value">{best_model.oof_roc_auc:.4f}</div><div class="kpi-label">{best_model.model_name} OOF ROC-AUC</div></div>
  <div class="kpi-card teal"><div class="kpi-value">{payload.shap_profile.actionable_share*100:.1f}%</div><div class="kpi-label">Actionable SHAP Share</div></div>
  <div class="kpi-card teal"><div class="kpi-value">+{opt_ev.net_roi_eur:,.0f} EUR</div><div class="kpi-label">Peak EV-Targeted Net ROI</div></div>
  <div class="kpi-card crimson"><div class="kpi-value">+{ev_lift:,.0f} EUR</div><div class="kpi-label">EV Gain vs Risk-Ranked</div></div>
</section>
<section class="panel">{chart_one}</section>
<section class="panel">{chart_two}</section>
<section class="panel">
  <h2>CRM Retention Segments &amp; Unit-Economics Exposure</h2>
  <table>
    <thead><tr><th>CRM Segment</th><th>Clients</th><th>Mean Pred Risk</th><th>Actual Churn</th><th>Mean 2Y CLV</th><th>At-Risk CLV</th><th>Recommended Retention Play</th></tr></thead>
    <tbody>{seg_rows}</tbody>
  </table>
</section>
<section class="panel">
  <h2>Leak-Free 5-Fold Stratified CV Benchmark &amp; Calibration</h2>
  <table>
    <thead><tr><th>Model Family</th><th>OOF ROC-AUC</th><th>OOF PR-AUC</th><th>Optimal F1</th><th>Raw Brier</th><th>Calibrated Brier</th><th>Calibrated ECE</th></tr></thead>
    <tbody>{bench_rows}</tbody>
  </table>
</section>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
