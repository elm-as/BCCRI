"""Retail-banking CLV proxy estimation, CRM aggregation, and retention ROI simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.churn_config import RetentionEconomicsConfig
from src.core.churn_contracts import (
    CrmSegmentSummary,
    RetentionPolicyPoint,
    RetentionSimulationSummary,
    SensitivityCell,
)
from src.evaluation.shap_lever_segmenter import SEGMENT_ACTIONS


def estimate_customer_clv_eur(
    features_df: pd.DataFrame,
    econ_config: RetentionEconomicsConfig,
) -> np.ndarray:
    """Estimate 2-year discounted customer lifetime value (CLV) in EUR.

    Inputs:
        features_df: Customer DataFrame containing ``Balance`` and ``EstimatedSalary``.
        econ_config: Unit-economics parameters (NIM, fee proxy rate, discount rate, floor).

    Returns:
        1D NumPy array of strictly positive CLV estimates floored at ``min_clv_eur``.

    Edge Cases:
        Zero-balance accounts receive salary-linked fee/cross-sell value and never
        fall below ``econ_config.min_clv_eur``. Input DataFrame is never mutated.
    """
    if "Balance" not in features_df.columns or "EstimatedSalary" not in features_df.columns:
        raise KeyError("DataFrame must contain Balance and EstimatedSalary for CLV estimation.")
    balance = features_df["Balance"].to_numpy(dtype=float)
    salary = features_df["EstimatedSalary"].to_numpy(dtype=float)
    annual_rev = balance * econ_config.nim_annual_rate + salary * econ_config.salary_fee_proxy_rate
    pv_factor = sum(
        1.0 / ((1.0 + econ_config.discount_rate) ** year)
        for year in range(1, econ_config.horizon_years + 1)
    )
    return np.maximum(econ_config.min_clv_eur, annual_rev * pv_factor)


def _segment_parameter_maps(
    econ_config: RetentionEconomicsConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """Build save-rate and offer-cost lookup dicts from frozen economics tuples."""
    save_map = {name: save_rate for name, save_rate, _ in econ_config.segment_economics}
    cost_map = {name: offer_cost for name, _, offer_cost in econ_config.segment_economics}
    return save_map, cost_map


def _evaluate_policy_at_fraction(
    ranked_frame: pd.DataFrame,
    strategy_name: str,
    fraction: float,
) -> RetentionPolicyPoint:
    """Compute realized campaign ROI when targeting the top ``fraction`` of ranked clients."""
    n_total = len(ranked_frame)
    n_target = max(1, int(round(fraction * n_total)))
    cohort = ranked_frame.iloc[:n_target]
    saved_clients = float((cohort["y_true"] * cohort["save_rate"]).sum())
    gross_saved = float((cohort["y_true"] * cohort["save_rate"] * cohort["clv_eur"]).sum())
    total_cost = float(cohort["offer_cost_eur"].sum())
    net_roi = gross_saved - total_cost
    roi_pct = 0.0 if total_cost == 0.0 else (net_roi / total_cost) * 100.0
    cutoff = float(cohort["rank_score"].iloc[-1])
    return RetentionPolicyPoint(
        strategy_name=strategy_name,
        outreach_fraction=float(fraction),
        min_score_cutoff=cutoff,
        targeted_clients=int(n_target),
        expected_saved_clients=saved_clients,
        gross_clv_saved_eur=gross_saved,
        campaign_cost_eur=total_cost,
        net_roi_eur=float(net_roi),
        roi_pct=float(roi_pct),
    )


def _evaluate_random_baseline_at_fraction(
    base_frame: pd.DataFrame,
    fraction: float,
) -> RetentionPolicyPoint:
    """Compute analytical expectation of uniform random outreach at budget ``fraction``."""
    n_target = max(1, int(round(fraction * len(base_frame))))
    scale = float(n_target) / float(len(base_frame))
    saved_clients = float((base_frame["y_true"] * base_frame["save_rate"]).sum()) * scale
    gross_saved = float((base_frame["y_true"] * base_frame["save_rate"] * base_frame["clv_eur"]).sum()) * scale
    total_cost = float(base_frame["offer_cost_eur"].sum()) * scale
    net_roi = gross_saved - total_cost
    roi_pct = 0.0 if total_cost == 0.0 else (net_roi / total_cost) * 100.0
    return RetentionPolicyPoint(
        strategy_name="random_outreach",
        outreach_fraction=float(fraction),
        min_score_cutoff=0.0,
        targeted_clients=int(n_target),
        expected_saved_clients=saved_clients,
        gross_clv_saved_eur=gross_saved,
        campaign_cost_eur=total_cost,
        net_roi_eur=float(net_roi),
        roi_pct=float(roi_pct),
    )


def build_crm_segment_summaries(
    eval_df: pd.DataFrame,
    econ_config: RetentionEconomicsConfig,
) -> tuple[CrmSegmentSummary, ...]:
    """Aggregate client volume, churn risk, and at-risk CLV across CRM segments.

    Inputs:
        eval_df: Segmented DataFrame containing ``crm_segment``, ``p_churn``, ``y_true``.
        econ_config: Retention economics configuration.

    Returns:
        Tuple of ``CrmSegmentSummary`` sorted descending by ``total_at_risk_clv_eur``.
    """
    frame = eval_df.copy()
    frame["clv_eur"] = estimate_customer_clv_eur(frame, econ_config)
    total_clients = float(len(frame))
    summaries: list[CrmSegmentSummary] = []

    for seg_name, _, _ in econ_config.segment_economics:
        sub = frame[frame["crm_segment"] == seg_name]
        if sub.empty:
            continue
        at_risk_clv = float((sub["p_churn"] * sub["clv_eur"]).sum())
        summaries.append(
            CrmSegmentSummary(
                segment_name=seg_name,
                client_count=int(len(sub)),
                client_share=float(len(sub) / total_clients),
                mean_churn_prob=float(sub["p_churn"].mean()),
                actual_churn_rate=float(sub["y_true"].mean()),
                mean_clv_eur=float(sub["clv_eur"].mean()),
                total_at_risk_clv_eur=at_risk_clv,
                recommended_action=SEGMENT_ACTIONS[seg_name],
            )
        )
    summaries.sort(key=lambda item: item.total_at_risk_clv_eur, reverse=True)
    return tuple(summaries)


def _compute_sensitivity_grid(
    base_frame: pd.DataFrame,
    econ_config: RetentionEconomicsConfig,
) -> tuple[SensitivityCell, ...]:
    """Evaluate peak Net EV-targeted ROI over a 2D grid of save rates and offer costs."""
    cells: list[SensitivityCell] = []
    anchor = econ_config.base_save_rate_anchor
    for save_rate in econ_config.sensitivity_save_rates:
        ratio = save_rate / anchor
        scaled_save = np.clip(base_frame["save_rate"].to_numpy() * ratio, 0.01, 0.90)
        for cost_eur in econ_config.sensitivity_offer_costs:
            ev_vec = (
                base_frame["p_churn"].to_numpy() * scaled_save * base_frame["clv_eur"].to_numpy()
                - cost_eur
            )
            realized_vec = (
                base_frame["y_true"].to_numpy() * scaled_save * base_frame["clv_eur"].to_numpy()
                - cost_eur
            )
            order = np.argsort(-ev_vec)
            best_roi, best_frac = _best_roi_across_fractions(realized_vec[order], econ_config.outreach_fractions)
            cells.append(
                SensitivityCell(
                    base_save_rate=float(save_rate),
                    offer_cost_eur=float(cost_eur),
                    optimal_net_roi_eur=best_roi,
                    optimal_fraction=best_frac,
                )
            )
    return tuple(cells)


def _best_roi_across_fractions(
    ordered_realized: np.ndarray,
    fractions: tuple[float, ...],
) -> tuple[float, float]:
    """Find the outreach fraction maximizing cumulative realized Net ROI."""
    n_total = len(ordered_realized)
    cum_roi = np.cumsum(ordered_realized)
    best_roi = -float("inf")
    best_frac = fractions[0]
    for frac in fractions:
        idx = max(1, int(round(frac * n_total))) - 1
        val = float(cum_roi[idx])
        if val > best_roi:
            best_roi = val
            best_frac = float(frac)
    return float(best_roi), float(best_frac)


def simulate_retention_strategies(
    eval_df: pd.DataFrame,
    econ_config: RetentionEconomicsConfig,
) -> RetentionSimulationSummary:
    """Simulate Random, Pure Risk-Ranked, and EV/Lever-Targeted retention policies.

    Inputs:
        eval_df: Customer DataFrame with ``p_churn``, ``y_true``, ``crm_segment``,
            ``Balance``, and ``EstimatedSalary`` (copied internally).
        econ_config: Retention unit-economics parameters and budget grid.

    Returns:
        Frozen ``RetentionSimulationSummary`` with optimal policy points, full curve,
        2D sensitivity grid, and CRM segment summaries.
    """
    frame = eval_df.copy().reset_index(drop=True)
    save_map, cost_map = _segment_parameter_maps(econ_config)
    frame["clv_eur"] = estimate_customer_clv_eur(frame, econ_config)
    frame["save_rate"] = frame["crm_segment"].map(save_map).fillna(0.05)
    frame["offer_cost_eur"] = frame["crm_segment"].map(cost_map).fillna(econ_config.base_offer_cost_eur)
    frame["expected_ev"] = frame["p_churn"] * frame["save_rate"] * frame["clv_eur"] - frame["offer_cost_eur"]

    risk_ranked = frame.assign(rank_score=frame["p_churn"]).sort_values("rank_score", ascending=False)
    ev_ranked = frame.assign(rank_score=frame["expected_ev"]).sort_values("rank_score", ascending=False)

    curve: list[RetentionPolicyPoint] = []
    ev_points: list[RetentionPolicyPoint] = []
    risk_points: list[RetentionPolicyPoint] = []
    for frac in econ_config.outreach_fractions:
        rand_pt = _evaluate_random_baseline_at_fraction(frame, frac)
        risk_pt = _evaluate_policy_at_fraction(risk_ranked, "risk_ranked_outreach", frac)
        ev_pt = _evaluate_policy_at_fraction(ev_ranked, "ev_lever_targeted_outreach", frac)
        curve.extend((rand_pt, risk_pt, ev_pt))
        risk_points.append(risk_pt)
        ev_points.append(ev_pt)

    opt_ev = max(ev_points, key=lambda pt: pt.net_roi_eur)
    opt_risk = max(risk_points, key=lambda pt: pt.net_roi_eur)
    rand_at_opt = _evaluate_random_baseline_at_fraction(frame, opt_ev.outreach_fraction)
    return RetentionSimulationSummary(
        optimal_ev_point=opt_ev,
        optimal_risk_point=opt_risk,
        random_at_optimal_fraction_point=rand_at_opt,
        policy_curve=tuple(curve),
        sensitivity_grid=_compute_sensitivity_grid(frame, econ_config),
        segment_summaries=build_crm_segment_summaries(frame, econ_config),
    )
