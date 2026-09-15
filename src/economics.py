"""Campaign-economics scenario analysis for retention targeting decisions.

Everything here is scenario-based decision analysis under explicit,
labeled assumptions (cost per contact, value of a retained customer,
assumed retention success rate). The dataset contains no randomized
treatment, so none of these numbers are causal estimates -- they let us
compare targeting policies under a stated set of assumptions and show how
the recommended policy moves as those assumptions change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_value_at_threshold(
    y_true,
    y_proba,
    threshold: float,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    contacted = y_proba >= threshold

    n_contacted = int(contacted.sum())
    true_churners_reached = int(y_true[contacted].sum())
    expected_retained = true_churners_reached * success_rate
    cost = n_contacted * cost_per_contact
    value_preserved = expected_retained * value_per_retained
    net_value = value_preserved - cost

    return {
        "threshold": threshold,
        "n_contacted": n_contacted,
        "true_churners_reached": true_churners_reached,
        "expected_retained": expected_retained,
        "cost": cost,
        "expected_value_preserved": value_preserved,
        "net_value": net_value,
        "roi": (net_value / cost) if cost > 0 else np.nan,
    }


def expected_value_at_topk(
    y_true,
    y_proba,
    k_fraction: float,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    n_contacted = max(1, int(round(n * k_fraction)))
    order = np.argsort(-y_proba)
    top_idx = order[:n_contacted]

    true_churners_reached = int(y_true[top_idx].sum())
    expected_retained = true_churners_reached * success_rate
    cost = n_contacted * cost_per_contact
    value_preserved = expected_retained * value_per_retained
    net_value = value_preserved - cost

    return {
        "top_k_pct": k_fraction * 100,
        "n_contacted": n_contacted,
        "true_churners_reached": true_churners_reached,
        "expected_retained": expected_retained,
        "cost": cost,
        "expected_value_preserved": value_preserved,
        "net_value": net_value,
        "roi": (net_value / cost) if cost > 0 else np.nan,
    }


def threshold_sweep(
    y_true,
    y_proba,
    thresholds,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
) -> pd.DataFrame:
    rows = [
        expected_value_at_threshold(
            y_true, y_proba, t, cost_per_contact, value_per_retained, success_rate
        )
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def best_threshold_by_net_value(
    y_true,
    y_proba,
    thresholds,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
    max_contact_fraction: float | None = None,
) -> dict:
    """The one legitimate way to pick a 'business' threshold: maximize expected
    net campaign value under explicit assumptions, evaluated on validation data.

    If max_contact_fraction is given (e.g. call-center capacity of 20% of the
    customer base), the search is restricted to thresholds that respect that
    capacity constraint -- this is the "budget-constrained targeting" framing.
    """
    sweep = threshold_sweep(
        y_true, y_proba, thresholds, cost_per_contact, value_per_retained, success_rate
    )
    if max_contact_fraction is not None:
        n = len(np.asarray(y_true))
        sweep = sweep[sweep["n_contacted"] <= max_contact_fraction * n]
        if sweep.empty:
            raise ValueError("No threshold in the grid satisfies the capacity constraint.")
    best_row = sweep.loc[sweep["net_value"].idxmax()]
    return best_row.to_dict()


def sensitivity_grid(
    y_true,
    y_proba,
    thresholds,
    cost_values: list[float],
    success_rate_values: list[float],
    value_values: list[float],
    max_contact_fraction: float | None = None,
) -> pd.DataFrame:
    """For every combination of assumptions, find the best threshold and its
    expected net value, so we can see how the recommended policy moves.
    """
    n = len(np.asarray(y_true))
    rows = []
    for cost in cost_values:
        for success_rate in success_rate_values:
            for value in value_values:
                best = best_threshold_by_net_value(
                    y_true,
                    y_proba,
                    thresholds,
                    cost,
                    value,
                    success_rate,
                    max_contact_fraction=max_contact_fraction,
                )
                rows.append(
                    {
                        "cost_per_contact": cost,
                        "success_rate": success_rate,
                        "value_per_retained": value,
                        "best_threshold": best["threshold"],
                        "n_contacted": best["n_contacted"],
                        "pct_contacted": 100 * best["n_contacted"] / n,
                        "net_value": best["net_value"],
                        "roi": best["roi"],
                    }
                )
    return pd.DataFrame(rows)


def targeting_policy_comparison(y_true, y_proba, k_fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Model-based top-K targeting vs. random targeting vs. contacting everyone."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    total_churners = int(y_true.sum())
    order = np.argsort(-y_proba)

    rows = []
    for k in k_fractions:
        n_contacted = max(1, int(round(n * k)))
        model_captured = int(y_true[order[:n_contacted]].sum())
        random_captured = total_churners * k  # expectation under uniform random sampling

        rows.append(
            {
                "top_k_pct": k * 100,
                "n_contacted": n_contacted,
                "model_churners_captured": model_captured,
                "model_recall": model_captured / total_churners,
                "random_churners_captured_expected": random_captured,
                "random_recall": k,
                "lift_vs_random": (model_captured / random_captured) if random_captured > 0 else np.nan,
            }
        )
    everyone_row = {
        "top_k_pct": 100.0,
        "n_contacted": n,
        "model_churners_captured": total_churners,
        "model_recall": 1.0,
        "random_churners_captured_expected": total_churners,
        "random_recall": 1.0,
        "lift_vs_random": 1.0,
    }
    rows.append(everyone_row)
    return pd.DataFrame(rows)
