"""Campaign-economics scenario analysis for retention targeting decisions.

Everything here is scenario-based decision analysis under explicit,
labeled assumptions (cost per contact, value of a retained customer,
assumed retention success rate). The dataset contains no randomized
treatment, so none of these numbers are causal estimates -- they let us
compare targeting policies under a stated set of assumptions and show how
the recommended policy moves as those assumptions change.

The decision threshold is derived ANALYTICALLY from the business
assumptions, not by scanning validation outcomes for whatever threshold
happened to maximize realized net value. For a customer with predicted
churn probability p, contacting them has expected scenario value

    EV(p) = p * success_rate * value_per_retained - cost_per_contact

which is positive iff

    p > cost_per_contact / (success_rate * value_per_retained) =: p*

p* (the break-even probability) depends only on the stated assumptions.
Contacting every customer with predicted p >= p* maximizes the sum of
EV(p) over the population, given that the model's probabilities are taken
at face value -- no outcome data is used to pick p*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def breakeven_threshold(
    cost_per_contact: float, success_rate: float, value_per_retained: float
) -> float:
    """Analytical break-even probability p* = cost / (success_rate * value).

    Depends only on the stated business assumptions -- never on outcome
    labels. p* > 1 means no customer's probability can clear the bar
    ("target nobody" is the rational policy); p* <= 0 means every contact
    is expected to be net positive regardless of risk score ("target
    everyone").
    """
    if success_rate <= 0 or value_per_retained <= 0:
        return np.inf
    return cost_per_contact / (success_rate * value_per_retained)


def policy_label(threshold: float) -> str:
    if threshold > 1:
        return "target nobody"
    if threshold <= 0:
        return "target everyone"
    return "target p >= threshold"


def apply_analytical_policy(
    y_true,
    y_proba,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
) -> dict:
    """Apply the analytically-derived break-even threshold and report the
    resulting scenario economics on whatever data is passed in.

    These are SCENARIO / EXPECTED figures under the stated assumptions,
    not observed or causal campaign outcomes.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    threshold = breakeven_threshold(cost_per_contact, success_rate, value_per_retained)

    if threshold > 1:
        contacted = np.zeros(n, dtype=bool)
    elif threshold <= 0:
        contacted = np.ones(n, dtype=bool)
    else:
        contacted = y_proba >= threshold

    n_contacted = int(contacted.sum())
    true_churners_reached = int(y_true[contacted].sum())
    expected_retained = true_churners_reached * success_rate
    cost = n_contacted * cost_per_contact
    value_preserved = expected_retained * value_per_retained
    net_value = value_preserved - cost

    return {
        "threshold": threshold,
        "policy": policy_label(threshold),
        "cost_per_contact": cost_per_contact,
        "value_per_retained": value_per_retained,
        "success_rate": success_rate,
        "n_contacted": n_contacted,
        "pct_contacted": 100 * n_contacted / n,
        "true_churners_reached": true_churners_reached,
        "expected_retained": expected_retained,
        "cost": cost,
        "expected_value_preserved": value_preserved,
        "net_value": net_value,
        "roi": (net_value / cost) if cost > 0 else np.nan,
    }


def analytical_sensitivity_grid(
    y_true,
    y_proba,
    cost_values: list[float],
    success_rate_values: list[float],
    value_values: list[float],
) -> pd.DataFrame:
    """For every combination of assumptions, compute the analytical break-even
    threshold and the resulting scenario economics on the data passed in
    (validation, as a diagnostic -- the threshold itself never depends on
    y_true; only the *reporting* of its consequences uses labels).
    """
    rows = []
    for cost in cost_values:
        for success_rate in success_rate_values:
            for value in value_values:
                result = apply_analytical_policy(
                    y_true, y_proba, cost, value, success_rate
                )
                rows.append(
                    {
                        "cost_per_contact": cost,
                        "success_rate": success_rate,
                        "value_per_retained": value,
                        "breakeven_threshold": result["threshold"],
                        "policy": result["policy"],
                        "n_contacted": result["n_contacted"],
                        "pct_contacted": result["pct_contacted"],
                        "net_value": result["net_value"],
                        "roi": result["roi"],
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic-only: outcome-scanning threshold. NEVER used to choose the
# business policy (that is the analytical break-even threshold above). Kept
# only to sanity-check the analytical threshold against what a hindsight
# scan of validation outcomes would have picked, exactly the way the max-F1
# threshold is kept only as a modeling diagnostic.
# ---------------------------------------------------------------------------


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


def threshold_sweep_diagnostic(
    y_true,
    y_proba,
    thresholds,
    cost_per_contact: float,
    value_per_retained: float,
    success_rate: float,
) -> pd.DataFrame:
    """DIAGNOSTIC ONLY. Sweeps thresholds and reports realized net value on
    the outcomes passed in. Do not use the argmax of this sweep as the
    business policy -- that is exactly the "threshold chosen by scanning
    validation outcomes" mistake this project explicitly avoids. Used only
    to plot the empirical net-value curve next to the analytical p* for a
    sanity check.
    """
    rows = [
        expected_value_at_threshold(
            y_true, y_proba, t, cost_per_contact, value_per_retained, success_rate
        )
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def targeting_policy_comparison(y_true, y_proba, k_fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Model-based top-K targeting vs. random targeting vs. contacting everyone.

    A separate, ranking-based decision framework from the analytical
    break-even threshold above -- used when outreach capacity, not a
    per-contact economic bar, is the binding constraint.
    """
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
