"""Evaluation metrics: classification quality, ranking quality, and calibration.

All functions take y_true (0/1 array) and y_proba (predicted probability of
the positive/churn class) and are pure computations with no data splitting
or fitting, so they are safe to reuse on validation or test data alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_proba, threshold: float) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def ranking_table(y_true, y_proba, k_fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Recall@K, Precision@K, and Lift@K for top-K% highest-risk customers."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    base_rate = y_true.mean()
    order = np.argsort(-y_proba)

    rows = []
    for k in k_fractions:
        n_contacted = max(1, int(round(n * k)))
        top_idx = order[:n_contacted]
        churners_captured = int(y_true[top_idx].sum())
        precision_at_k = churners_captured / n_contacted
        recall_at_k = churners_captured / y_true.sum()
        lift_at_k = precision_at_k / base_rate if base_rate > 0 else np.nan
        rows.append(
            {
                "top_k_pct": k * 100,
                "n_contacted": n_contacted,
                "churners_captured": churners_captured,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_at_k": lift_at_k,
            }
        )
    return pd.DataFrame(rows)


def cumulative_gains_curve(y_true, y_proba) -> pd.DataFrame:
    """Cumulative share of churners captured vs. cumulative share of population contacted."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    n = len(y_true)
    total_positives = y_sorted.sum()

    cum_positives = np.cumsum(y_sorted)
    pct_contacted = np.arange(1, n + 1) / n
    pct_captured = cum_positives / total_positives

    curve = pd.DataFrame({"pct_contacted": pct_contacted, "pct_captured": pct_captured})
    # Downsample to ~500 points for lightweight plotting.
    if len(curve) > 500:
        step = len(curve) // 500
        curve = curve.iloc[::step].reset_index(drop=True)
    return curve


def calibration_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_prob": mean_pred, "fraction_of_positives": frac_pos})


def calibration_summary(y_true, y_proba) -> dict:
    return {
        "brier_score": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
    }


def segment_report(
    df: pd.DataFrame,
    segment_col: str,
    y_true_col: str,
    targeted_col: str,
    policy_name: str,
) -> pd.DataFrame:
    """Churn rate, targeting share, and recall broken out by a single segment column.

    `targeted_col` is a precomputed boolean column (0/1) saying whether each
    customer would be contacted under a specific, named policy
    (`policy_name`) -- e.g. proba >= 0.5, proba >= the analytical break-even
    threshold, or membership in a Top-K% risk list. Every row reports the
    absolute churner count and the absolute captured count alongside the
    recall rate, so a "0% recall" headline is never shown without its
    denominator, and the policy it refers to is always explicit.

    Segments are purely descriptive/correlational cuts of the data; they are
    not causal claims about what drives churn within a segment.
    """
    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        y_true = group[y_true_col].to_numpy()
        y_pred = group[targeted_col].to_numpy().astype(int)
        n = len(group)
        n_churners = int(y_true.sum())
        n_targeted = int(y_pred.sum())
        n_churners_captured = int((y_true & y_pred).sum())
        recall = (n_churners_captured / n_churners) if n_churners > 0 else np.nan
        precision = (n_churners_captured / n_targeted) if n_targeted > 0 else np.nan
        rows.append(
            {
                segment_col: segment_value,
                "policy": policy_name,
                "n_customers": n,
                "n_churners": n_churners,
                "churn_rate": y_true.mean(),
                "share_of_data": n / len(df),
                "n_targeted": n_targeted,
                "share_targeted_within_segment": n_targeted / n,
                "n_churners_captured": n_churners_captured,
                "recall_within_segment": recall,
                "precision_within_segment": precision,
            }
        )
    return pd.DataFrame(rows).sort_values("churn_rate", ascending=False).reset_index(drop=True)


def full_comparison_row(y_true, y_proba, threshold: float) -> dict:
    """Classification + ranking-adjacent + calibration metrics in one row, for
    a head-to-head model comparison that goes beyond ROC-AUC/F1.
    """
    row = classification_report_dict(y_true, y_proba, threshold)
    row.update(calibration_summary(y_true, y_proba))
    return row


def random_targeting_table(y_true, k_fractions=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Expected recall/precision/lift if customers were contacted at random (no model)."""
    y_true = np.asarray(y_true)
    base_rate = y_true.mean()
    rows = []
    for k in k_fractions:
        rows.append(
            {
                "top_k_pct": k * 100,
                "precision_at_k": base_rate,
                "recall_at_k": k,
                "lift_at_k": 1.0,
            }
        )
    return pd.DataFrame(rows)
