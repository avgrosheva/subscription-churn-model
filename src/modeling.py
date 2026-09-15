"""Model training: Logistic Regression baseline and CatBoost main model.

Both models are fit on train only. CatBoost uses validation as its
eval_set for early stopping. Test is never touched by any function here.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score

from src.features import build_logreg_pipeline, catboost_cat_feature_indices

RANDOM_SEED = 42


def fit_logreg(X_train, y_train, numeric_cols, categorical_cols, class_weight=None):
    pipe = build_logreg_pipeline(
        numeric_cols, categorical_cols, class_weight=class_weight
    )
    pipe.fit(X_train, y_train)
    return pipe


def choose_logreg_class_weight(X_train, y_train, X_val, y_val, numeric_cols, categorical_cols) -> dict:
    """Compare class_weight=None vs 'balanced' on validation PR-AUC; do not assume balancing helps."""
    results = {}
    for cw in (None, "balanced"):
        pipe = fit_logreg(X_train, y_train, numeric_cols, categorical_cols, class_weight=cw)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        results[str(cw)] = {
            "pipeline": pipe,
            "val_pr_auc": average_precision_score(y_val, val_proba),
        }
    best_key = max(results, key=lambda k: results[k]["val_pr_auc"])
    return {"results": results, "best_class_weight": best_key}


def fit_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict | None = None,
    class_weights=None,
) -> CatBoostClassifier:
    """Fit CatBoost with validation (not test) as the early-stopping eval_set."""
    cat_idx = catboost_cat_feature_indices(feature_cols, categorical_cols)

    train_pool = Pool(data=X_train[feature_cols], label=y_train, cat_features=cat_idx)
    val_pool = Pool(data=X_val[feature_cols], label=y_val, cat_features=cat_idx)

    default_params = dict(
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=RANDOM_SEED,
        early_stopping_rounds=50,
        iterations=2000,
        verbose=False,
    )
    if params:
        default_params.update(params)
    if class_weights is not None:
        default_params["class_weights"] = class_weights

    model = CatBoostClassifier(**default_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def small_catboost_grid_search(
    X_train, y_train, X_val, y_val, feature_cols, categorical_cols, grid: list[dict]
) -> list[dict]:
    """Evaluate a small, meaningful hyperparameter grid on validation PR-AUC only."""
    results = []
    for params in grid:
        model = fit_catboost(X_train, y_train, X_val, y_val, feature_cols, categorical_cols, params=params)
        val_proba = model.predict_proba(X_val[feature_cols])[:, 1]
        results.append(
            {
                "params": params,
                "best_iteration": model.get_best_iteration(),
                "val_pr_auc": average_precision_score(y_val, val_proba),
            }
        )
    return sorted(results, key=lambda r: -r["val_pr_auc"])


def calibrate_on_validation(fitted_estimator, X_val, y_val, method: str = "sigmoid"):
    """Wrap an already-fitted estimator with probability calibration fit on validation only.

    FrozenEstimator prevents CalibratedClassifierCV from refitting the base
    estimator; only the calibration map is learned, and only from
    validation data (never train or test).
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_estimator), method=method)
    calibrated.fit(X_val, y_val)
    return calibrated
