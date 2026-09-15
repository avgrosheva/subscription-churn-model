"""Model training: Logistic Regression baseline and CatBoost main model.

Both models are fit on train only. CatBoost uses validation as its
eval_set for early stopping during hyperparameter/model selection. Once
hyperparameters (including the early-stopped iteration count) are frozen,
the final CatBoost estimator -- and its probability calibration -- are fit
using only train data (calibration via k-fold cross-validation on train).
Validation is reserved for model comparison and diagnostics after that
point; test is never touched by any function here.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, Pool
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


def freeze_catboost_params(best_params: dict, best_iteration: int, class_weights=None) -> dict:
    """Freeze hyperparameters selected via train/validation early stopping into
    a fixed, no-early-stopping configuration for the final estimator.

    `best_iteration` is CatBoost's 0-indexed best iteration from early
    stopping; the equivalent fixed tree count is best_iteration + 1.
    """
    frozen = dict(
        depth=best_params.get("depth", 6),
        learning_rate=best_params.get("learning_rate", 0.1),
        iterations=best_iteration + 1,
        loss_function="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
    )
    if class_weights is not None:
        frozen["class_weights"] = class_weights
    return frozen


def fit_frozen_catboost(X_train, y_train, categorical_cols, frozen_params: dict) -> CatBoostClassifier:
    """Fit CatBoost with fixed hyperparameters/iterations, on train only, no eval_set.

    This is the "raw" (uncalibrated) final model: used for SHAP and as the
    before-calibration baseline in diagnostic comparisons.
    """
    model = CatBoostClassifier(cat_features=categorical_cols, **frozen_params)
    model.fit(X_train, y_train)
    return model


class TrainCVCalibratedCatBoost:
    """CatBoost with probability calibration fit by k-fold cross-validation on
    TRAIN ONLY.

    `sklearn.calibration.CalibratedClassifierCV` cannot be used here: it
    calls `sklearn.base.clone()` on the base estimator, and CatBoost's
    sklearn wrapper does not round-trip `cat_features` through
    `get_params`/`set_params` cleanly enough to satisfy scikit-learn's
    (relatively new, stricter) clone check. This class replicates the same
    leakage-safe procedure by hand:

    1. Split train into `cv` stratified folds.
    2. For each fold, fit a fresh CatBoost model (frozen hyperparameters,
       no early stopping) on the other folds and predict on the held-out
       fold, producing out-of-fold (OOF) probabilities that were never seen
       by the model that produced them.
    3. Fit a calibration map (sigmoid = logistic regression on the OOF
       probability, or isotonic regression) on those OOF probabilities vs.
       the true training labels.
    4. Refit CatBoost once on the *entire* training set (for the strongest
       possible final model) and apply the calibration map to its
       predictions at inference time.

    Neither validation nor test labels are used anywhere in this
    procedure; validation is reserved for model comparison and diagnostics.
    """

    def __init__(self, categorical_cols, frozen_params: dict, method: str = "sigmoid", cv: int = 5):
        self.categorical_cols = categorical_cols
        self.frozen_params = frozen_params
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        X = X.reset_index(drop=True)
        y = np.asarray(y)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_SEED)

        oof_proba = np.zeros(len(y))
        for fold_train_idx, fold_hold_idx in skf.split(X, y):
            fold_model = CatBoostClassifier(cat_features=self.categorical_cols, **self.frozen_params)
            fold_model.fit(X.iloc[fold_train_idx], y[fold_train_idx])
            oof_proba[fold_hold_idx] = fold_model.predict_proba(X.iloc[fold_hold_idx])[:, 1]
        self.oof_proba_ = oof_proba

        if self.method == "sigmoid":
            self.calibrator_ = LogisticRegression()
            self.calibrator_.fit(oof_proba.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator_.fit(oof_proba, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.base_model_ = CatBoostClassifier(cat_features=self.categorical_cols, **self.frozen_params)
        self.base_model_.fit(X, y)
        return self

    def predict_proba(self, X):
        raw = self.base_model_.predict_proba(X)[:, 1]
        if self.method == "sigmoid":
            calibrated = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator_.predict(raw)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1 - calibrated, calibrated])


def calibrate_via_train_cv(
    X_train, y_train, categorical_cols, frozen_params: dict, method: str = "sigmoid", cv: int = 5
) -> TrainCVCalibratedCatBoost:
    """Fit probability calibration using k-fold cross-validation on TRAIN ONLY.

    See `TrainCVCalibratedCatBoost` for the exact procedure. Neither
    validation nor test labels are used anywhere in this call.
    """
    model = TrainCVCalibratedCatBoost(categorical_cols, frozen_params, method=method, cv=cv)
    model.fit(X_train, y_train)
    return model
