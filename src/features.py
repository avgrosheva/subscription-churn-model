"""Feature definitions and preprocessing pipelines.

Numeric and categorical column lists are derived once from the cleaned
training data (see data.get_feature_columns) so that every model consumes
the same feature set. Preprocessing objects are always fit on training data
only; callers must not fit on validation or test data.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logreg_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer for Logistic Regression: scale numeric, one-hot categorical.

    OneHotEncoder drops the first level of each category to avoid perfect
    collinearity in a linear model, and ignores unseen categories at
    inference time so a pipeline fit on train never breaks on val/test.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )


def build_logreg_pipeline(numeric_cols: list[str], categorical_cols: list[str], **logreg_kwargs) -> Pipeline:
    from sklearn.linear_model import LogisticRegression

    preprocessor = build_logreg_preprocessor(numeric_cols, categorical_cols)
    logreg_defaults = dict(max_iter=5000, solver="lbfgs", random_state=42)
    logreg_defaults.update(logreg_kwargs)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**logreg_defaults)),
        ]
    )


def catboost_cat_feature_indices(feature_cols: list[str], categorical_cols: list[str]) -> list[int]:
    """Positional indices of categorical columns within feature_cols, for catboost.Pool."""
    return [feature_cols.index(c) for c in categorical_cols]


def select_X_y(df: pd.DataFrame, feature_cols: list[str], target: str):
    return df[feature_cols], df[target]
