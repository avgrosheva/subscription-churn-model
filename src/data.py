"""Data loading, cleaning, and leakage-free splitting for the churn project.

All splitting happens exactly once, here, with a fixed random seed, so that
every notebook and script downstream shares the same train / validation /
test partition. No other module is allowed to re-split the data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn.csv"

TARGET_RAW = "Churn"
TARGET = "churn_flag"
ID_COL = "customerID"

# Split sizes: 60% train / 20% validation / 20% test.
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV, unmodified except for dtypes."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, leakage-free cleaning steps.

    TotalCharges is stored as a string in the raw file and contains 11 blank
    values. Every one of those blanks belongs to a customer with tenure == 0,
    i.e. a customer who has not yet been billed. We encode that as
    TotalCharges = 0, which matches the billing semantics (no billing period
    has elapsed) rather than treating it as a missing value to impute.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    blank_mask = df["TotalCharges"].isna()
    if not (df.loc[blank_mask, "tenure"] == 0).all():
        raise ValueError(
            "Found blank TotalCharges rows with tenure != 0; "
            "the zero-fill assumption no longer holds and must be revisited."
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df[TARGET] = (df[TARGET_RAW] == "Yes").astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) used as model inputs.

    customerID is an identifier with no predictive content and is excluded.
    Churn / churn_flag are the target and are excluded.
    """
    exclude = {ID_COL, TARGET_RAW, TARGET}
    numeric_cols = [
        c
        for c in df.select_dtypes(include=["int64", "float64"]).columns
        if c not in exclude
    ]
    categorical_cols = [
        c for c in df.select_dtypes(include=["object"]).columns if c not in exclude
    ]
    return numeric_cols, categorical_cols


def make_splits(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split, done exactly once.

    Test is held out first and touched nowhere else in the pipeline until
    final evaluation. Validation is used for all model/threshold/calibration
    decisions.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )
    relative_val_size = val_size / (train_size + val_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val[TARGET],
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def load_and_split(path: Path | str = DATA_PATH, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    df = clean(load_raw(path))
    return make_splits(df, seed=seed)
