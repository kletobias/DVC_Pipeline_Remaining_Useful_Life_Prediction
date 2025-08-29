# dependencies/modeling/prepare_features_targets_groups.py
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_features_targets_groups(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
    list[int],
    list[int],
]:
    """Return X/y for train-test plus group array and lists of unit IDs."""
    if "unit" not in df_train.columns:
        logger.error("Column 'unit' missing in training data")
        raise KeyError

    feature_cols = [
        c for c in df_train.columns if c not in [target_col, "unit", "index"]
    ]
    if (
        df_test[feature_cols].columns.tolist()
        != df_train[feature_cols].columns.tolist()
    ):
        logger.error("Feature columns in train and test data do not match")
        raise ValueError

    X_train = pd.DataFrame(df_train[feature_cols], index=df_train.index)
    X_test = pd.DataFrame(df_test[feature_cols], index=df_test.index)
    y_train = pd.Series(df_train[target_col], index=df_train.index)
    y_test = pd.Series(df_test[target_col], index=df_test.index)
    groups_train = df_train["unit"].astype(int).to_numpy()

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        groups_train,
        sorted(np.unique(groups_train)),
        sorted(df_test["unit"].astype(int).unique()),
    )
