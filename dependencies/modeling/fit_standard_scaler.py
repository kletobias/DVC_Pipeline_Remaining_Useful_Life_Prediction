# dependencies/modeling/fit_standard_scaler.py
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_standard_scaler(
    X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training data and transform both train and test data."""
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    X_train = pd.DataFrame(
        scaler.transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index
    )

    return X_train, X_test, scaler
