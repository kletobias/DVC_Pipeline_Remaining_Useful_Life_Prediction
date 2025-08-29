# dependencies/validations/check_row_count.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.errors as pe


@dataclass
class CheckRowCountConfig:
    row_count: int = 0


def check_row_count(df: pd.DataFrame, row_count: int) -> pd.DataFrame:
    actual_row_count = df.shape[0]
    if row_count > 0 and actual_row_count != row_count:
        raise pe.SchemaError(
            schema=None,
            data=df,
            message=f"Row count mismatch: found {actual_row_count} \
                != expected {row_count}",
        )
    return df
