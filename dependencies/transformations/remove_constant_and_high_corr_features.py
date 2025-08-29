# dependencies/transformations/remove_constant_and_high_corr_features.py
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from dependencies.io.params_to_file import params_to_file


@dataclass
class RemoveConstantAndHighCorrFeaturesConfig:
    threshold: float = 0.95
    unit_col_name: str = "unit"
    output_params_file_path: str = ""
    suffix: str = ""


logger = logging.getLogger(__name__)


def remove_constant_and_high_corr_features(
    df: pd.DataFrame,
    threshold: float,
    unit_col_name: str,
    output_params_file_path: str,
    suffix: str,
) -> pd.DataFrame:
    """
    - drop columns with zero variance (per-unit OR global)
    - drop one of each pair whose |corr| > threshold in every unit

    - returns a DataFrame with the unit column and the remaining columns
    """
    if suffix == "_dev":
        # ---- 1. remove zero-variance columns ------------------------------
        num_cols = df.select_dtypes(include="number").columns.difference(
            [unit_col_name]
        )
        is_const = (
            df.groupby(unit_col_name)[num_cols]
            .nunique()
            .le(1)  # ≤1 unique value per unit → constant
            .all()  # constant in every unit
        )
        keep = num_cols[~is_const]

        # ---- 2. high-corr pruning ----------------------------------------
        keep_mask = np.ones(keep.size, dtype=bool)
        for _, g in df.groupby(unit_col_name):
            corr = g[keep].corr().abs().values
            high = np.triu(corr > threshold, k=1)
            for i, j in zip(*np.where(high)):
                if keep_mask[j]:
                    keep_mask[j] = False if keep_mask[i] else keep_mask[j]

        cols_final = [unit_col_name, *keep[keep_mask].tolist()]
        params_to_file({"remaining_columns": cols_final}, output_params_file_path)
    elif suffix == "_test" and os.path.exists(output_params_file_path):
        logger.debug("suffix: %s", suffix)
        logger.debug("File output_params_file_path: %s exists", output_params_file_path)
        cols_final = list(OmegaConf.load(output_params_file_path)["remaining_columns"])
    else:
        if suffix not in ["_dev", "_test"]:
            logger.error("suffix: %s is not valid", suffix)
            raise ValueError
        if suffix == "_test" and not os.path.exists(output_params_file_path):
            logger.error(
                "File output_params_file_path: %s does not exist",
                output_params_file_path,
            )
            raise FileNotFoundError
        raise ValueError

    return df[cols_final]
