# dependencies/io/dataframe_to_csv.py
import logging
import os

import pandas as pd

from dependencies.general.mkdir_if_not_exists import mkdir_if_not_exists

logger = logging.getLogger(__name__)


def dataframe_to_parquet(
    df: pd.DataFrame,
    output_file_path: str,
    include_index: bool = False,
) -> None:
    mkdir_if_not_exists(os.path.dirname(output_file_path))
    logger.debug("Output PARQUET file path: %s", output_file_path)
    df.to_parquet(output_file_path, index=include_index)
    logger.info("Exported df to parquet using filepath: %s", output_file_path)
