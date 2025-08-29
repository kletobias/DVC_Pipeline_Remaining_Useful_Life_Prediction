# dependencies/io/parquet_to_dataframe.py
"""Reads a parquet file and returns a pandas DataFrame."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parquet_to_dataframe(input_file_path: str) -> pd.DataFrame:
    """Reads a parquet file and returns a pandas DataFrame.
    Args:
        input_file_path (str): Path to the parquet file.
        low_memory (bool): If True, uses low memory mode. Default is False.
    Returns:
        pd.DataFrame: DataFrame containing the data from the parquet file.
    """
    try:
        df = pd.read_parquet(input_file_path)
        logger.info("Successfully read parquet file %s", input_file_path)
        return df
    except Exception as e:
        logger.error("Error reading parquet file %s: %s", input_file_path, e)
        raise
