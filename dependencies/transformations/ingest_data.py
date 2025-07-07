# dependencies/transformations/ingest_data.py
import logging
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from omegaconf import MISSING

logger = logging.getLogger(__name__)


@dataclass
class IngestDataConfig:
    """
    Configuration for the IngestData class.
    """

    input_file_path: str = MISSING
    suffix: str = MISSING


def ingest_data(
    input_file_path: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Ingests data from an HDF5 file and saves it as Parquet files.

    Args:
        input_file_path (str): Path to the input HDF5 file.
        suffix (str): Suffix to identify the data split.
                    Should be either '_dev', '_test'.
    Returns:
        pd.DataFrame: DataFrame containing the ingested data.
    """

    with h5py.File(input_file_path, "r") as hdf:
        logger.debug("input_file_path: %s", input_file_path)
        logger.debug("suffix: %s", suffix)
        if suffix == "_dev":
            logger.info("Reading dev data from HDF5 file")
            # dev (train) / test splits
            W_dev = hdf["W_dev"][:]
            Xs_dev = hdf["X_s_dev"][:]
            Xv_dev = hdf["X_v_dev"][:]
            T_dev = hdf["T_dev"][:]
            Y_dev = hdf["Y_dev"][:]
            A_dev = hdf["A_dev"][:]
        elif suffix == "_test":
            logger.info("Reading test data from HDF5 file")
            W_test = hdf["W_test"][:]
            Xs_test = hdf["X_s_test"][:]
            Xv_test = hdf["X_v_test"][:]
            T_test = hdf["T_test"][:]
            Y_test = hdf["Y_test"][:]
            A_test = hdf["A_test"][:]

        W_var = hdf["W_var"][:].astype("U32").tolist()
        Xs_var = hdf["X_s_var"][:].astype("U32").tolist()
        Xv_var = hdf["X_v_var"][:].astype("U32").tolist()
        T_var = hdf["T_var"][:].astype("U32").tolist()
        A_var = hdf["A_var"][:].astype("U32").tolist()

    cols = A_var + W_var + Xs_var + Xv_var + T_var + ["RUL"]

    # build dataframe
    if suffix == "_dev":
        logger.debug("Creating dev DataFrame")
        df = pd.DataFrame(
            np.hstack([A_dev, W_dev, Xs_dev, Xv_dev, T_dev, Y_dev]),
            columns=cols,
        )

    elif suffix == "_test":
        logger.debug("Creating test DataFrame")
        df = pd.DataFrame(
            np.hstack([A_test, W_test, Xs_test, Xv_test, T_test, Y_test]),
            columns=cols,
        )

    else:
        logger.error("Invalid suffix: %s, must be one of '_dev', or '_test'", suffix)
        raise ValueError

    return df
