# scripts/universal_step.py
"""Execute a pipeline step specified by Hydra configuration.
This script reads/writes data, applies transformations, and runs tests as configured."""

import logging
from typing import Any, Callable

import hydra
import pandas as pd
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError

# Config schemas imports
from dependencies.config_schemas.RootConfig import RootConfig

# io imports
from dependencies.io.dataframe_to_parquet import dataframe_to_parquet

# Logging imports
from dependencies.io.parquet_to_dataframe import parquet_to_dataframe
from dependencies.logging_utils.log_cfg_job import log_cfg_job
from dependencies.logging_utils.log_function_call import log_function_call
from dependencies.logging_utils.setup_logging import setup_logging

# Metadata imports
from dependencies.metadata.calculate_metadata import calculate_and_save_metadata
from dependencies.modeling.ridge_optuna_trial import (
    RidgeOptunaTrialConfig,
    ridge_optuna_trial,
)

# Test imports
from dependencies.tests.check_required_columns import (
    CheckRequiredColumnsConfig,
    check_required_columns,
)
from dependencies.tests.check_row_count import CheckRowCountConfig, check_row_count

# Transformation imports
from dependencies.transformations.ingest_data import IngestDataConfig, ingest_data
from dependencies.transformations.remove_constant_and_high_corr_features import (
    RemoveConstantAndHighCorrFeaturesConfig,
    remove_constant_and_high_corr_features,
)

TRANSFORMATIONS: dict[str, dict[str, Any]] = {
    "ingest_data": {
        "transform": log_function_call(ingest_data),
        "Config": IngestDataConfig,
    },
    "remove_constant_and_high_corr_features": {
        "transform": log_function_call(remove_constant_and_high_corr_features),
        "Config": RemoveConstantAndHighCorrFeaturesConfig,
    },
    "ridge_optuna_trial": {
        "transform": log_function_call(ridge_optuna_trial),
        "Config": RidgeOptunaTrialConfig,
    },
    "ridge_optuna_trial_2": {
        "transform": log_function_call(ridge_optuna_trial),
        "Config": RidgeOptunaTrialConfig,
    },
}

TESTS: dict[str, dict[str, Any]] = {
    "check_required_columns": {
        "test": log_function_call(check_required_columns),
        "Config": CheckRequiredColumnsConfig,
    },
    "check_row_count": {
        "test": log_function_call(check_row_count),
        "Config": CheckRowCountConfig,
    },
}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def universal_step(cfg: RootConfig) -> None:
    """
    Orchestrate a universal pipeline step using the provided RootConfig:
    1) Identify the transformation to run.
    2) Optionally read data from Parquet.
    3) Validate transformation output if it should be a DataFrame.
    4) Execute configured tests on the resulting data.
    5) Optionally write the resulting data and metadata.
    """
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    log_cfg_job_flag = cfg.logging_utils.log_cfg_job.log_for_each_step
    if log_cfg_job_flag:
        logger.info("Override: 'log_cfg_job_flag' set to %s", bool(log_cfg_job_flag))
        log_cfg_job(cfg)
    else:
        logger.debug(
            "Not logging cfg job: 'log_cfg_job_flag' == %s",
            bool(log_cfg_job_flag),
        )

    read_input = cfg.io_policy.READ_INPUT
    write_output = cfg.io_policy.WRITE_OUTPUT

    # Convert Hydra configs to dict
    transform_config = OmegaConf.to_container(cfg.transformations, resolve=True)
    read_params = OmegaConf.to_container(
        cfg.utility_functions.utility_function_read, resolve=True
    )
    write_params = OmegaConf.to_container(
        cfg.utility_functions.utility_function_write, resolve=True
    )
    meta_params = OmegaConf.to_container(
        cfg.utility_functions.utility_function_metadata, resolve=True
    )
    try:
        tests_config = OmegaConf.to_container(cfg.tests, resolve=True)
    except InterpolationKeyError as e:
        logger.error("Failed to resolve tests_config: %s", str(e))
        tests_config = {}

    return_type = transform_config.get("return_type", None)
    transform_name = cfg.setup.get("script_base_name", None)

    if transform_name not in TRANSFORMATIONS:
        logger.error("Unrecognized transform: '%s'", transform_name)
        return

    step_info = TRANSFORMATIONS[transform_name]
    step_fn = step_info["transform"]
    step_cls = step_info.get("Config", None)

    step_params = transform_config.get(transform_name, None)
    if read_input:
        df = parquet_to_dataframe(**read_params)
    else:
        pass
    if step_cls is None:
        df = step_fn() if return_type == "df" else None
    else:
        cfg_obj = step_cls(**step_params)
        returned_value = (
            step_fn(df, **cfg_obj.__dict__)
            if read_input
            else step_fn(**cfg_obj.__dict__)
        )
        if return_type == "df" and returned_value is not None:
            if not isinstance(returned_value, pd.DataFrame):
                logger.error("%s did not return a DataFrame.", transform_name)
                raise TypeError
            df = returned_value

    if write_output:
        for test_key, test_dict in TESTS.items():
            if transform_config.get(test_key, False):
                test_fn: Callable[..., pd.DataFrame] = test_dict["test"]
                test_params_dict = tests_config.get(test_key, {})
                df = test_fn(df, **test_params_dict)

        dataframe_to_parquet(df, **write_params)
        calculate_and_save_metadata(df, **meta_params)

    logger.info("Successfully executed step: %s", transform_name)


if __name__ == "__main__":
    universal_step()
