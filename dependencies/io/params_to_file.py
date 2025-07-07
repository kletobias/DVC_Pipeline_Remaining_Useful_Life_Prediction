# dependencies/io/params_to_file.py
import logging
import os
from typing import Any

from omegaconf import OmegaConf

from dependencies.general.mkdir_if_not_exists import mkdir_if_not_exists

logger = logging.getLogger(__name__)


def params_to_file(
    params_to_export: list | dict[Any, Any],
    output_params_file_path: str,
) -> None:
    mkdir_if_not_exists(os.path.dirname(output_params_file_path))
    logger.debug("Output params file path: %s", output_params_file_path)
    OmegaConf.save(params_to_export, output_params_file_path)
    logger.info("Exported params to YAML using filepath: %s", output_params_file_path)
