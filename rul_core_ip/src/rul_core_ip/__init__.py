"""RUL Core IP Package - Proprietary implementation"""

__version__ = "1.0.0"

from .inference import run_inference
from .template_generator import generate_dvc_yaml_core

__all__ = ["generate_dvc_yaml_core", "run_inference"]
