"""RUL Core IP Package - Proprietary implementation"""

__version__ = "1.0.0"

from .inference import main as inference_main
from .template_generator import generate_dvc_yaml_core

__all__ = ["generate_dvc_yaml_core", "inference_main"]
