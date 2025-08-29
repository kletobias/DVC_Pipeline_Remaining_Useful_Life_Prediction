"""DVC YAML Generator - Public stub
This stub imports the proprietary implementation from the rul-core-ip package.
"""

try:
    from rul_core_ip.template_generator import generate_dvc_yaml_core

    # Re-export the function
    __all__ = ["generate_dvc_yaml_core"]

except ImportError:

    def generate_dvc_yaml_core(*args, **kwargs):  # type: ignore[misc] # noqa: ARG001
        msg = (
            "RUL Core IP package not installed. Install with: pip install ./rul_core_ip"
        )
        raise ImportError(msg)
