#!/usr/bin/env python
"""Inference simulation - Public stub
This stub imports the proprietary implementation from the rul-core-ip package.
"""

import sys

import hydra
from omegaconf import DictConfig

try:
    from rul_core_ip.inference import run_inference

    @hydra.main(version_base=None, config_path="../configs/inference", config_name="cv")
    def main(cfg: DictConfig) -> None:
        """Main entry point for Hydra config handling."""
        run_inference(cfg)

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    import sys

    msg = (
        "Error: RUL Core IP package not installed.\n"
        "Install with: pip install ./rul_core_ip\n"
        f"Details: {e}"
    )
    sys.stderr.write(msg + "\n")
    sys.exit(1)
