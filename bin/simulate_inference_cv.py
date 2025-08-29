#!/usr/bin/env python
"""Inference simulation - Public stub
This stub imports the proprietary implementation from the rul-core-ip package.
"""

import sys

try:
    from rul_core_ip.inference import main

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
