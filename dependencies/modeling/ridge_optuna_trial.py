"""Ridge Optuna Trial - Public stub
This stub imports the proprietary implementation from the rul-core-ip package.
"""

try:
    from rul_core_ip.modeling.ridge_optuna_trial import ridge_optuna_trial

    # Re-export the function
    __all__ = ["ridge_optuna_trial"]

except ImportError:

    def ridge_optuna_trial(*args, **kwargs):  # type: ignore[misc] # noqa: ARG001
        msg = (
            "RUL Core IP package not installed. Install with: pip install ./rul_core_ip"
        )
        raise ImportError(msg)
