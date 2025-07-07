# dependencies/modeling/ridge_optuna_trial.py
import json
import logging
import os
from dataclasses import dataclass
from math import sqrt
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.utils.autologging_utils import INPUT_EXAMPLE_SAMPLE_ROWS
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dependencies.io.parquet_to_dataframe import parquet_to_dataframe
from dependencies.logging_utils.calculate_and_log_importances_as_artifact import (
    calculate_and_log_importances_as_artifact,
)
from dependencies.modeling.compute_rmse import compute_rmse
from dependencies.modeling.fit_standard_scaler import fit_standard_scaler
from dependencies.modeling.leave_one_group_out_cv import leave_one_group_out_cv
from dependencies.modeling.optuna_random_search_util import optuna_random_search_util
from dependencies.modeling.prepare_features_targets_groups import (
    prepare_features_targets_groups,
)
from dependencies.modeling.ridge_sklearn_instantiate_ridge_class import (
    ridge_sklearn_instantiate_ridge_class,
)
from dependencies.modeling.validate_parallelism import validate_parallelism

logger = logging.getLogger(__name__)


@dataclass
class RidgeOptunaTrialConfig:
    target_col: str
    train_file_path: str
    test_file_path: str
    n_trials: int
    permutation_importances_filename: str
    hyperparameters: dict[str, Any]
    n_jobs_study: int
    n_jobs_cv: int
    random_state: int
    experiment_tags: dict[str, Any]
    experiment_name: str = "ridge_optuna_trial"


def ridge_optuna_trial(
    target_col: str,
    train_file_path: str,
    test_file_path: str,
    n_trials: int,
    permutation_importances_filename: str,
    hyperparameters: dict[str, Any],
    n_jobs_study: int,
    n_jobs_cv: int,
    random_state: int,
    experiment_tags: dict[str, Any],
    experiment_name: str,
) -> None:
    """
    Single entry-point for DVC stage:
      - Ridge model, log-transform on target,
      - standard scaling,
      - Leave-One-Group-Out cross-validation on the training set.
      - Evaluate on a separate hold-out test set for final metrics.

    Logs:
      - Cross-val fold predictions to X_test_and_y_pred_val.json (training folds).
      - Final test predictions to final_test_predictions.json.
      - "cv_rmse" & "cv_r2" for cross-validation in a nested "train" run (per trial).
      - "train_rmse", "train_mae", "test_rmse", "test_mae" in final_model run.
      - Also logs "rmse" as the same as "test_rmse" for a single "rmse" metric in the
        UI.
    """

    validate_parallelism(n_jobs_cv=n_jobs_cv, n_jobs_study=n_jobs_study)

    # ---------------------- 1. Load data ----------------------
    df_train = parquet_to_dataframe(train_file_path)
    df_test = parquet_to_dataframe(test_file_path)
    logger.debug(
        "Train set read from %s, Test set read from %s", train_file_path, test_file_path
    )
    logger.info("Starting ridge_optuna_trial with %i trials to run", n_trials)

    # ---------------------- 2. Prepare splits & scale ----------------------
    (
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw,
        groups_train,
        _,
        _,
    ) = prepare_features_targets_groups(df_train, df_test, target_col)

    X_train_scaled, X_test_scaled, scaler_obj = fit_standard_scaler(
        X_train_raw, X_test_raw
    )
    joblib.dump(scaler_obj, "scaler.pkl")

    # Log-transform the target
    y_train = y_train_raw.apply(np.log1p)
    y_test = y_test_raw.apply(np.log1p)

    # Create LOGO splitter
    logo = leave_one_group_out_cv()
    fold_unit_mapping = {
        str(i): int(np.unique(groups_train[test_idx])[0])
        for i, (_, test_idx) in enumerate(
            logo.split(X_train_raw, y_train, groups_train)
        )
    }

    # ---------------------- 3. MLflow setup ----------------------
    mlflow.set_tracking_uri("file:./mlruns")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        experiment_id = exp.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Attach tags to the experiment so they're visible at the experiment level
    experiment_tags["fold_unit_mapping"] = json.dumps(fold_unit_mapping)
    for tag, value in experiment_tags.items():
        mlflow.set_experiment_tag(tag, value)

    # ---------------------- 4. Optuna objective ----------------------
    def objective(trial: optuna.Trial) -> float:
        """Perform cross-validation for a single set of hyperparameters."""
        ridge_params = optuna_random_search_util(trial, hyperparameters)
        ridge_estimator = ridge_sklearn_instantiate_ridge_class(ridge_params)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ridge_estimator),
            ]
        )

        fold_rmses = []
        fold_r2s = []

        for train_idx, test_idx in logo.split(X_train_raw, y_train, groups_train):
            X_train_cv = X_train_raw.iloc[train_idx]
            y_train_cv = y_train.iloc[train_idx]
            X_test_cv = X_train_raw.iloc[test_idx]
            y_test_cv = y_train.iloc[test_idx]

            pipe.fit(X_train_cv, y_train_cv)
            y_pred_cv_log = pipe.predict(X_test_cv)

            y_pred_cv = np.expm1(y_pred_cv_log)
            y_test_cv_orig = np.expm1(y_test_cv)

            # Per-fold metrics
            fold_rmse = compute_rmse(y_test_cv_orig, y_pred_cv)[0]
            fold_r2 = r2_score(y_test_cv_orig, y_pred_cv)
            fold_rmses.append(fold_rmse)
            fold_r2s.append(fold_r2)

        cv_rmse = float(np.mean(fold_rmses))
        cv_r2 = float(np.mean(fold_r2s))

        # Optional: Log a message about each trial
        completed = [
            t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            best_val = trial.study.best_value or float("inf")
            logger.info(
                "Trial %d => RMSE=%.3f R2=%.3f (Best so far=%.3f)",
                trial.number,
                cv_rmse,
                cv_r2,
                min(best_val, cv_rmse),
            )
        else:
            logger.info(
                "Trial %d => RMSE=%.3f R2=%.3f (No best yet)",
                trial.number,
                cv_rmse,
                cv_r2,
            )
        return cv_rmse

    # Create and run the study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs_study)

    # ---------------------- 5. Final model on entire train set ----------------------
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logger.warning("No completed trials found, skipping final model")
        return

    logger.info("Training final model on best_params")
    best_params = study.best_params
    logger.info("Best params: %s", best_params)

    final_model = Ridge(**best_params, random_state=random_state)
    final_model.fit(X_train_scaled, y_train)

    # Evaluate on training set (original scale)
    y_pred_train_log = final_model.predict(X_train_scaled)
    y_pred_train = np.expm1(y_pred_train_log)
    y_train_orig = np.expm1(y_train)
    train_rmse = sqrt(mean_squared_error(y_train_orig, y_pred_train))
    train_mae = mean_absolute_error(y_train_orig, y_pred_train)

    # Evaluate on test set (original scale)
    y_pred_test_log = final_model.predict(X_test_scaled)
    y_pred_test = np.expm1(y_pred_test_log)
    y_test_orig = np.expm1(y_test)
    test_rmse = sqrt(mean_squared_error(y_test_orig, y_pred_test))
    test_mae = mean_absolute_error(y_test_orig, y_pred_test)

    input_example = X_train_scaled.head(INPUT_EXAMPLE_SAMPLE_ROWS)
    output_example = pd.DataFrame(
        data=y_train[:INPUT_EXAMPLE_SAMPLE_ROWS], columns=[target_col]
    )

    # - 6. Turn on autologging only for the final model with MLflow -
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
    )

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="final_model",
        nested=True,
    ):
        mlflow.log_params(best_params)

        # Train metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_mae", float(train_mae))

        # Test metrics + unified 'rmse'
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", float(test_mae))
        mlflow.log_metric("rmse", test_rmse)

        # Summaries saved as JSON
        summary_metrics = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        }
        mlflow.log_dict(summary_metrics, artifact_file="metrics_predictions.json")

        # Log final test predictions
        final_preds_df = pd.DataFrame(
            {"y_pred_test": y_pred_test, "y_test": y_test_orig}
        )
        mlflow.log_table(final_preds_df, artifact_file="final_test_predictions.json")

        # Save the scaler + model
        mlflow.log_artifact("scaler.pkl", artifact_path="scaler")
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="RidgeModel",
            input_example=input_example,
            signature=infer_signature(
                model_input=input_example, model_output=output_example
            ),
        )

        # Permutation importances
        calculate_and_log_importances_as_artifact(
            permutation_importances_filename,
            final_model,
            X_train_scaled,
            y_train,
        )

    os.remove("scaler.pkl")
    logger.info("ridge_optuna_trial completed - all artifacts in ./mlruns/")
