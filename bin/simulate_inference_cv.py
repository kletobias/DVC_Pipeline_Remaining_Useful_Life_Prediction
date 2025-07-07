# bin/simulate_inference_cv.py
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import boto3
import hydra
import joblib
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def get_bucket_from_tfstate(tfstate_path: Path, key: str) -> str | None:
    if not tfstate_path.exists():
        logger.warning("tfstate file %s not found - skipping", tfstate_path)
        return None
    with tfstate_path.open() as f:
        state = json.load(f)
    return state.get("outputs", {}).get(key, {}).get("value")


@dataclass
class SimulateInferenceMlflowConfig:
    terraform_state_file: str
    tracking_uri: str
    input_data_is_s3_bucket: bool
    bucket_key_in_pt_0: str | None
    bucket_key_in_pt_1: str
    input_data_file_download_file_path: str
    input_data_file_path: str | None
    model_uri: str
    scaler_uri: str
    rmse_threshold: float
    output_data_is_s3_bucket: bool
    bucket_key_out_pt_0: str | None
    bucket_key_out_pt_1: str
    output_data_file_path: str
    feature_names: list[str]
    target_name: str


def _download_from_s3(s3_uri: str, local_path: str) -> None:
    """s3://bucket/key -> local_path"""
    _, bucket, *key_parts = s3_uri.split("/", 3)
    key = "/".join(key_parts)
    boto3.client("s3").download_file(bucket, key, local_path)


def load_scaler(scaler_uri: str) -> StandardScaler:
    if scaler_uri.startswith("s3://"):
        tmp_path = "/tmp/scaler.pkl"
        _download_from_s3(scaler_uri, tmp_path)
        return joblib.load(tmp_path)  # type: ignore[return-value]
    if scaler_uri.startswith(("runs:", "models:")):
        local_path = mlflow.artifacts.download_artifacts(scaler_uri)
        return joblib.load(local_path)  # type: ignore[return-value]
    return joblib.load(scaler_uri)  # type: ignore[return-value]


def simulate_inference_mlflow(
    *,
    input_data_file_path: str,
    input_data_is_s3_bucket: bool,
    bucket_key_in_pt_0: str,
    bucket_key_in_pt_1: str,
    input_data_file_download_file_path: str,
    model_uri: str,
    scaler_uri: str,
    rmse_threshold: float,
    output_data_is_s3_bucket: bool,
    bucket_key_out_pt_0: str,
    bucket_key_out_pt_1: str,
    output_data_file_path: str,
    feature_names: list[str],
    target_name: str,
    run_mode: str,
) -> None:
    logger.info("starting inference (%s mode)…", run_mode)

    if input_data_is_s3_bucket:
        boto3.client("s3").download_file(
            bucket_key_in_pt_0, bucket_key_in_pt_1, input_data_file_download_file_path
        )
        df = pd.read_parquet(input_data_file_download_file_path)
    else:
        df = pd.read_parquet(input_data_file_path)

    model = mlflow.pyfunc.load_model(model_uri)
    mlflow.pyfunc.get_model_dependencies(model_uri)
    scaler: StandardScaler = load_scaler(scaler_uri)

    X_scaled = pd.DataFrame(
        scaler.transform(df[feature_names]), columns=feature_names, index=df.index
    )
    preds_log1p = model.predict(X_scaled)
    preds_original = np.expm1(preds_log1p)
    rmse = np.sqrt(mean_squared_error(df[target_name], preds_original))
    logger.info("RMSE %.4f", rmse)

    df[f"{target_name}_log1p"] = preds_log1p
    df[f"{target_name}_original"] = preds_original

    if output_data_is_s3_bucket:
        df.to_parquet(output_data_file_path, index=False)
        boto3.client("s3").upload_file(
            output_data_file_path, bucket_key_out_pt_0, bucket_key_out_pt_1
        )
    else:
        df.to_parquet(output_data_file_path, index=False)

    if rmse > rmse_threshold:
        logger.warning("RMSE %.2f exceeds threshold %.2f", rmse, rmse_threshold)

    assert rmse <= 10, f"Drift detected: RMSE={rmse:.2f}"

    logger.info("inference completed")


@hydra.main(version_base=None, config_path="../configs/inference", config_name="cv")
def main(cfg: DictConfig) -> None:
    project_root = Path(os.getenv("PROJECT_ROOT", "."))
    run_cfg = cfg.simulate_inference_mlflow
    if cfg.get("use_aws", False):
        tfstate_path = project_root / cfg.simulate_inference_mlflow.terraform_state_file

        dvc_bucket = (
            os.getenv("DVC_BUCKET")
            or get_bucket_from_tfstate(tfstate_path, "dvc_bucket")
            or ""
        )
        mlflow_bucket = (
            os.getenv("MLFLOW_BUCKET")
            or get_bucket_from_tfstate(tfstate_path, "mlflow_bucket")
            or ""
        )
        run_cfg.bucket_key_in_pt_0 = dvc_bucket
        run_cfg.bucket_key_out_pt_0 = mlflow_bucket
        os.environ["MLFLOW_S3_BUCKET"] = mlflow_bucket

        run_cfg.input_data_file_download_file_path = str(
            project_root / run_cfg.input_data_file_download_file_path
        )

    run_cfg.output_data_file_path = str(project_root / run_cfg.output_data_file_path)

    if run_cfg.tracking_uri:
        mlflow.set_tracking_uri(run_cfg.tracking_uri)

    simulate_inference_mlflow(
        input_data_file_path=run_cfg.input_data_file_path,
        input_data_is_s3_bucket=run_cfg.input_data_is_s3_bucket,
        bucket_key_in_pt_0=run_cfg.bucket_key_in_pt_0,
        bucket_key_in_pt_1=run_cfg.bucket_key_in_pt_1,
        input_data_file_download_file_path=run_cfg.input_data_file_download_file_path,
        model_uri=run_cfg.model_uri,
        scaler_uri=run_cfg.scaler_uri,
        rmse_threshold=float(run_cfg.rmse_threshold),
        output_data_is_s3_bucket=run_cfg.output_data_is_s3_bucket,
        bucket_key_out_pt_0=run_cfg.bucket_key_out_pt_0,
        bucket_key_out_pt_1=run_cfg.bucket_key_out_pt_1,
        output_data_file_path=run_cfg.output_data_file_path,
        feature_names=run_cfg.feature_names,
        target_name=run_cfg.target_name,
        run_mode=os.getenv("RUN_MODE", "full"),
    )


if __name__ == "__main__":
    main()
