"""
Model Registry helpers — thin wrappers around the MLflow Model Registry
for promoting, demoting, and loading models by stage.
"""

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _client(config: dict) -> MlflowClient:
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    return MlflowClient()


def get_latest_version(config: dict, stage: str = "None") -> str:
    client = _client(config)
    model_name = config["mlflow"]["model_name"]
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise ValueError(f"No model version found in stage '{stage}' for '{model_name}'")
    return versions[0].version


def transition_model(config: dict, version: str, target_stage: str) -> None:
    client = _client(config)
    model_name = config["mlflow"]["model_name"]
    client.transition_model_version_stage(
        name=model_name, version=version, stage=target_stage
    )
    logger.info(f"Model '{model_name}' v{version} → {target_stage}")


def load_production_model(config: dict):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["model_name"]
    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading model from registry: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


def promote_to_staging(config: dict) -> None:
    version = get_latest_version(config, stage="None")
    transition_model(config, version, "Staging")


def promote_to_production(config: dict) -> None:
    version = get_latest_version(config, stage="Staging")
    transition_model(config, version, "Production")


def archive_production(config: dict) -> None:
    version = get_latest_version(config, stage="Production")
    transition_model(config, version, "Archived")


if __name__ == "__main__":
    cfg = load_config()
    promote_to_staging(cfg)
