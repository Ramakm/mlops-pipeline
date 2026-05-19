"""
Model Training — fits a Logistic Regression, logs params / metrics /
artefacts to MLflow, and registers the model in the MLflow Model Registry.
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
import joblib
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict, data: dict | None = None) -> tuple[LogisticRegression, str]:
    processed_path = config["data"]["processed_path"]

    if data is None:
        X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
        y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).squeeze()
        X_val   = pd.read_csv(os.path.join(processed_path, "X_val.csv"))
        y_val   = pd.read_csv(os.path.join(processed_path, "y_val.csv")).squeeze()
    else:
        X_train, y_train = data["X_train"], data["y_train"]
        X_val,   y_val   = data["X_val"],   data["y_val"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    model_params = config["model"]["params"]

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started: {run_id}")

        # Log config params
        mlflow.log_params(model_params)
        mlflow.log_param("dataset", config["data"]["dataset_name"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))

        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        # Validation metrics
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "val_accuracy":  accuracy_score(y_val, y_pred),
            "val_precision": precision_score(y_val, y_pred, zero_division=0),
            "val_recall":    recall_score(y_val, y_pred, zero_division=0),
            "val_f1":        f1_score(y_val, y_pred, zero_division=0),
            "val_roc_auc":   roc_auc_score(y_val, y_prob),
        }
        mlflow.log_metrics(metrics)

        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # Save and log model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/logistic_regression.joblib")
        mlflow.log_artifact("models/logistic_regression.joblib")
        mlflow.log_artifact("models/scaler.joblib")
        mlflow.log_artifact("configs/config.yaml")

        # Register in Model Registry
        model_uri = f"runs:/{run_id}/logistic_regression.joblib"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression",
            registered_model_name=config["mlflow"]["model_name"],
        )

        logger.info(
            f"Model registered as '{config['mlflow']['model_name']}' "
            f"(run_id={run_id})"
        )

    return model, run_id


if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
