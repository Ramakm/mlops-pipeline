"""
Inference Pipeline — loads the scaler + model (from disk or MLflow
Model Registry) and returns predictions for a feature dict or DataFrame.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import mlflow.sklearn
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class InferencePipeline:
    def __init__(self, config: dict, use_registry: bool = False):
        self.config = config
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

        self.scaler = joblib.load("models/scaler.joblib")

        if use_registry:
            model_name = config["mlflow"]["model_name"]
            model_uri = f"models:/{model_name}/Production"
            logger.info(f"Loading model from registry: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
        else:
            self.model = joblib.load("models/logistic_regression.joblib")
            logger.info("Loaded model from local disk")

    def predict(self, features: dict | pd.DataFrame) -> dict:
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        else:
            X = features.copy()

        X_scaled = self.scaler.transform(X)
        prediction = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0].tolist()

        label_map = {0: "malignant", 1: "benign"}
        return {
            "prediction": prediction,
            "label": label_map[prediction],
            "probabilities": {
                "malignant": round(probabilities[0], 4),
                "benign":    round(probabilities[1], 4),
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        return pd.DataFrame(results)


if __name__ == "__main__":
    cfg = load_config()
    pipeline = InferencePipeline(cfg)

    # Demo prediction with dummy values
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer()
    sample = dict(zip(dataset.feature_names, dataset.data[0]))
    result = pipeline.predict(sample)
    logger.info(f"Sample prediction: {result}")
