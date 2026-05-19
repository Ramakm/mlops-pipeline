"""
Data Ingestion — loads the Breast Cancer Wisconsin dataset and persists
raw CSV files for downstream pipeline stages.
"""

import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def ingest_data(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading Breast Cancer Wisconsin dataset from sklearn...")

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    raw_path = config["data"]["raw_path"]
    os.makedirs(raw_path, exist_ok=True)

    X.to_csv(os.path.join(raw_path, "features.csv"), index=False)
    y.to_csv(os.path.join(raw_path, "target.csv"), index=False)

    logger.info(f"Dataset shape: {X.shape} | Classes: {dataset.target_names.tolist()}")
    logger.info(f"Raw data saved to {raw_path}")

    return X, y


if __name__ == "__main__":
    cfg = load_config()
    ingest_data(cfg)
