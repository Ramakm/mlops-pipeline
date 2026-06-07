"""
Data Preprocessing — feature scaling, train/val/test split, saves
processed artefacts and a reference snapshot for drift monitoring.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess(config: dict) -> dict[str, pd.DataFrame | pd.Series]:
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    os.makedirs(processed_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X = pd.read_csv(os.path.join(raw_path, "features.csv"))
    y = pd.read_csv(os.path.join(raw_path, "target.csv")).squeeze()

    test_size = config["data"]["test_size"]
    val_size = config["data"]["validation_size"]
    seed = config["data"]["random_state"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Save splits
    for name, arr in [
        ("X_train", X_train_scaled), ("X_val", X_val_scaled), ("X_test", X_test_scaled),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        arr.to_csv(os.path.join(processed_path, f"{name}.csv"), index=False)

    # Reference snapshot for monitoring (first 100 training rows)
    reference = X_train_scaled.head(100).copy()
    reference["target"] = y_train.head(100).values
    reference.to_csv(os.path.join(processed_path, "reference.csv"), index=False)

    joblib.dump(scaler, "models/scaler.joblib")
    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    logger.info(f"Scaler saved to models/scaler.joblib")

    return {
        "X_train": X_train_scaled, "X_val": X_val_scaled, "X_test": X_test_scaled,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "scaler": scaler,
    }


if __name__ == "__main__":
    cfg = load_config()
    preprocess(cfg)
