"""Unit tests for data ingestion and preprocessing."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess


@pytest.fixture(scope="module")
def config():
    import yaml
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


class TestDataIngestion:
    def test_ingest_returns_dataframe_and_series(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")

        X, y = ingest_data(cfg)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_feature_count(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")

        X, _ = ingest_data(cfg)
        assert X.shape[1] == 30

    def test_target_is_binary(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")

        _, y = ingest_data(cfg)
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_values(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")

        X, y = ingest_data(cfg)
        assert X.isnull().sum().sum() == 0


class TestDataPreprocessing:
    def test_split_sizes_are_reasonable(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")
        cfg["data"]["processed_path"] = str(tmp_path / "processed")

        ingest_data(cfg)
        data = preprocess(cfg)

        total = len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"])
        assert total == 569  # full breast cancer dataset

    def test_scaler_zero_mean(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")
        cfg["data"]["processed_path"] = str(tmp_path / "processed")

        ingest_data(cfg)
        data = preprocess(cfg)

        means = data["X_train"].mean()
        assert (means.abs() < 0.1).all(), "Training features should be approx zero-mean"

    def test_target_alignment(self, config, tmp_path):
        cfg = config.copy()
        cfg["data"] = cfg["data"].copy()
        cfg["data"]["raw_path"] = str(tmp_path / "raw")
        cfg["data"]["processed_path"] = str(tmp_path / "processed")

        ingest_data(cfg)
        data = preprocess(cfg)

        assert len(data["X_train"]) == len(data["y_train"])
        assert len(data["X_test"]) == len(data["y_test"])
