"""Unit tests for model training and evaluation."""

import os
import sys
import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="module")
def config():
    import yaml
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sample_data():
    ds = load_breast_cancer()
    scaler = StandardScaler()
    X = scaler.fit_transform(ds.data)
    y = ds.target
    split = int(0.8 * len(X))
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_test":  X[split:], "y_test":  y[split:],
    }


class TestLogisticRegressionModel:
    def test_model_trains_without_error(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        assert model is not None

    def test_accuracy_above_threshold(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        accuracy = model.score(sample_data["X_test"], sample_data["y_test"])
        assert accuracy > 0.90, f"Accuracy {accuracy:.3f} below 90% threshold"

    def test_predict_returns_binary_labels(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        preds = model.predict(sample_data["X_test"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        proba = model.predict_proba(sample_data["X_test"])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_model_has_correct_number_of_features(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        assert model.n_features_in_ == 30

    def test_coefficient_shape(self, config, sample_data):
        params = config["model"]["params"]
        model = LogisticRegression(**params)
        model.fit(sample_data["X_train"], sample_data["y_train"])
        assert model.coef_.shape == (1, 30)
