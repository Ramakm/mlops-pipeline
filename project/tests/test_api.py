"""Integration tests for the FastAPI serving layer."""

import os
import sys
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SAMPLE_PAYLOAD = {
    "mean_radius": 14.0, "mean_texture": 19.0, "mean_perimeter": 91.0,
    "mean_area": 654.0, "mean_smoothness": 0.097, "mean_compactness": 0.106,
    "mean_concavity": 0.100, "mean_concave_points": 0.048,
    "mean_symmetry": 0.181, "mean_fractal_dimension": 0.063,
    "radius_error": 0.35, "texture_error": 1.19, "perimeter_error": 2.40,
    "area_error": 27.0, "smoothness_error": 0.006,
    "compactness_error": 0.019, "concavity_error": 0.025,
    "concave_points_error": 0.008, "symmetry_error": 0.020,
    "fractal_dimension_error": 0.003,
    "worst_radius": 17.3, "worst_texture": 27.2, "worst_perimeter": 116.0,
    "worst_area": 942.0, "worst_smoothness": 0.148,
    "worst_compactness": 0.265, "worst_concavity": 0.327,
    "worst_concave_points": 0.120, "worst_symmetry": 0.290,
    "worst_fractal_dimension": 0.088,
}


@pytest.fixture(scope="module")
def mock_pipeline():
    mock = MagicMock()
    mock.predict.return_value = {
        "prediction": 1,
        "label": "benign",
        "probabilities": {"malignant": 0.12, "benign": 0.88},
    }
    mock.predict_batch.return_value.__iter__ = iter
    return mock


@pytest.fixture(scope="module")
def client(mock_pipeline):
    with patch("api.main.InferencePipeline", return_value=mock_pipeline):
        # Patch module-level pipeline variable directly
        import api.main as app_module
        app_module.pipeline = mock_pipeline
        from api.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        data = response.json()
        assert "prediction" in data
        assert "label" in data
        assert "probabilities" in data
        assert "latency_ms" in data

    def test_predict_label_is_valid(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.json()["label"] in {"malignant", "benign"}

    def test_predict_missing_field_returns_422(self, client):
        incomplete = {k: v for k, v in list(SAMPLE_PAYLOAD.items())[:5]}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_probabilities_sum_to_one(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        probs = response.json()["probabilities"]
        total = probs["malignant"] + probs["benign"]
        assert abs(total - 1.0) < 0.01
