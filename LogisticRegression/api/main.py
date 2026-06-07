"""
FastAPI serving layer — exposes /predict, /predict/batch,
/health, and /metrics endpoints.
"""

import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from loguru import logger
import yaml

from pipelines.inference_pipeline import InferencePipeline


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Pydantic schemas ────────────────────────────────────────────────────────

class FeatureInput(BaseModel):
    mean_radius: float = Field(..., example=14.0)
    mean_texture: float = Field(..., example=19.0)
    mean_perimeter: float = Field(..., example=91.0)
    mean_area: float = Field(..., example=654.0)
    mean_smoothness: float = Field(..., example=0.097)
    mean_compactness: float = Field(..., example=0.106)
    mean_concavity: float = Field(..., example=0.100)
    mean_concave_points: float = Field(..., example=0.048)
    mean_symmetry: float = Field(..., example=0.181)
    mean_fractal_dimension: float = Field(..., example=0.063)
    radius_error: float = Field(..., example=0.35)
    texture_error: float = Field(..., example=1.19)
    perimeter_error: float = Field(..., example=2.40)
    area_error: float = Field(..., example=27.0)
    smoothness_error: float = Field(..., example=0.006)
    compactness_error: float = Field(..., example=0.019)
    concavity_error: float = Field(..., example=0.025)
    concave_points_error: float = Field(..., example=0.008)
    symmetry_error: float = Field(..., example=0.020)
    fractal_dimension_error: float = Field(..., example=0.003)
    worst_radius: float = Field(..., example=17.3)
    worst_texture: float = Field(..., example=27.2)
    worst_perimeter: float = Field(..., example=116.0)
    worst_area: float = Field(..., example=942.0)
    worst_smoothness: float = Field(..., example=0.148)
    worst_compactness: float = Field(..., example=0.265)
    worst_concavity: float = Field(..., example=0.327)
    worst_concave_points: float = Field(..., example=0.120)
    worst_symmetry: float = Field(..., example=0.290)
    worst_fractal_dimension: float = Field(..., example=0.088)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probabilities: dict[str, float]
    latency_ms: float


# ── Lifespan ────────────────────────────────────────────────────────────────

pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    cfg = load_config()
    pipeline = InferencePipeline(cfg, use_registry=False)
    logger.info("Model loaded and API ready.")
    yield
    logger.info("Shutting down API.")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="MLOps-grade Logistic Regression model serving endpoint",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/metrics")
def metrics():
    """Returns latest evaluation metrics from reports/metrics.json."""
    import json
    try:
        with open("reports/metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics not found. Run evaluation first.")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: FeatureInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    features = payload.model_dump()

    # Rename keys to match sklearn feature names (spaces)
    key_map = {k: k.replace("_", " ") for k in features}
    renamed = {key_map[k]: v for k, v in features.items()}

    result = pipeline.predict(renamed)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(**result, latency_ms=round(latency_ms, 2))


@app.post("/predict/batch")
def predict_batch(payloads: list[FeatureInput]):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = []
    for p in payloads:
        features = p.model_dump()
        key_map = {k: k.replace("_", " ") for k in features}
        rows.append({key_map[k]: v for k, v in features.items()})

    df = pd.DataFrame(rows)
    results_df = pipeline.predict_batch(df)
    return results_df.to_dict(orient="records")
