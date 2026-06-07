# MLOps Logistic Regression — End-to-End Project

A production-grade MLOps pipeline for breast cancer classification using
**Logistic Regression**, demonstrating the complete lifecycle from raw data
to monitored production serving.

---

## Table of Contents

1. [What is MLOps?](#what-is-mlops)
2. [MLOps Maturity Levels](#mlops-maturity-levels)
3. [Project Architecture](#project-architecture)
4. [MLOps Concepts Implemented](#mlops-concepts-implemented)
   - [Data Management](#1-data-management)
   - [Experiment Tracking](#2-experiment-tracking-mlflow)
   - [Model Registry](#3-model-registry)
   - [CI/CD Pipeline](#4-cicd-pipeline)
   - [Model Serving](#5-model-serving)
   - [Monitoring & Drift Detection](#6-monitoring--drift-detection)
   - [Containerisation](#7-containerisation)
   - [Testing Strategy](#8-testing-strategy)
   - [Configuration Management](#9-configuration-management)
5. [Dataset](#dataset)
6. [Model](#model)
7. [Quick Start](#quick-start)
8. [API Reference](#api-reference)
9. [MLflow Guide](#mlflow-guide)
10. [Docker Deployment](#docker-deployment)
11. [CI/CD Guide](#cicd-guide)
12. [Monitoring Guide](#monitoring-guide)
13. [Project Structure](#project-structure)
14. [Extending the Project](#extending-the-project)

---

## What is MLOps?

**MLOps** (Machine Learning Operations) is a set of practices that combines
Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems
in production reliably and efficiently.

### The Problem MLOps Solves

Traditional ML development suffers from:

| Problem | Consequence |
|---------|-------------|
| Manual, ad-hoc training | Reproducibility failures |
| No experiment tracking | Lost experiments, duplicated work |
| Code/model coupling | Hard to update models independently |
| No testing | Silent regressions in production |
| No monitoring | Model decay goes undetected |
| Manual deployment | Slow, error-prone releases |

MLOps solves these by treating ML systems like software systems: automated,
versioned, tested, monitored, and continuously improved.

### The Three Pillars of MLOps

```
┌──────────────────────────────────────────────────────────┐
│                      MLOps Pillars                        │
│                                                          │
│  1. People         2. Processes        3. Technology     │
│  ─────────────     ────────────────    ──────────────    │
│  Data Scientists   Agile ML workflow   MLflow            │
│  ML Engineers      Experiment cycles   CI/CD (GitHub)    │
│  DevOps/Platform   Review gates        Docker            │
│  Product Owners    Model governance    FastAPI           │
│                    Incident response   Monitoring tools  │
└──────────────────────────────────────────────────────────┘
```

---

## MLOps Maturity Levels

Google and Microsoft both define MLOps maturity models.  Here is a combined view:

### Level 0 — Manual Process (No MLOps)
- Data scientists train models in notebooks
- Manual deployment (copy files to server)
- No versioning, no monitoring
- **Risk**: High — "works on my machine" syndrome

### Level 1 — ML Pipeline Automation
- Automated training pipeline
- Experiment tracking (MLflow)
- Basic model versioning
- **This project reaches Level 1**

### Level 2 — CI/CD Pipeline Automation
- Automated testing of ML code
- Automated model validation (quality gates)
- Automated deployment on passing tests
- Model registry with stage management
- **This project demonstrates Level 2**

### Level 3 — Full MLOps (Continuous Training)
- Automated retraining triggered by data drift
- A/B testing / shadow deployments
- Feature stores
- Online learning capabilities

```
Level 0    Level 1         Level 2            Level 3
  │           │               │                  │
Manual    Automated       CI/CD + Registry   Continuous
Training  Pipelines       + Testing          Training
  │           │               │                  │
  └───────────┴───────────────┴──────────────────┘
                    Increasing Automation
```

---

## Project Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    MLOps Architecture                               │
│                                                                    │
│  SOURCE DATA           TRAINING PIPELINE          SERVING          │
│  ──────────────        ─────────────────────      ───────────────  │
│  sklearn               data_ingestion.py           FastAPI         │
│  Breast Cancer    ──►  data_preprocessing.py  ──►  /predict        │
│  Dataset               model_training.py           /predict/batch  │
│                        model_evaluation.py         /health         │
│                             │                      /metrics        │
│                             ▼                           │           │
│                    ┌─────────────────┐                 │           │
│                    │  MLflow Server  │◄────────────────┘           │
│                    │  - Experiments  │                              │
│                    │  - Runs         │                              │
│                    │  - Models       │                              │
│                    │  - Artefacts    │                              │
│                    └────────┬────────┘                             │
│                             │                                      │
│  MODEL REGISTRY        MONITORING              CI/CD               │
│  ─────────────         ──────────────          ──────────────────  │
│  None                  drift_detector.py       GitHub Actions      │
│    │                   - PSI                   lint → test         │
│    ▼                   - KS-test               → train             │
│  Staging               - Evidently             → quality gate      │
│    │                   HTML report             → docker push       │
│    ▼                                                               │
│  Production                                                        │
│    │                                                               │
│    ▼                                                               │
│  Archived                                                          │
└────────────────────────────────────────────────────────────────────┘
```

---

## MLOps Concepts Implemented

### 1. Data Management

**What it is:** Treating data as a first-class citizen — tracked, versioned,
and reproducible.

**Implementation:**
- Raw data saved to `data/raw/features.csv` and `data/raw/target.csv`
- Processed splits saved to `data/processed/` (X_train, X_val, X_test, y_*)
- Reference snapshot (`data/processed/reference.csv`) stored for drift monitoring
- Scaler fitted only on training data, persisted to `models/scaler.joblib`

**Why this matters:** Rebuilding the exact same dataset is essential for
diagnosing production issues and fair model comparison.

```python
# data_ingestion.py
X, y = ingest_data(config)     # → data/raw/

# data_preprocessing.py
data = preprocess(config)      # → data/processed/ + models/scaler.joblib
```

**Key principle — Train/Val/Test leakage prevention:**
```
StandardScaler.fit()     ← only on X_train
StandardScaler.transform() ← applied to X_val and X_test separately
```

---

### 2. Experiment Tracking (MLflow)

**What it is:** Systematic recording of every training run — parameters,
metrics, code version, and artefacts — so experiments are reproducible and
comparable.

**Why MLflow:**
- Open source, language-agnostic
- Local or remote tracking server
- Built-in Model Registry
- Clean Python API

**What is logged per run:**

| Category | Examples |
|----------|---------|
| Parameters | C=1.0, max_iter=1000, solver=lbfgs |
| Metrics | val_accuracy, val_f1, val_roc_auc |
| Artefacts | model .joblib, scaler .joblib, config.yaml |
| Tags | dataset name, split sizes |

**Viewing experiments:**
```bash
make mlflow-ui     # opens http://localhost:5000
```

**Code pattern:**
```python
with mlflow.start_run() as run:
    mlflow.log_params(model_params)
    model.fit(X_train, y_train)
    mlflow.log_metrics({"val_accuracy": acc, "val_f1": f1})
    mlflow.sklearn.log_model(model, artifact_path="logistic_regression",
                             registered_model_name="breast-cancer-classifier")
```

**Run naming convention:**
- Experiment: `logistic-regression-experiment`
- Each run gets a UUID (`run_id`) used to cross-reference evaluation metrics

---

### 3. Model Registry

**What it is:** A centralised catalogue of model versions with lifecycle stages,
enabling controlled promotion from development to production.

**Stage lifecycle:**
```
Training run completes
       │
       ▼
   ┌───────┐
   │ None  │  ← newly registered, untested
   └───┬───┘
       │  make promote-staging
       ▼
  ┌─────────┐
  │ Staging │  ← passed quality gate, under review
  └────┬────┘
       │  make promote-production
       ▼
 ┌────────────┐
 │ Production │  ← live, serving real traffic
 └─────┬──────┘
       │  (on next promotion)
       ▼
  ┌──────────┐
  │ Archived │  ← retired, kept for audit
  └──────────┘
```

**Governance rules:**
- Only models that pass all quality gate thresholds get promoted to Staging
- A human review step (or automated A/B test result) gates promotion to Production
- Archived models are never deleted — audit trail is preserved

**Loading by stage:**
```python
model_uri = "models:/breast-cancer-classifier/Production"
model = mlflow.sklearn.load_model(model_uri)
```

---

### 4. CI/CD Pipeline

**What it is:** Automated sequence of steps that runs on every code push,
ensuring code quality, model quality, and a deployable Docker image.

**Pipeline stages:**

```
┌────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌────────┐
│  Lint  │ ──►│  Tests   │ ──►│ Training │ ──►│ Quality Gate  │ ──►│ Docker │
│ ruff   │    │ pytest   │    │ pipeline │    │ acc ≥ 0.92    │    │ push   │
│ black  │    │ --cov    │    │ MLflow   │    │ f1  ≥ 0.90    │    │        │
└────────┘    └──────────┘    └──────────┘    │ auc ≥ 0.95    │    └────────┘
                                               └───────────────┘
```

**Quality gate thresholds** (`.github/workflows/ci_cd.yml`):

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| test_accuracy | ≥ 0.92 | Minimum acceptable overall correctness |
| test_f1 | ≥ 0.90 | Harmonic mean penalises imbalance |
| test_roc_auc | ≥ 0.95 | Calibration quality across all thresholds |

**Branch strategy:**
- `develop` → lint + tests only
- `main` → full pipeline + Docker push

**Setting up:**
1. Fork the repo
2. Add GitHub secrets: `DOCKER_USERNAME`, `DOCKER_TOKEN`
3. Push to `main` to trigger the full pipeline

---

### 5. Model Serving

**What it is:** Exposing the trained model as an HTTP API so downstream
applications can request predictions without knowing ML internals.

**FastAPI chosen because:**
- Auto-generates OpenAPI docs at `/docs`
- Pydantic validation catches malformed inputs before they reach the model
- Async support for high-throughput serving
- Lifespan context manager loads model once (not per request)

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check — returns `{"status": "ok"}` |
| `/metrics` | GET | Latest evaluation metrics from `reports/metrics.json` |
| `/predict` | POST | Single sample prediction |
| `/predict/batch` | POST | Batch predictions (list of samples) |

**Latency tracking:** Every `/predict` response includes `latency_ms` —
the wall-clock time from request receipt to response serialisation.

**Request example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 14.0,
    "mean_texture": 19.0,
    ...
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "benign",
  "probabilities": {
    "malignant": 0.1234,
    "benign": 0.8766
  },
  "latency_ms": 2.41
}
```

---

### 6. Monitoring & Drift Detection

**What it is:** Continuous statistical comparison of production data against
training data to detect when the model's input distribution has changed
(data drift) or when its accuracy has degraded (concept drift).

#### Why Models Degrade

```
Time 0 (training)           Time T (production)
──────────────────          ───────────────────
Data distribution D₀   →   Data distribution Dₜ ≠ D₀
Model trained on D₀         Model still predicting for D₀
                            Performance silently degrades
```

Common causes:
- Seasonal patterns
- Instrument / sensor calibration drift
- Change in patient population (for medical data)
- Data pipeline changes

#### Population Stability Index (PSI)

PSI measures how much a variable's distribution has shifted.

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | Stable | No action |
| 0.1 – 0.2 | Minor shift | Monitor closely |
| > 0.2 | Major shift | Investigate / retrain |

#### KS Test

The Kolmogorov-Smirnov test checks if two samples come from the same
distribution.  A p-value < 0.10 means the distributions are significantly
different.

#### Evidently Integration

```bash
pip install evidently
make drift    # generates reports/evidently_drift_report.html
```

The HTML report includes interactive charts for each feature, showing
reference vs. current distribution side-by-side.

#### Retraining Trigger

When drift is detected:
1. `drift_detected: true` in `reports/drift_report.json`
2. Alert logged (can be wired to email/Slack)
3. Trigger `make pipeline` to retrain on new data
4. Compare new model metrics vs. current production model in MLflow
5. If improvement confirmed → promote via Model Registry

---

### 7. Containerisation

**What it is:** Packaging the application and all its dependencies into a
portable Docker image that runs identically on any machine.

**Multi-service setup (docker-compose.yml):**

```
┌─────────────────────────────────────────────┐
│            Docker Compose                    │
│                                             │
│  ┌─────────────┐    ┌──────────────────┐   │
│  │ mlflow      │    │ api              │   │
│  │ :5000       │◄───│ :8000            │   │
│  │             │    │ (depends on      │   │
│  │ artefact    │    │  mlflow healthy) │   │
│  │ store       │    │                  │   │
│  └─────────────┘    └──────────────────┘   │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │ training (profile: train)            │  │
│  │ runs pipeline, then exits            │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘

Shared volumes: ./mlruns, ./models, ./data, ./reports
```

**Commands:**
```bash
# Start MLflow + API
make docker-up

# Run training inside Docker
make docker-train

# Stop everything
make docker-down
```

**Dockerfile design:**
- `python:3.11-slim` — minimal base image
- Dependencies installed before code (layer caching)
- `HEALTHCHECK` ensures orchestrators can verify readiness
- `PYTHONUNBUFFERED=1` — logs appear in real time

---

### 8. Testing Strategy

**Three test layers:**

```
┌──────────────────────────────────────────────────────┐
│                Testing Pyramid                        │
│                                                      │
│  /tests/test_api.py      Integration Tests           │
│  ─────────────────────   FastAPI endpoints           │
│         ▲                (mocked model)              │
│         │                                            │
│  /tests/test_model.py    Unit Tests                  │
│  ─────────────────────   sklearn model properties    │
│         ▲                                            │
│         │                                            │
│  /tests/test_data.py     Unit Tests                  │
│  ─────────────────────   ingestion + preprocessing   │
└──────────────────────────────────────────────────────┘
```

**What is tested:**

| File | Tests |
|------|-------|
| `test_data.py` | Feature count (30), binary target, no nulls, correct split sizes, zero-mean after scaling |
| `test_model.py` | Trains without error, accuracy > 90%, binary predictions, proba sums to 1, coef shape |
| `test_api.py` | /health 200, /predict returns required fields, valid label, 422 on missing field, proba sum |

**Running tests:**
```bash
make test           # all tests + coverage report
make test-fast      # data + model only (no FastAPI dependency)
```

**Coverage target:** > 80% on `src/` and `pipelines/`

---

### 9. Configuration Management

**Principle:** All tuneable values in one file, never hardcoded.

**`configs/config.yaml`** is the single source of truth for:
- Data paths, split ratios, random seeds
- Model hyperparameters
- MLflow tracking URI and experiment names
- API host/port
- Monitoring thresholds

**Pattern used across all modules:**
```python
def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
```

**Benefits:**
- Change `C: 1.0` to `C: 0.1` in one place → affects training, evaluation, docs
- Different configs for dev/staging/prod environments
- Config is logged as an MLflow artefact → every run is reproducible

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numeric (mean, error, worst of 10 cell nucleus measurements)
- **Target:** Binary — 0 = malignant, 1 = benign
- **Class distribution:** ~37% malignant, ~63% benign

**Feature groups:**

| Group | Features (×10) |
|-------|---------------|
| Mean | radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension |
| SE (error) | Same 10 features, standard error |
| Worst | Same 10 features, worst (largest) value |

---

## Model

**Logistic Regression (`sklearn.linear_model.LogisticRegression`)**

Logistic Regression is a linear classifier that models the probability of a
binary outcome using the sigmoid function:

```
P(y=1 | X) = σ(Xw + b) = 1 / (1 + e^(-(Xw + b)))
```

**Why Logistic Regression for MLOps demonstrations:**
- Deterministic and highly reproducible
- Fast training (< 1 second on this dataset)
- Interpretable coefficients → easy to audit
- Strong baseline: typically achieves > 95% accuracy on this dataset
- No random element in prediction (unlike tree ensembles)

**Hyperparameters (`configs/config.yaml`):**

| Param | Value | Meaning |
|-------|-------|---------|
| `C` | 1.0 | Inverse regularisation strength |
| `max_iter` | 1000 | Maximum solver iterations |
| `solver` | lbfgs | L-BFGS — good for small-medium datasets |
| `class_weight` | balanced | Auto-adjusts for class imbalance |
| `random_state` | 42 | Reproducibility |

**Expected performance (test set):**

| Metric | Expected |
|--------|----------|
| Accuracy | ~96% |
| F1 Score | ~95% |
| ROC-AUC | ~99% |

---

## Quick Start

```bash
# Clone / enter project
cd MLOps-LogisticRegression

# Install dependencies
pip install -r requirements.txt

# Run full ML pipeline
make pipeline

# Outputs:
#   data/raw/          - raw CSVs
#   data/processed/    - scaled splits
#   models/            - logistic_regression.joblib + scaler.joblib
#   reports/           - metrics.json, confusion_matrix.png, roc_curve.png
#   mlruns/            - MLflow experiment data

# View MLflow dashboard
make mlflow-ui
# → http://localhost:5000

# Start API
make serve
# → http://localhost:8000/docs

# Run tests
make test
```

---

## API Reference

### GET /health
```json
{"status": "ok", "model_loaded": true}
```

### GET /metrics
Returns `reports/metrics.json`:
```json
{
  "test_accuracy": 0.9646,
  "test_precision": 0.9583,
  "test_recall": 0.9583,
  "test_f1": 0.9583,
  "test_roc_auc": 0.9912
}
```

### POST /predict
**Request body** — all 30 features required (see `/docs` for full schema).

**Response:**
```json
{
  "prediction": 1,
  "label": "benign",
  "probabilities": {"malignant": 0.0412, "benign": 0.9588},
  "latency_ms": 1.87
}
```

### POST /predict/batch
**Request body:** Array of feature objects.
**Response:** Array of prediction objects.

---

## MLflow Guide

### Starting the tracking server
```bash
make mlflow-ui
# or
mlflow ui --backend-store-uri mlruns --port 5000
```

### Comparing runs
1. Open http://localhost:5000
2. Select the `logistic-regression-experiment`
3. Check multiple runs → click "Compare"
4. View parallel coordinates plot for hyperparameter search

### Promoting a model
```bash
# After verifying metrics in the UI:
make promote-staging      # None → Staging
make promote-production   # Staging → Production
```

### Loading from registry in code
```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/breast-cancer-classifier/Production")
```

---

## Docker Deployment

```bash
# Start all services
make docker-up

# Access:
# API:    http://localhost:8000
# MLflow: http://localhost:5000

# Run training inside Docker
make docker-train

# View logs
docker logs breast-cancer-api -f
docker logs mlflow-server -f

# Stop
make docker-down
```

---

## CI/CD Guide

### GitHub Actions workflow: `.github/workflows/ci_cd.yml`

**Jobs:**

| Job | Trigger | Outcome |
|-----|---------|---------|
| lint | every push | Fail on style errors |
| test | every push (after lint) | Fail on test failures |
| train | push to main | Run pipeline, upload artefacts |
| quality-gate | push to main | Fail if metrics below threshold |
| docker | push to main | Push image to Docker Hub |

**Required secrets:**
```
Settings → Secrets → Actions:
  DOCKER_USERNAME  ← your Docker Hub username
  DOCKER_TOKEN     ← Docker Hub access token
```

---

## Monitoring Guide

### Running drift detection
```bash
make drift
```

### Understanding the output
```json
{
  "dataset_psi": 0.043,
  "psi_label": "stable",
  "drift_detected": false,
  "drifted_features": [],
  "n_drifted": 0,
  "total_features": 30,
  "feature_psi": {"mean radius": 0.021, ...},
  "feature_ks_stat": {"mean radius": 0.085, ...}
}
```

### Simulating drift (for testing)
```python
import numpy as np, pandas as pd
ref = pd.read_csv("data/processed/reference.csv").drop(columns=["target"])
current = ref + np.random.normal(0, 2.0, ref.shape)  # large perturbation

from monitoring.drift_detector import DriftDetector
from src.data_ingestion import load_config
detector = DriftDetector(load_config())
report = detector.detect(current)
```

### Retraining workflow
1. Drift detected → `drift_report.json` shows `drift_detected: true`
2. Collect new labelled data
3. Update `data/raw/` with new samples
4. Run `make pipeline` to retrain
5. Compare new vs. old model in MLflow UI
6. If improved: `make promote-production`

---

## Project Structure

```
MLOps-LogisticRegression/
│
├── CLAUDE.md                   # AI assistant instructions
├── README.md                   # This file
├── Makefile                    # Common commands
├── requirements.txt            # Python dependencies
├── setup.py                    # Package metadata
│
├── configs/
│   └── config.yaml             # All configuration
│
├── data/                       # Git-ignored in production
│   ├── raw/                    # Original dataset CSVs
│   └── processed/              # Scaled splits + reference snapshot
│
├── src/                        # Core ML modules
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_registry.py
│
├── pipelines/                  # Orchestration
│   ├── training_pipeline.py    # Runs steps 1-5 end-to-end
│   └── inference_pipeline.py   # Loads model + scaler for prediction
│
├── api/
│   └── main.py                 # FastAPI application
│
├── monitoring/
│   └── drift_detector.py       # PSI + KS-test drift detection
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/                  # EDA / scratch work
│
├── models/                     # Git-ignored
│   ├── logistic_regression.joblib
│   └── scaler.joblib
│
├── reports/                    # Git-ignored
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.csv
│   └── drift_report.json
│
└── mlruns/                     # MLflow store (git-ignored)
```

---

## Extending the Project

### Add DVC for data versioning
```bash
pip install dvc
dvc init
dvc add data/raw/
git add data/raw/.gitignore data/raw.dvc
dvc remote add -d myremote s3://my-bucket/dvc
dvc push
```

### Add a Feature Store (Feast)
```python
# feature_store/features.py
from feast import Entity, FeatureView, Field
# Define features once, reuse across training and serving
```

### Add online serving with model caching
```python
# api/cache.py — Redis-backed prediction cache
import redis
cache = redis.Redis()
# Cache predictions for identical feature vectors
```

### Add A/B testing
```python
# api/main.py — shadow model
@app.post("/predict")
def predict(payload: FeatureInput):
    result_a = model_a.predict(features)  # Production
    result_b = model_b.predict(features)  # Shadow
    log_comparison(result_a, result_b)    # To MLflow
    return result_a                        # Always serve A
```

### Add continuous retraining
```yaml
# .github/workflows/retrain.yml
on:
  schedule:
    - cron: "0 2 * * 1"   # Every Monday at 2 AM
```

---

*Built with scikit-learn, MLflow, FastAPI, Docker, and GitHub Actions.*
