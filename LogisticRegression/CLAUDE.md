# CLAUDE.md — MLOps Logistic Regression Project

## Project Overview

End-to-end MLOps project using **Logistic Regression** to classify breast tumours
(malignant vs. benign).  The project demonstrates the full MLOps lifecycle:
data versioning, experiment tracking, model registry, CI/CD, containerisation,
and production monitoring.

---

## Directory Layout

```
MLOps-LogisticRegression/
├── configs/
│   └── config.yaml          # Single source of truth for all config
├── src/
│   ├── data_ingestion.py    # Load dataset → data/raw/
│   ├── data_preprocessing.py# Scale + split → data/processed/ + models/scaler.joblib
│   ├── model_training.py    # Fit LR, log to MLflow, register model
│   ├── model_evaluation.py  # Test metrics, confusion matrix, ROC curve
│   └── model_registry.py    # MLflow Model Registry helpers
├── pipelines/
│   ├── training_pipeline.py # Orchestrates steps 1-5 end-to-end
│   └── inference_pipeline.py# Loads model + scaler, returns predictions
├── api/
│   └── main.py              # FastAPI: /predict, /predict/batch, /health, /metrics
├── monitoring/
│   └── drift_detector.py    # PSI + KS-test, optional Evidently HTML report
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml   # mlflow + api + training services
├── .github/workflows/
│   └── ci_cd.yml            # lint → test → train → docker → quality gate
├── Makefile                 # All common commands
├── requirements.txt
└── setup.py
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run full training pipeline (ingest → preprocess → train → evaluate → registry)
make pipeline

# 3. Open MLflow UI
make mlflow-ui        # http://localhost:5000

# 4. Serve the API
make serve            # http://localhost:8000
# Docs at http://localhost:8000/docs

# 5. Run tests
make test

# 6. Check for data drift
make drift
```

---

## Key Design Decisions

### Configuration
All tuneable values live in `configs/config.yaml`.  Never hardcode paths, seeds,
or hyperparameters in module code.  Modules call `load_config()` which reads the
YAML; the pipeline passes the config dict downstream.

### MLflow Tracking
- `tracking_uri: mlruns` (local file store, change to a remote URI for teams)
- Every training run logs: params, val metrics, model artefact, scaler, config
- The model is registered as `breast-cancer-classifier` in the Model Registry
- Stages used: `None → Staging → Production → Archived`

### Data Splits
```
All data (569 samples)
  └── 80% train+val
        ├── ~89% train  (~405 samples)
        └── ~11% val    (~51 samples)
  └── 20% test          (~113 samples)
```
Splits are stratified on `target` to preserve class ratio.

### Scaler
`StandardScaler` is fit **only** on training data and applied to val/test.
The fitted scaler is persisted to `models/scaler.joblib` and must be shipped
alongside the model artefact.

### API
FastAPI with lifespan context manager loads model once at startup (not per request).
`/predict` accepts a single sample JSON; `/predict/batch` accepts a list.
All feature keys use underscores (Pydantic) and are remapped to sklearn feature names
(spaces) before inference.

### Monitoring
`DriftDetector.detect(current_df)` computes:
- **PSI** (Population Stability Index) per feature — thresholds: < 0.1 stable,
  0.1–0.2 minor shift, > 0.2 major shift
- **KS-test** p-value < 0.10 → feature flagged as drifted
- Optional Evidently HTML report saved to `reports/evidently_drift_report.html`

---

## MLOps Lifecycle Commands

| Command | Description |
|---------|-------------|
| `make ingest` | Load dataset to `data/raw/` |
| `make preprocess` | Scale + split to `data/processed/` |
| `make train` | Full training pipeline with MLflow |
| `make evaluate` | Evaluate on test set, save reports |
| `make pipeline` | All of the above in sequence |
| `make drift` | Run drift detection on reference data |
| `make serve` | Start FastAPI (dev, port 8000) |
| `make mlflow-ui` | Start MLflow UI (port 5000) |
| `make test` | Pytest with coverage |
| `make promote-staging` | Move latest model → Staging |
| `make promote-production` | Move Staging model → Production |
| `make docker-up` | Start MLflow + API containers |
| `make docker-train` | Run training inside Docker |
| `make clean` | Remove data, models, reports |

---

## Adding New Features

### New hyperparameter
1. Add to `configs/config.yaml` under `model.params`
2. `model_training.py` passes `config["model"]["params"]` directly to
   `LogisticRegression(**model_params)` — no code change needed

### New metric
1. Add to the `metrics` dict in `src/model_evaluation.py`
2. It is automatically logged to MLflow and written to `reports/metrics.json`
3. Add a threshold check to `.github/workflows/ci_cd.yml` quality-gate step

### New API endpoint
1. Add route in `api/main.py`
2. Add a corresponding test in `tests/test_api.py`

### Replace the model
Swap `LogisticRegression` in `src/model_training.py` for any sklearn-compatible
estimator.  Everything else (MLflow, API, monitoring) remains unchanged because
the pipeline is model-agnostic.

---

## CI/CD Flow

```
push to main
  │
  ├─ 1. Lint (ruff + black)
  ├─ 2. Unit + integration tests (pytest --cov)
  ├─ 3. Training pipeline (full run, artefacts uploaded)
  ├─ 4. Quality gate  (accuracy ≥ 0.92, F1 ≥ 0.90, ROC-AUC ≥ 0.95)
  └─ 5. Docker build + push to Docker Hub
```

Secrets required in GitHub:
- `DOCKER_USERNAME`
- `DOCKER_TOKEN`

---

## Environment Notes

- Python 3.11+
- MLflow local store: `mlruns/` (git-ignored in production)
- Model artefacts: `models/` (git-ignored)
- Reports: `reports/` (git-ignored)
- No database required for local development
