# MLOps Pipeline Project

## Overview
This repository implements a production‑grade end‑to‑end machine‑learning pipeline for time‑series forecasting using a custom dataset. It covers the full MLOps lifecycle:
- Data ingestion & feature engineering
- LSTM model training with transfer learning
- Experiment tracking with **MLflow**
- Asynchronous training jobs
- Real‑time inference API built with **FastAPI** (JWT auth, Redis caching, rate limiting)
- Monitoring via **Prometheus** and **Grafana**
- Docker Compose for reproducible environments

## Quick Start
```bash
# Install dependencies with uv (recommended)
uv sync

# Copy example environment variables
cp .env

# Build and start services
docker compose up -d

# Run the API
uv run python -m src.api.main
```

## Environment Variables (`.env`)
The application reads configuration from a `.env` file (see `.env.example` for a template). Required variables:
```
PROJECT_NAME=MLOps Pipeline
API_V1_STR=/api/v1
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30
REDIS_URL=redis://localhost:6379/0
MLFLOW_TRACKING_URI=http://localhost:5000
```
Adjust values as needed for your deployment.

## Documentation
- API docs: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`
- Grafana: `http://localhost:3000` (default credentials admin/admin)

## License
MIT