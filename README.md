# MLOps Pipeline Project
<img width="2732" height="2048" alt="image" src="https://github.com/user-attachments/assets/f933e35a-b6d1-4c48-9291-bac9a5e0de52" />

![Stars](https://img.shields.io/github/stars/Ramakm/mlops-pipeline?style=flat-square)
![Forks](https://img.shields.io/github/forks/Ramakm/mlops-pipeline?style=flat-square)
![PRs](https://img.shields.io/github/issues-pr/Ramakm/mlops-pipeline?style=flat-square)
![Issues](https://img.shields.io/github/issues/Ramakm/mlops-pipeline?style=flat-square)
![Contributors](https://img.shields.io/github/contributors/Ramakm/mlops-pipeline?style=flat-square)
![License](https://img.shields.io/github/license/Ramakm/mlops-pipeline?style=flat-square)



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

- This project is licensed under the MIT License. See `LICENSE` for details.

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ramakrushnamohapatra/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)

