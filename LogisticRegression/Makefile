.PHONY: install train evaluate serve test lint clean docker-up docker-down drift

PYTHON := python3
CONFIG := configs/config.yaml

install:
	pip install -r requirements.txt

# ── ML Pipeline ──────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) src/data_ingestion.py

preprocess:
	$(PYTHON) src/data_preprocessing.py

train:
	$(PYTHON) pipelines/training_pipeline.py

evaluate:
	$(PYTHON) src/model_evaluation.py

pipeline: ingest preprocess train evaluate
	@echo "Full training pipeline complete."

# ── Drift Monitoring ─────────────────────────────────────────────────────────
drift:
	$(PYTHON) monitoring/drift_detector.py

# ── API ───────────────────────────────────────────────────────────────────────
serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --backend-store-uri mlruns --port 5000

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov=pipelines --cov-report=term-missing

test-fast:
	pytest tests/test_model.py tests/test_data.py -v

# ── Lint ──────────────────────────────────────────────────────────────────────
lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t breast-cancer-classifier -f docker/Dockerfile .

docker-up:
	docker compose -f docker/docker-compose.yml up -d mlflow api

docker-train:
	docker compose -f docker/docker-compose.yml --profile train up training

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Model Registry ────────────────────────────────────────────────────────────
promote-staging:
	$(PYTHON) -c "from src.model_registry import promote_to_staging, load_config; promote_to_staging(load_config())"

promote-production:
	$(PYTHON) -c "from src.model_registry import promote_to_production, load_config; promote_to_production(load_config())"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw/* data/processed/* models/* reports/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-all: clean
	rm -rf mlruns/
