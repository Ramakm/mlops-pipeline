"""
Training Pipeline — orchestrates the full ML lifecycle:
  1. Data Ingestion
  2. Data Preprocessing
  3. Model Training  (MLflow run started here)
  4. Model Evaluation
  5. Model Registry promotion (→ Staging)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
import yaml

from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess
from src.model_training import train
from src.model_evaluation import evaluate
from src.model_registry import promote_to_staging


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training_pipeline(config_path: str = "configs/config.yaml") -> dict:
    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("STEP 1/5  Data Ingestion")
    logger.info("=" * 60)
    ingest_data(config)

    logger.info("=" * 60)
    logger.info("STEP 2/5  Data Preprocessing")
    logger.info("=" * 60)
    data = preprocess(config)

    logger.info("=" * 60)
    logger.info("STEP 3/5  Model Training")
    logger.info("=" * 60)
    model, run_id = train(config, data)

    logger.info("=" * 60)
    logger.info("STEP 4/5  Model Evaluation")
    logger.info("=" * 60)
    metrics = evaluate(config, run_id=run_id)

    logger.info("=" * 60)
    logger.info("STEP 5/5  Promote to Staging")
    logger.info("=" * 60)
    try:
        promote_to_staging(config)
    except Exception as e:
        logger.warning(f"Registry promotion skipped: {e}")

    logger.info("Training pipeline complete.")
    logger.info(f"Final metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    run_training_pipeline()
