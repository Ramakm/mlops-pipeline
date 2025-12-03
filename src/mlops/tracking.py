import mlflow
from src.core.config import settings
import structlog

logger = structlog.get_logger()

def setup_mlflow_tracking():
    """Configures MLflow tracking URI."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking URI set", uri=settings.MLFLOW_TRACKING_URI)

def log_experiment(experiment_name: str, params: dict, metrics: dict, artifacts: dict = None):
    """Logs an experiment run to MLflow."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
