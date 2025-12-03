from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from src.core.config import settings
from src.core.logging import setup_logging, logger
from src.api.auth import create_access_token, verify_password, get_password_hash
from src.api.dependencies import get_current_user
from src.data.ingestion import DataIngestion
from src.data.processing import FeaturePipeline
from src.models.train import ModelTrainer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import structlog
import redis
import json
import asyncio
import numpy as np

setup_logging()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Redis connection
redis_client = redis.from_url(settings.REDIS_URL)

# Mock user database
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),
    }
}

@app.post(f"{settings.API_V1_STR}/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post(f"{settings.API_V1_STR}/train")
async def train_model(
    background_tasks: BackgroundTasks,
    epochs: int = 10,
    current_user: str = Depends(get_current_user)
):
    job_id = f"job_{int(asyncio.get_event_loop().time())}"
    
    def training_task(job_id):
        logger.info("Starting background training task", job_id=job_id)
        try:
            # 1. Ingest Data
            ingestion = DataIngestion()
            df = ingestion.load_data()
            
            # 2. Feature Engineering
            pipeline = FeaturePipeline()
            df_processed = pipeline.fit_transform(df)
            X, y = pipeline.create_sequences(df_processed)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 3. Train Model
            trainer = ModelTrainer(input_dim=X.shape[2])
            result = trainer.train(X_train, y_train, epochs=epochs)
            
            # 4. Evaluate
            metrics = trainer.evaluate(X_test, y_test)
            
            logger.info("Training completed", job_id=job_id, metrics=metrics)
            redis_client.set(job_id, json.dumps({"status": "completed", "metrics": metrics, "run_id": result["run_id"]}))
            
        except Exception as e:
            logger.error("Training failed", job_id=job_id, error=str(e))
            redis_client.set(job_id, json.dumps({"status": "failed", "error": str(e)}))

    redis_client.set(job_id, json.dumps({"status": "running"}))
    background_tasks.add_task(training_task, job_id)
    
    return {"job_id": job_id, "status": "submitted"}

@app.post(f"{settings.API_V1_STR}/predict")
@limiter.limit("5/minute")
async def predict(
    data: dict,
    request: Request,
    current_user: str = Depends(get_current_user)
):
    """
    Real-time prediction endpoint with Redis caching.
    Expected input: {"features": [[...]]} (list of sequences)
    """
    try:
        features = data.get("features")
        if not features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        # Create a cache key based on input
        cache_key = f"pred_{hash(str(features))}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            logger.info("Cache hit", key=cache_key)
            return json.loads(cached_result)
        
        # Load latest model (mocking this part for now, ideally load from MLflow registry)
        # In a real scenario, we would load the model once at startup or have a model serving class
        # For this demo, we assume a model is available or we use a dummy prediction
        
        # Mock prediction logic for demo purposes if no model is loaded
        # In production: model = load_model_from_mlflow()
        # prediction = model.predict(np.array(features))
        
        # Simulating prediction
        prediction = [0.5 * sum(seq) / len(seq) for seq in features] 
        
        result = {"prediction": prediction}
        
        # Cache result for 1 hour
        redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return result
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/status/{{job_id}}")
async def get_status(job_id: str, current_user: str = Depends(get_current_user)):
    status = redis_client.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(status)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
