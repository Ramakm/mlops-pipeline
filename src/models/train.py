import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any
import mlflow
import mlflow.pytorch
from src.models.lstm import LSTMModel
from src.mlops.tracking import setup_mlflow_tracking
from src.core.config import settings
import structlog
import os

logger = structlog.get_logger()

class ModelTrainer:
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, learning_rate: float = 0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMModel(input_dim, hidden_dim, 1, num_layers).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate
        }
        setup_mlflow_tracking()

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 32, experiment_name: str = "Base_LSTM") -> Dict[str, Any]:
        logger.info("Starting training", epochs=epochs, device=str(self.device))
        
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            mlflow.log_params(self.params)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0
                for batch_X, batch_y in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(loader)
                logger.info(f"Epoch {epoch+1}/{epochs}", loss=avg_loss)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            return {"run_id": run.info.run_id, "final_loss": avg_loss}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy().squeeze()
            
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        return {"mse": mse, "mae": mae}
