import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import structlog
import joblib
from pathlib import Path

logger = structlog.get_logger()

class FeaturePipeline:
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.feature_columns = ['consumption', 'hour', 'day_of_week']

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the scaler and transforms the data."""
        logger.info("Fitting and transforming data")
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        return df_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms data using the fitted scaler."""
        df_scaled = df.copy()
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df_scaled

    def create_sequences(self, data: pd.DataFrame, target_col: str = 'consumption') -> Tuple[np.ndarray, np.ndarray]:
        """Creates sequences for LSTM training."""
        logger.info("Creating sequences", window_size=self.window_size)
        
        sequences = []
        targets = []
        
        data_values = data[self.feature_columns].values
        target_values = data[target_col].values
        
        for i in range(len(data) - self.window_size):
            seq = data_values[i : i + self.window_size]
            label = target_values[i + self.window_size]
            sequences.append(seq)
            targets.append(label)
            
        return np.array(sequences), np.array(targets)

    def save_scaler(self, path: str):
        """Saves the fitted scaler."""
        logger.info("Saving scaler", path=path)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Loads a fitted scaler."""
        logger.info("Loading scaler", path=path)
        self.scaler = joblib.load(path)
