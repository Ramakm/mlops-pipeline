import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()

class DataIngestion:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None

    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generates synthetic energy consumption data."""
        logger.info("Generating synthetic data", n_samples=n_samples)
        
        date_rng = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
        
        # Trend
        trend = np.linspace(0, 10, n_samples)
        
        # Seasonality (Daily + Weekly)
        daily_seasonality = 10 * np.sin(2 * np.pi * date_rng.hour / 24)
        weekly_seasonality = 5 * np.sin(2 * np.pi * date_rng.dayofweek / 7)
        
        # Noise
        noise = np.random.normal(0, 2, n_samples)
        
        # Target variable (Energy Consumption)
        consumption = 50 + trend + daily_seasonality + weekly_seasonality + noise
        
        df = pd.DataFrame(date_rng, columns=['timestamp'])
        df['consumption'] = consumption
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df

    def load_data(self) -> pd.DataFrame:
        """Loads data from CSV or generates synthetic data."""
        if self.data_path and self.data_path.exists():
            logger.info("Loading data from file", path=str(self.data_path))
            return pd.read_csv(self.data_path, parse_dates=['timestamp'])
        else:
            logger.warning("Data file not found or not provided, using synthetic data")
            return self.generate_synthetic_data()
