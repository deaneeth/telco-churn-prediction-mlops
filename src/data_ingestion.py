from typing import Tuple
import pandas as pd
from .utils import get_logger
logger = get_logger("data_ingestion")

class DataIngestion:
    def __init__(self, path: str):
        self.path = path

    def load_csv(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.path}")
        df = pd.read_csv(self.path)
        logger.info(f"Loaded dataframe shape: {df.shape}")
        return df

    def validate_columns(self, df: pd.DataFrame, required_cols: list) -> bool:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        return True
