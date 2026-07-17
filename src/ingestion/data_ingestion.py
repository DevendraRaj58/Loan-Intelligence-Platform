"""
Data Ingestion Module

Responsible for loading raw datasets into the project.
"""

from pathlib import Path
import pandas as pd

from src.utils.config import config
from src.utils.logger import logger


class DataIngestion:
    """
    Handles loading datasets from the raw data directory.
    """

    def __init__(self):

        self.raw_data_path = config["data"]["raw_path"]

    def load_csv(self, filename: str):

        file_path = self.raw_data_path / filename

        if not file_path.exists():

            logger.error(f"{filename} not found.")

            raise FileNotFoundError(file_path)

        logger.info(f"Loading {filename}")

        df = pd.read_csv(file_path)

        logger.info(f"Dataset loaded successfully.")

        logger.info(f"Shape : {df.shape}")

        return df