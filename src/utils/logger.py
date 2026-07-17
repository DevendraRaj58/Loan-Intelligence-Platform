"""
Logging utility for the Loan Intelligence Platform.
"""

import logging
from pathlib import Path

from src.utils.config import config


LOG_DIR = Path(config["logging"]["log_dir"])
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / config["logging"]["log_file"]


logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger("LoanIntelligencePlatform")