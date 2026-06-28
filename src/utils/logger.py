from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------------
# Project Root
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

formatter = logging.Formatter(LOG_FORMAT)

logger = logging.getLogger("AgenticAI")

logger.setLevel(logging.INFO)

if not logger.handlers:

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Application Log
    app_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    logger.addHandler(app_handler)

    # Error Log
    error_handler = RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)