"""Canonical paths for the project. Single source of truth."""
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/Vol_fitter")
RAW_DIR = PROJECT_ROOT / "data_SPY"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
FITTED_DIR = PROJECT_ROOT / "data_fitted"
AUX_DIR = PROJECT_ROOT / "data_raw_aux"
LOCAL_RAW = Path("/content/data_SPY")  # session-scoped staging
