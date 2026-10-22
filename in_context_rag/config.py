"""Configuration and local data-path defaults for the in-context pipeline."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "private"
TRAIN_FILENAME = "train.csv"
CLINICAL_DATA_ROOT_ENV = "CLINICAL_DATA_ROOT"


def clinical_data_root() -> Path:
    """Return the configured local clinical-data directory.

    ``CLINICAL_DATA_ROOT`` takes precedence; otherwise use the repository's
    ignored ``data/private/`` directory. The directory is not created here.
    """
    configured = os.environ.get(CLINICAL_DATA_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT


def training_data_path() -> Path:
    """Return the default path to the local training CSV."""
    return clinical_data_root() / TRAIN_FILENAME
