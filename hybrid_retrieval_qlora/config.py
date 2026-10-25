"""Configuration and path resolution for Approach 2 retrieval and embedding cache."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "private"
DEFAULT_CACHE_DIR = REPOSITORY_ROOT / "artifacts" / "pubmedbert_cache"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
CLINICAL_DATA_ROOT_ENV = "CLINICAL_DATA_ROOT"

# PubMedBERT Embedding Specifications
DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_EMBEDDING_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
EXPECTED_EMBEDDING_DIM = 768
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_POOLING_METHOD = "normalized-chunks-new-token-weighted-v1"

# Retrieval Weights and Candidate Limits
DEFAULT_TOP_K = 5
DEFAULT_WEIGHTS: dict[str, float] = {
    "embedding_similarity": 0.60,
    "keyword_similarity": 0.20,
    "modality_match": 0.10,
    "body_part_match": 0.10,
}


def load_project_environment() -> None:
    """Load local repository settings from .env without overriding shell variables."""
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)


def clinical_data_root() -> Path:
    """Return the configured clinical-data root path.

    Resolves ``CLINICAL_DATA_ROOT`` environment variable if present;
    otherwise falls back to the ignored repository ``data/private/`` directory.
    """
    configured = os.environ.get(CLINICAL_DATA_ROOT_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_DATA_ROOT


def training_data_path() -> Path:
    """Return the default local path to the training CSV."""
    return clinical_data_root() / TRAIN_FILENAME


def test_data_path() -> Path:
    """Return the default local path to the evaluation/test CSV."""
    return clinical_data_root() / TEST_FILENAME
