"""Biomedical retrieval and embedding cache for Approach 2."""

from __future__ import annotations

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    EXPECTED_EMBEDDING_DIM,
)
from .preprocess.load_data import create_deterministic_splits, filter_training_pool, load_cases
from .retrieval.cache_manifest import EmbeddingCacheManager, EmbeddingSpec
from .retrieval.chunk_pooling import (
    aggregate_chunk_vectors,
    compute_chunk_weights,
    validate_embedding_matrix,
)
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.pubmedbert_embeddings import PubMedBERTEmbedder
from .schemas import ClinicalCase, RetrievalCandidate, RetrievalResult, SplitManifest

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "DEFAULT_TOP_K",
    "DEFAULT_WEIGHTS",
    "EXPECTED_EMBEDDING_DIM",
    "ClinicalCase",
    "EmbeddingCacheManager",
    "EmbeddingSpec",
    "HybridRetriever",
    "PubMedBERTEmbedder",
    "RetrievalCandidate",
    "RetrievalResult",
    "SplitManifest",
    "aggregate_chunk_vectors",
    "compute_chunk_weights",
    "create_deterministic_splits",
    "filter_training_pool",
    "load_cases",
    "validate_embedding_matrix",
]
