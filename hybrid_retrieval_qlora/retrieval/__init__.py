"""Biomedical text retrieval, chunk pooling, and embedding cache modules."""

from __future__ import annotations

from .cache_manifest import EmbeddingCacheManager, EmbeddingSpec
from .chunk_pooling import (
    aggregate_chunk_vectors,
    compute_chunk_weights,
    validate_embedding_matrix,
)
from .hybrid_retriever import HybridRetriever
from .pubmedbert_embeddings import PubMedBERTEmbedder

__all__ = [
    "EmbeddingCacheManager",
    "EmbeddingSpec",
    "HybridRetriever",
    "PubMedBERTEmbedder",
    "aggregate_chunk_vectors",
    "compute_chunk_weights",
    "validate_embedding_matrix",
]
