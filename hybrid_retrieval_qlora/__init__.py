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
from .preprocess.chat_formatting import (
    build_chat_messages,
    extract_template_sections,
    format_case_input_payload,
    format_sft_example,
)
from .preprocess.completion_mask import (
    MockQwenTokenizer,
    get_qwen_tokenizer,
    tokenize_sft_dataset,
    tokenize_sft_example,
)
from .preprocess.load_data import create_deterministic_splits, filter_training_pool, load_cases
from .preprocess.validation import validate_tokenized_sft_record
from .retrieval.cache_manifest import EmbeddingCacheManager, EmbeddingSpec
from .retrieval.chunk_pooling import (
    aggregate_chunk_vectors,
    compute_chunk_weights,
    validate_embedding_matrix,
)
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.pubmedbert_embeddings import PubMedBERTEmbedder
from .schemas import (
    ChatMessage,
    ClinicalCase,
    FormattedSFTExample,
    RetrievalCandidate,
    RetrievalResult,
    SplitManifest,
    TokenizedSFTRecord,
)

__all__ = [
    "ChatMessage",
    "ClinicalCase",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_REVISION",
    "DEFAULT_TOP_K",
    "DEFAULT_WEIGHTS",
    "EXPECTED_EMBEDDING_DIM",
    "EmbeddingCacheManager",
    "EmbeddingSpec",
    "FormattedSFTExample",
    "HybridRetriever",
    "MockQwenTokenizer",
    "PubMedBERTEmbedder",
    "RetrievalCandidate",
    "RetrievalResult",
    "SplitManifest",
    "TokenizedSFTRecord",
    "aggregate_chunk_vectors",
    "build_chat_messages",
    "compute_chunk_weights",
    "create_deterministic_splits",
    "extract_template_sections",
    "filter_training_pool",
    "format_case_input_payload",
    "format_sft_example",
    "get_qwen_tokenizer",
    "load_cases",
    "tokenize_sft_dataset",
    "tokenize_sft_example",
    "validate_embedding_matrix",
    "validate_tokenized_sft_record",
]
