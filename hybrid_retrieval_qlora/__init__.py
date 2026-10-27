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
from .inference.generate_reports import generate_reports
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
    EvaluationScore,
    EvaluationSummary,
    FormattedSFTExample,
    GenerationResult,
    RetrievalCandidate,
    RetrievalResult,
    SplitManifest,
    TokenizedSFTRecord,
)
from .training.evaluate import evaluate_reports
from .training.qlora_config import QLoRAConfig
from .training.sft_dataset import DemonstrationUsage, prepare_sft_records, to_hf_dataset
from .training.train import train_qlora

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
    "DemonstrationUsage",
    "EXPECTED_EMBEDDING_DIM",
    "EmbeddingCacheManager",
    "EmbeddingSpec",
    "EvaluationScore",
    "EvaluationSummary",
    "FormattedSFTExample",
    "GenerationResult",
    "HybridRetriever",
    "MockQwenTokenizer",
    "PubMedBERTEmbedder",
    "QLoRAConfig",
    "RetrievalCandidate",
    "RetrievalResult",
    "SplitManifest",
    "TokenizedSFTRecord",
    "aggregate_chunk_vectors",
    "build_chat_messages",
    "compute_chunk_weights",
    "create_deterministic_splits",
    "evaluate_reports",
    "extract_template_sections",
    "filter_training_pool",
    "format_case_input_payload",
    "format_sft_example",
    "generate_reports",
    "get_qwen_tokenizer",
    "load_cases",
    "prepare_sft_records",
    "to_hf_dataset",
    "tokenize_sft_dataset",
    "tokenize_sft_example",
    "train_qlora",
    "validate_embedding_matrix",
    "validate_tokenized_sft_record",
]
