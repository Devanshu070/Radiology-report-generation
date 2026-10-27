"""SFT dataset preparation, demonstration retrieval, and tokenized dataset building."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Any, Literal

from ..config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_REVISION,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_SFT_SYSTEM_PROMPT,
    DEFAULT_TOP_K,
)
from ..preprocess.chat_formatting import format_sft_example
from ..preprocess.completion_mask import get_qwen_tokenizer, tokenize_sft_example
from ..retrieval.cache_manifest import EmbeddingCacheManager
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.pubmedbert_embeddings import PubMedBERTEmbedder
from ..schemas import ClinicalCase, FormattedSFTExample, TokenizedSFTRecord


@dataclass(frozen=True)
class DemonstrationUsage:
    """Tracking metadata for demonstration cases used in an SFT example."""

    case_id: str
    n_examples: int
    demonstration_ids: list[str]
    input_tokens: int
    full_tokens: int


def prepare_sft_records(
    cases: Sequence[ClinicalCase],
    tokenizer: Any,
    *,
    retriever: HybridRetriever | None = None,
    candidate_embeddings: Any | None = None,
    query_embeddings: Any | None = None,
    cache_manager: EmbeddingCacheManager | None = None,
    embedder: PubMedBERTEmbedder | None = None,
    supervised: bool = True,
    top_k: int = DEFAULT_TOP_K,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    system_prompt: str = DEFAULT_SFT_SYSTEM_PROMPT,
    padding: bool | Literal["max_length", "longest"] = False,
) -> tuple[list[TokenizedSFTRecord], list[DemonstrationUsage], list[FormattedSFTExample]]:
    """Format and tokenize clinical cases into aligned SFT training/evaluation records.

    For each case, retrieves top-k candidates from the training pool (if retriever provided),
    formats the ChatML conversation turns with whole-demonstration budget pruning,
    and applies target-only completion masking.

    Args:
        cases: Target clinical cases to process.
        tokenizer: Hugging Face or Mock tokenizer.
        retriever: Optional HybridRetriever fitted strictly on training pool.
        candidate_embeddings: Optional precomputed embeddings for the retriever candidate pool.
        query_embeddings: Optional precomputed embeddings for query cases.
        cache_manager: Optional cache manager for PubMedBERT embeddings.
        embedder: Optional PubMedBERT embedding model.
        supervised: If True, includes ground-truth report in completion and applies loss.
        top_k: Number of demonstration examples to retrieve per case.
        max_input_tokens: Maximum tokens permitted for prompt turns.
        max_seq_length: Maximum allowed total sequence length.
        system_prompt: System instructions for the chat template.
        padding: Padding strategy for tokenized records.

    Returns:
        Tuple of (tokenized_records, demonstration_usages, formatted_examples).
    """
    tokenized_records: list[TokenizedSFTRecord] = []
    usages: list[DemonstrationUsage] = []
    formatted_examples: list[FormattedSFTExample] = []

    # Map candidate cases by ID if retriever is available
    candidate_pool = getattr(retriever, "candidate_pool", []) if retriever else []
    training_cases_by_id = {c.case_id: c for c in candidate_pool}

    for idx, case in enumerate(cases):
        candidate_demos: list[ClinicalCase] = []
        if retriever is not None and top_k > 0:
            q_emb = None
            if query_embeddings is not None:
                q_emb = query_embeddings[idx : idx + 1]

            retrieval_results = retriever.retrieve(
                [case],
                candidate_embeddings=candidate_embeddings,
                query_embeddings=q_emb,
                cache_manager=cache_manager,
                embedder=embedder,
                top_k=top_k,
            )
            if retrieval_results and retrieval_results[0].candidates:
                for cand in retrieval_results[0].candidates:
                    # Guarantee candidate is from training pool
                    demo_case = training_cases_by_id.get(cand.candidate_case_id)
                    if demo_case and demo_case.case_id != case.case_id:
                        candidate_demos.append(demo_case)

        formatted = format_sft_example(
            case,
            candidate_demos,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
            supervised=supervised,
        )
        formatted_examples.append(formatted)

        record = tokenize_sft_example(
            formatted,
            tokenizer,
            max_length=max_seq_length,
            padding=padding,
        )

        if supervised:
            ones_count = record.completion_mask.count(1)
            zeros_count = record.completion_mask.count(0)
            if ones_count == 0 or zeros_count == 0:
                raise ValueError(
                    f"Case '{case.case_id}' produced invalid completion mask: "
                    f"supervised={ones_count}, masked={zeros_count}. "
                    "Expected both masked input tokens and supervised completion tokens."
                )

        tokenized_records.append(record)
        usages.append(
            DemonstrationUsage(
                case_id=case.case_id,
                n_examples=len(formatted.demonstration_ids),
                demonstration_ids=formatted.demonstration_ids,
                input_tokens=len(record.input_ids) - record.completion_mask.count(1),
                full_tokens=len(record.input_ids),
            )
        )

    return tokenized_records, usages, formatted_examples


class DictDataset(dict):
    """Fallback dataset mapping supporting both column lookups and row index access."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return {col: values[key] for col, values in self.items()}
        return super().__getitem__(key)

    def __len__(self) -> int:
        first_val = next(iter(self.values()), [])
        return len(first_val)


def to_hf_dataset(
    records: Sequence[TokenizedSFTRecord],
) -> Any:
    """Convert tokenized SFT records to a Hugging Face Dataset (or DictDataset).

    The resulting dataset is pre-tokenized. The ``labels`` column already
    contains ``-100`` for prompt and padding positions; downstream collators
    must preserve it rather than reconstructing labels from ``input_ids``.
    """
    data_dict: dict[str, list[Any]] = {
        "case_id": [r.case_id for r in records],
        "input_ids": [r.input_ids for r in records],
        "attention_mask": [r.attention_mask for r in records],
        "completion_mask": [r.completion_mask for r in records],
        "labels": [r.labels for r in records],
    }

    try:
        from datasets import Dataset

        return Dataset.from_dict(data_dict)
    except ImportError:
        return DictDataset(data_dict)
