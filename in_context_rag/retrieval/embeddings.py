"""Chunk-aware text embeddings and fingerprinted reuse for few-shot search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .cache import embedding_cache_path, embedding_fingerprint, load_embedding_cache, save_embedding_cache


DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
DEFAULT_MODEL_REVISION = "main"
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SEQ_LENGTH = 512


def encode_texts(
    texts: list[str],
    model: Any,
    *,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode long inputs in overlapping chunks and token-weight their vectors."""
    if not texts:
        dimension = model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("The embedding model did not provide its output dimension.")
        return np.zeros((0, int(dimension)), dtype=np.float32)
    if chunk_overlap < 0 or chunk_overlap >= model.max_seq_length or batch_size < 1:
        raise ValueError("chunk_overlap must be smaller than max_seq_length and batch_size must be positive.")

    import torch

    tokenizer = model.tokenizer
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=model.max_seq_length,
        stride=chunk_overlap,
        return_overflowing_tokens=True,
        return_attention_mask=True,
    )
    owners = np.asarray(encoded.pop("overflow_to_sample_mapping"), dtype=np.int64)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    chunk_weights = np.asarray(
        [max(1, int(sum(mask)) - special_tokens) for mask in encoded["attention_mask"]],
        dtype=np.float32,
    )
    overlap_positions = np.flatnonzero(owners[1:] == owners[:-1]) + 1
    chunk_weights[overlap_positions] = np.maximum(
        1.0, chunk_weights[overlap_positions] - chunk_overlap
    )

    vectors: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(owners), batch_size):
            end = start + batch_size
            batch = tokenizer.pad(
                {key: values[start:end] for key, values in encoded.items()},
                padding=True,
                return_tensors="pt",
            )
            features = {key: value.to(model.device) for key, value in batch.items()}
            output = model(features)
            sentence_vectors = output["sentence_embedding"]
            normalized = torch.nn.functional.normalize(sentence_vectors, p=2, dim=1)
            vectors.append(normalized.float().cpu().numpy())

    chunk_vectors = np.concatenate(vectors, axis=0)
    dimension = model.get_embedding_dimension()
    if dimension is None or chunk_vectors.shape[1] != int(dimension):
        raise ValueError("Embedding output dimension is missing or inconsistent with the model.")
    document_vectors = np.zeros((len(texts), int(dimension)), dtype=np.float32)
    np.add.at(document_vectors, owners, chunk_vectors * chunk_weights[:, None])
    norms = np.linalg.norm(document_vectors, axis=1, keepdims=True)
    document_vectors /= np.maximum(norms, 1e-12)
    return document_vectors


def load_or_create_embeddings(
    training_texts: list[str],
    query_texts: list[str],
    *,
    cache_dir: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str = DEFAULT_MODEL_REVISION,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached vectors or encode and atomically cache both case groups."""
    fingerprint = embedding_fingerprint(
        training_texts,
        query_texts,
        model_name=model_name,
        model_revision=model_revision,
        max_seq_length=max_seq_length,
        chunk_overlap=chunk_overlap,
    )
    path = embedding_cache_path(cache_dir, fingerprint)
    cached = load_embedding_cache(path, fingerprint)
    if cached is not None:
        training_vectors, query_vectors = cached
        if len(training_vectors) == len(training_texts) and len(query_vectors) == len(query_texts):
            return training_vectors, query_vectors

    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the base requirements to use PubMedBERT few-shot retrieval."
            ) from error
        model = SentenceTransformer(model_name, revision=model_revision)
    model.max_seq_length = max_seq_length
    training_vectors = encode_texts(training_texts, model, chunk_overlap=chunk_overlap, batch_size=batch_size)
    query_vectors = encode_texts(query_texts, model, chunk_overlap=chunk_overlap, batch_size=batch_size)
    if training_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Training and query embedding dimensions differ.")
    save_embedding_cache(path, fingerprint, training_vectors, query_vectors)
    return training_vectors, query_vectors
