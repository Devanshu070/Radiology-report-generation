"""Fingerprint-checked, atomic local caches for text embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np


def embedding_fingerprint(
    training_texts: list[str],
    query_texts: list[str],
    *,
    model_name: str,
    model_revision: str,
    max_seq_length: int,
    chunk_overlap: int,
) -> str:
    """Hash model/tokenization configuration and ordered text digests only."""
    def text_digest(texts: list[str]) -> str:
        digest = hashlib.sha256()
        for text in texts:
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        return digest.hexdigest()

    descriptor = {
        "cache_format": 1,
        "model_name": model_name,
        "model_revision": model_revision,
        "max_seq_length": max_seq_length,
        "chunk_overlap": chunk_overlap,
        "training_count": len(training_texts),
        "query_count": len(query_texts),
        "training_digest": text_digest(training_texts),
        "query_digest": text_digest(query_texts),
    }
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedding_cache_path(cache_dir: str | Path, fingerprint: str) -> Path:
    """Return a content-addressed cache path without exposing input text."""
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Embedding cache fingerprints must be 64-character lowercase SHA-256 digests.")
    return Path(cache_dir) / f"embeddings-{fingerprint}.npz"


def load_embedding_cache(path: str | Path, fingerprint: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a valid cache pair, treating corrupt or stale files as cache misses."""
    cache_path = Path(path)
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as stored:
            if str(stored["fingerprint"].item()) != fingerprint:
                return None
            training = np.asarray(stored["training"], dtype=np.float32)
            queries = np.asarray(stored["queries"], dtype=np.float32)
        if training.ndim != 2 or queries.ndim != 2 or training.shape[1] != queries.shape[1]:
            return None
        if not np.isfinite(training).all() or not np.isfinite(queries).all():
            return None
        return training, queries
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def save_embedding_cache(
    path: str | Path,
    fingerprint: str,
    training: np.ndarray,
    queries: np.ndarray,
) -> None:
    """Atomically persist vectors and a fingerprint, never raw clinical text."""
    cache_path = Path(path)
    training = np.asarray(training, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if training.ndim != 2 or queries.ndim != 2 or training.shape[1] != queries.shape[1]:
        raise ValueError("Training and query embeddings must be compatible two-dimensional arrays.")
    if not np.isfinite(training).all() or not np.isfinite(queries).all():
        raise ValueError("Embedding cache cannot store non-finite vectors.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{cache_path.name}.", suffix=".tmp", dir=cache_path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(
                handle,
                fingerprint=np.asarray(fingerprint),
                training=training,
                queries=queries,
            )
        os.replace(temporary_name, cache_path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
