"""Persistent vector cache and manifest verification for PubMedBERT embeddings."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from ..config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_POOLING_METHOD,
    EXPECTED_EMBEDDING_DIM,
)


class IncompatibleCacheError(ValueError):
    """Raised when an existing cache directory configuration conflicts with current settings."""


class InvalidCacheVectorError(ValueError):
    """Raised when a cached vector file has invalid shape, non-finite values, or zero norm."""


class EmbeddingSpec(BaseModel):
    """Canonical descriptor for cache identity and compatibility verification."""

    model_config = ConfigDict(extra="forbid")

    model: str = DEFAULT_EMBEDDING_MODEL
    revision: str = DEFAULT_EMBEDDING_REVISION
    dimension: int = EXPECTED_EMBEDDING_DIM
    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_CHUNK_OVERLAP
    pooling: str = DEFAULT_POOLING_METHOD


def compute_cache_key(text: str, spec: EmbeddingSpec) -> str:
    """Generate a deterministic SHA-256 cache key from embedding spec and input text."""
    payload = {
        **spec.model_dump(),
        "text": text,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cached_vector(path: str | Path, *, expected_dim: int = EXPECTED_EMBEDDING_DIM) -> np.ndarray:
    """Load and validate an individual .npy vector file.

    Raises:
        InvalidCacheVectorError: If shape != (expected_dim,), contains NaN/Inf, or norm is zero.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Cache vector file not found: {file_path}")

    try:
        vector = np.load(file_path, allow_pickle=False)
    except Exception as exc:
        raise InvalidCacheVectorError(f"Corrupt .npy vector at {file_path}: {exc}") from exc

    if not isinstance(vector, np.ndarray):
        raise InvalidCacheVectorError(f"File {file_path} did not produce a numpy array.")

    if vector.shape != (expected_dim,):
        raise InvalidCacheVectorError(
            f"Cached vector at {file_path} has incompatible dimension {vector.shape}; expected ({expected_dim},)"
        )

    if not np.isfinite(vector).all():
        raise InvalidCacheVectorError(f"Cached vector at {file_path} contains non-finite (NaN/Inf) values.")

    norm = np.linalg.norm(vector)
    if norm <= 1e-7:
        raise InvalidCacheVectorError(f"Cached vector at {file_path} has zero norm ({norm}).")

    return vector.astype(np.float32)


class EmbedderCallable(Protocol):
    """Protocol for injectable text embedders."""

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class EmbeddingCacheManager:
    """Manages content-addressed .npy vector caching with compatibility validation."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        *,
        spec: EmbeddingSpec | None = None,
        extra_cache_dirs: list[str | Path] | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.spec = spec or EmbeddingSpec()
        self.extra_cache_dirs = [Path(p) for p in (extra_cache_dirs or [])]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verify_or_write_spec()

    def spec_file_path(self) -> Path:
        """Path to the cache directory's specification file."""
        return self.cache_dir / "embedding_spec.json"

    def verify_or_write_spec(self) -> None:
        """Verify compatibility with existing spec file, or write active spec if absent."""
        spec_file = self.spec_file_path()
        if spec_file.is_file():
            try:
                stored_spec = json.loads(spec_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise IncompatibleCacheError(f"Failed to read cache spec file {spec_file}: {exc}") from exc

            current_dict = self.spec.model_dump()
            mismatches = {
                k: (stored_spec.get(k), current_dict[k])
                for k in current_dict
                if stored_spec.get(k) != current_dict[k]
            }
            if mismatches:
                raise IncompatibleCacheError(
                    f"Cache at {self.cache_dir} is incompatible with current spec: mismatches={mismatches}"
                )
        else:
            # Write initial spec file atomically
            tmp_spec = spec_file.with_suffix(".tmp")
            tmp_spec.write_text(json.dumps(self.spec.model_dump(), indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_spec, spec_file)

    def vector_path(self, text: str) -> Path:
        """Get the primary cache file path for a given text."""
        key = compute_cache_key(text, self.spec)
        return self.cache_dir / f"{key}.npy"

    def lookup_vector(self, text: str) -> np.ndarray | None:
        """Lookup a valid vector for the given text across primary and extra cache dirs."""
        key = compute_cache_key(text, self.spec)
        primary_path = self.cache_dir / f"{key}.npy"

        # Check primary cache
        if primary_path.is_file():
            try:
                return validate_cached_vector(primary_path, expected_dim=self.spec.dimension)
            except InvalidCacheVectorError:
                primary_path.unlink(missing_ok=True)

        # Check secondary / extra cache directories
        for extra_dir in self.extra_cache_dirs:
            extra_path = extra_dir / f"{key}.npy"
            if extra_path.is_file():
                try:
                    vector = validate_cached_vector(extra_path, expected_dim=self.spec.dimension)
                    # Copy to primary cache for future lookups
                    self.save_vector(text, vector)
                    return vector
                except (InvalidCacheVectorError, OSError):
                    pass

        return None

    def save_vector(self, text: str, vector: np.ndarray) -> Path:
        """Atomically persist an embedding vector to a .npy file.

        Raises:
            ValueError: If vector dimension != spec.dimension or contains non-finite values.
        """
        if vector.shape != (self.spec.dimension,):
            raise ValueError(
                f"Cannot save vector with shape {vector.shape}; expected ({self.spec.dimension},)"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Cannot cache non-finite vector.")
        if np.linalg.norm(vector) <= 1e-7:
            raise ValueError("Cannot cache zero-norm vector.")

        key = compute_cache_key(text, self.spec)
        target_path = self.cache_dir / f"{key}.npy"
        temp_path = self.cache_dir / f"{key}.{os.getpid()}.tmp"

        with temp_path.open("wb") as handle:
            np.save(handle, vector.astype(np.float32), allow_pickle=False)

        os.replace(temp_path, target_path)
        return target_path

    def get_or_compute_embeddings(
        self,
        texts: list[str],
        embedder: Any,
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Retrieve embeddings from cache, computing and caching only missing items.

        Args:
            texts: List of clinical texts to encode.
            embedder: Object or callable with an ``encode(texts: list[str]) -> np.ndarray`` method.
            batch_size: Batch size for encoding missing texts.

        Returns:
            2D numpy array of shape (len(texts), 768), exactly aligned with input order.
        """
        if not texts:
            return np.zeros((0, self.spec.dimension), dtype=np.float32)

        # Discover unique texts and check cache
        cached_vectors: dict[str, np.ndarray] = {}
        unique_texts = list(dict.fromkeys(texts))

        for text in unique_texts:
            vec = self.lookup_vector(text)
            if vec is not None:
                cached_vectors[text] = vec

        missing_texts = [text for text in unique_texts if text not in cached_vectors]

        # Compute missing texts if any
        if missing_texts:
            for start_idx in range(0, len(missing_texts), batch_size):
                batch = missing_texts[start_idx : start_idx + batch_size]
                if hasattr(embedder, "encode") and callable(embedder.encode):
                    batch_vectors = embedder.encode(batch)
                elif callable(embedder):
                    batch_vectors = embedder(batch)
                else:
                    raise TypeError(f"Embedder {type(embedder)} must have an encode method or be callable.")

                if batch_vectors.shape != (len(batch), self.spec.dimension):
                    raise ValueError(
                        f"Embedder returned shape {batch_vectors.shape}; expected ({len(batch)}, {self.spec.dimension})"
                    )

                for txt, vec in zip(batch, batch_vectors):
                    self.save_vector(txt, vec)
                    cached_vectors[txt] = vec.astype(np.float32)

        # Build output matrix deterministically aligned with texts
        aligned = np.stack([cached_vectors[text] for text in texts], axis=0)
        return aligned.astype(np.float32)
