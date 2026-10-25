"""PubMedBERT text embedder with sliding-window chunk pooling and dimension validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    EXPECTED_EMBEDDING_DIM,
)
from .chunk_pooling import (
    aggregate_chunk_vectors,
    compute_chunk_weights,
    validate_embedding_matrix,
)


class PubMedBERTEmbedder:
    """PubMedBERT text embedder supporting long clinical text chunking and pooling.

    Allows injection of custom models or encoder functions for offline unit testing
    without external network or heavy weight downloads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        revision: str = DEFAULT_EMBEDDING_REVISION,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        model: Any | None = None,
        encoder_fn: Callable[[list[str]], np.ndarray] | None = None,
    ):
        self.model_name = model_name
        self.revision = revision
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.device = device
        self._model = model
        self._encoder_fn = encoder_fn

        if self.chunk_size <= 2:
            raise ValueError(f"chunk_size must be > 2, got {chunk_size}")
        if not (0 <= self.chunk_overlap < self.chunk_size - 2):
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must satisfy 0 <= overlap < chunk_size - 2 ({chunk_size - 2})"
            )

    def _get_model(self) -> Any:
        """Lazy-load the underlying SentenceTransformer model if not injected."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            target_device = self.device
            if target_device is None:
                if torch.cuda.is_available():
                    target_device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"

            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                device=target_device,
            )
            self._model.eval()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode an ordered list of clinical texts into normalized 768-d vectors.

        Args:
            texts: List of clinical dictations or reports.

        Returns:
            Float32 numpy array of shape (len(texts), 768).
        """
        if not texts:
            return np.zeros((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)

        # Delegated / injected custom encoder callable
        if self._encoder_fn is not None:
            matrix = self._encoder_fn(texts)
            return validate_embedding_matrix(matrix, expected_dim=EXPECTED_EMBEDDING_DIM)

        model = self._get_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            # Fallback if injected model is a simple embedding callable
            if hasattr(model, "encode") and callable(model.encode):
                raw = model.encode(texts)
                return validate_embedding_matrix(np.asarray(raw), expected_dim=EXPECTED_EMBEDDING_DIM)
            raise AttributeError("Injected model does not provide a tokenizer or encode method.")

        import torch

        # Tokenize with sliding-window chunking
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.chunk_size,
            stride=self.chunk_overlap,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        sample_mapping = np.asarray(encoded.pop("overflow_to_sample_mapping"), dtype=np.int64)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        chunk_weights = compute_chunk_weights(
            encoded["attention_mask"],
            sample_mapping,
            num_special_tokens=special_tokens,
            chunk_overlap=self.chunk_overlap,
        )

        chunk_vectors_list: list[np.ndarray] = []
        model_device = getattr(model, "device", "cpu")

        model.eval()
        with torch.inference_mode():
            for start in range(0, len(sample_mapping), self.batch_size):
                batch_dict = {
                    k: v[start : start + self.batch_size]
                    for k, v in encoded.items()
                }
                padded = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
                features = {k: v.to(model_device) for k, v in padded.items()}
                output = model(features)
                if isinstance(output, dict) and "sentence_embedding" in output:
                    embeddings = output["sentence_embedding"]
                elif hasattr(output, "last_hidden_state"):
                    embeddings = output.last_hidden_state[:, 0, :]
                else:
                    embeddings = output

                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                chunk_vectors_list.append(normalized.float().cpu().numpy())

        all_chunk_vectors = np.concatenate(chunk_vectors_list, axis=0) if chunk_vectors_list else np.zeros((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)

        pooled = aggregate_chunk_vectors(
            all_chunk_vectors,
            chunk_weights,
            sample_mapping,
            num_samples=len(texts),
            expected_dim=EXPECTED_EMBEDDING_DIM,
        )

        return validate_embedding_matrix(pooled, expected_dim=EXPECTED_EMBEDDING_DIM)
