r"""Sliding-window chunk token-weighting and aggregation for clinical text embeddings.

Clinical Background & Mathematical Formulation
----------------------------------------------
Clinical radiology dictations and full reports regularly exceed the 512-token maximum sequence
length supported by pretrained PubMedBERT architecture. Truncating text at token 512 discards
critical diagnostic conclusions often recorded toward the end of an examination (such as the
IMPRESSION section or concluding observations).

To capture the entire text without loss while maintaining dense representation fidelity:
1. Tokenization with Stride:
   The text is split into chunks of maximum length :math:`L = 512` tokens using a sliding window
   with stride :math:`S_{\text{stride}} = L - O = 448` tokens, creating an overlap of :math:`O = 64`
   tokens between consecutive windows of the same text.

2. Unique Token Weighting (Overlap Discount):
   Each chunk :math:`c` possesses an attention mask indicating active tokens. PubMedBERT uses
   two special tokens (``[CLS]`` and ``[SEP]``) per chunk (:math:`S_{\text{special}} = 2`).
   The active content token count is:
   .. math::
       N_c = \sum_{j} m_{c, j} - S_{\text{special}}

   For the initial chunk of a sample (:math:`c = 0`):
   .. math::
       w_0 = N_0

   For subsequent overflowing chunks (:math:`c > 0`) of the same sample:
   The first :math:`O = 64` tokens duplicate the trailing tokens of chunk :math:`c - 1`.
   To prevent double-counting shared token representations in the document embedding, the overlap
   is subtracted from the chunk weight:
   .. math::
       w_c = \max(N_c - O, 0.0)

3. Chunk Normalization:
   Each chunk vector :math:`\mathbf{v}_c \in \mathbb{R}^{768}` output by the sentence encoder is
   individually projected onto the unit hypersphere:
   .. math::
       \hat{\mathbf{v}}_c = \frac{\mathbf{v}_c}{\max(\|\mathbf{v}_c\|_2, 10^{-12})}

4. Weighted Accumulation:
   The case vector :math:`\mathbf{u}_i` is formed as the linear combination of unit chunk vectors,
   weighted by each chunk's contribution of unique tokens:
   .. math::
       \mathbf{u}_i = \sum_{c \in \text{chunks}(i)} w_c \hat{\mathbf{v}}_c

5. Document Normalization:
   The accumulated vector is normalized to unit length:
   .. math::
       \mathbf{e}_i = \frac{\mathbf{u}_i}{\max(\|\mathbf{u}_i\|_2, 10^{-12})}

This guarantees that cosine similarity between any two cases :math:`A` and :math:`B` is
computed directly by the inner product :math:`\mathbf{e}_A \cdot \mathbf{e}_B`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_chunk_weights(
    attention_masks: Sequence[Sequence[int]] | np.ndarray,
    sample_mapping: Sequence[int] | np.ndarray,
    *,
    num_special_tokens: int = 2,
    chunk_overlap: int = 64,
) -> np.ndarray:
    """Calculate token weights for each chunk, subtracting overlap on continued chunks.

    Args:
        attention_masks: Attention mask array/lists for all tokenized chunks.
        sample_mapping: Mapping of chunk index to sample (document) index.
        num_special_tokens: Count of special tokens per chunk (typically 2 for [CLS], [SEP]).
        chunk_overlap: Number of tokens overlapping with the prior chunk of the same sample.

    Returns:
        1D float32 numpy array of non-negative weights for each chunk.
    """
    if len(attention_masks) == 0:
        return np.zeros((0,), dtype=np.float32)

    owners = np.asarray(sample_mapping, dtype=np.int64)
    if len(owners) != len(attention_masks):
        raise ValueError(
            f"Length mismatch: {len(attention_masks)} attention masks but {len(owners)} sample mapping entries."
        )

    raw_weights = np.asarray(
        [max(0, int(np.sum(mask)) - num_special_tokens) for mask in attention_masks],
        dtype=np.float32,
    )

    if len(owners) > 1:
        # Subtract chunk_overlap from chunk c if chunk c and c-1 belong to the same document
        consecutive_same_owner = owners[1:] == owners[:-1]
        raw_weights[1:] -= chunk_overlap * consecutive_same_owner
        np.maximum(raw_weights, 0.0, out=raw_weights)

    return raw_weights


def aggregate_chunk_vectors(
    chunk_vectors: np.ndarray,
    chunk_weights: np.ndarray,
    sample_mapping: np.ndarray,
    num_samples: int,
    *,
    expected_dim: int = 768,
) -> np.ndarray:
    """Aggregate unit-normalized chunk embeddings into one normalized vector per sample.

    Args:
        chunk_vectors: Array of chunk embeddings of shape (N_chunks, D).
        chunk_weights: Array of scalar weights of shape (N_chunks,).
        sample_mapping: Array mapping each chunk to its sample index of shape (N_chunks,).
        num_samples: Total number of parent samples (texts).
        expected_dim: Expected embedding dimension (default 768).

    Returns:
        Float32 array of shape (num_samples, expected_dim) where each row has unit L2 norm.
    """
    if num_samples < 0:
        raise ValueError("num_samples cannot be negative.")
    if num_samples == 0:
        return np.zeros((0, expected_dim), dtype=np.float32)

    if chunk_vectors.ndim != 2 or chunk_vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Expected chunk_vectors of shape (*, {expected_dim}), got {chunk_vectors.shape}"
        )
    if len(chunk_weights) != len(chunk_vectors):
        raise ValueError(
            f"Length mismatch: {len(chunk_vectors)} chunk vectors but {len(chunk_weights)} weights."
        )
    if len(sample_mapping) != len(chunk_vectors):
        raise ValueError(
            f"Length mismatch: {len(chunk_vectors)} chunk vectors but {len(sample_mapping)} mapping items."
        )

    # Normalize individual chunk vectors to unit length
    norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    unit_chunks = chunk_vectors / np.maximum(norms, 1e-12)

    # Weighted accumulation by parent sample index
    weighted_chunks = unit_chunks * chunk_weights[:, np.newaxis]
    aggregated = np.zeros((num_samples, expected_dim), dtype=np.float32)
    np.add.at(aggregated, sample_mapping, weighted_chunks)

    # Normalize final aggregated vectors to unit length
    doc_norms = np.linalg.norm(aggregated, axis=1, keepdims=True)
    aggregated /= np.maximum(doc_norms, 1e-12)

    return aggregated


def validate_embedding_matrix(matrix: np.ndarray, *, expected_dim: int = 768) -> np.ndarray:
    """Validate that an embedding matrix has shape (*, expected_dim), is finite, and non-zero.

    Raises:
        ValueError: If shape is invalid, NaN/Inf values are present, or rows have zero magnitude.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(matrix)}")

    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-dimensional embedding matrix, got ndim={matrix.ndim}")

    if matrix.shape[1] != expected_dim:
        raise ValueError(
            f"Incompatible embedding dimension: expected {expected_dim}, got {matrix.shape[1]}"
        )

    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains non-finite (NaN or Inf) values.")

    if matrix.shape[0] > 0:
        norms = np.linalg.norm(matrix, axis=1)
        if (norms <= 1e-7).any():
            zero_indices = np.where(norms <= 1e-7)[0].tolist()
            raise ValueError(f"Embedding matrix contains zero-norm vector(s) at row(s): {zero_indices}")

    return matrix.astype(np.float32)
