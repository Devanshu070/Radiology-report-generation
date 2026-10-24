"""Hybrid biomedical retrieval engine combining PubMedBERT, TF-IDF, and metadata."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import (
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    EXPECTED_EMBEDDING_DIM,
)
from ..schemas import (
    ClinicalCase,
    RetrievalCandidate,
    RetrievalResult,
    SplitManifest,
)
from .cache_manifest import EmbeddingCacheManager
from .pubmedbert_embeddings import PubMedBERTEmbedder


class HybridRetriever:
    """Ensemble biomedical retriever with strict train/val/test leakage prevention.

    Combines:
      - Dense semantic similarity via PubMedBERT chunk-weighted embeddings (0.60 default).
      - Sparse lexical keyword similarity via TF-IDF (1-2 n-grams) (0.20 default).
      - Exact clinical modality match (0.10 default).
      - Exact anatomical body-part match (0.10 default).
    """

    def __init__(
        self,
        candidate_pool: Sequence[ClinicalCase],
        *,
        manifest: SplitManifest,
        weights: dict[str, float] | None = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.candidate_pool = list(candidate_pool)
        self.manifest = manifest
        self.top_k = top_k
        self.weights = dict(weights or DEFAULT_WEIGHTS)

        if not isinstance(manifest, SplitManifest):
            raise TypeError("A valid SplitManifest must be provided to enforce leakage protection.")

        if not self.candidate_pool:
            raise ValueError("Candidate pool cannot be empty.")

        # Validate weights
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-5):
            raise ValueError(f"Retrieval weights must sum to 1.0, got {total_weight}")
        if any(w < 0.0 for w in self.weights.values()):
            raise ValueError("Retrieval weights must be non-negative.")

        # Candidate pool leakage verification
        candidate_ids = [c.case_id for c in self.candidate_pool]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Candidate pool contains duplicate case IDs.")

        for c in self.candidate_pool:
            if not c.report or not c.report.strip():
                raise ValueError(
                    f"Candidate case '{c.case_id}' lacks a labeled reference report and cannot serve as demonstration."
                )

        invalid_ids = set(candidate_ids) - self.manifest.train_ids
        if invalid_ids:
            raise ValueError(
                f"Data leakage detected! Candidate pool contains non-training case IDs: {sorted(invalid_ids)}"
            )

        # Fit TF-IDF on candidate pool dictations
        self._tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        self._pool_tfidf = self._tfidf.fit_transform([c.dictation for c in self.candidate_pool])

    @staticmethod
    def _metadata_matrix(queries: list[ClinicalCase], candidates: list[ClinicalCase], attr: str) -> np.ndarray:
        """Compute binary match matrix for categorical clinical attributes."""
        q_vals = [str(getattr(c, attr)).strip().casefold() for c in queries]
        c_vals = [str(getattr(c, attr)).strip().casefold() for c in candidates]
        return np.asarray([
            [float(bool(q) and q == c) for c in c_vals]
            for q in q_vals
        ], dtype=np.float32)

    def retrieve(
        self,
        query_cases: Sequence[ClinicalCase],
        *,
        query_embeddings: np.ndarray | None = None,
        candidate_embeddings: np.ndarray | None = None,
        cache_manager: EmbeddingCacheManager | None = None,
        embedder: PubMedBERTEmbedder | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k highest scoring training cases for each query case.

        Args:
            query_cases: Query cases needing demonstrations.
            query_embeddings: Precomputed 768-d query embeddings.
            candidate_embeddings: Precomputed 768-d candidate embeddings.
            cache_manager: Cache manager to obtain embeddings if not precomputed.
            embedder: PubMedBERTEmbedder instance for missing vector generation.
            top_k: Number of demonstrations to retrieve per query (default: self.top_k).

        Returns:
            List of RetrievalResult instances, one per query case in exact input order.
        """
        queries = list(query_cases)
        if not queries:
            return []

        k = top_k if top_k is not None else self.top_k
        if not (1 <= k <= len(self.candidate_pool)):
            k = max(1, min(k, len(self.candidate_pool)))

        # Obtain candidate embeddings
        if candidate_embeddings is None:
            if cache_manager is None or embedder is None:
                raise ValueError("Must provide either precomputed candidate_embeddings or both cache_manager and embedder.")
            candidate_embeddings = cache_manager.get_or_compute_embeddings(
                [c.dictation for c in self.candidate_pool],
                embedder,
            )

        if candidate_embeddings.shape != (len(self.candidate_pool), EXPECTED_EMBEDDING_DIM):
            raise ValueError(
                f"Candidate embeddings shape mismatch: {candidate_embeddings.shape}, expected ({len(self.candidate_pool)}, {EXPECTED_EMBEDDING_DIM})"
            )

        # Obtain query embeddings
        if query_embeddings is None:
            if cache_manager is None or embedder is None:
                raise ValueError("Must provide either precomputed query_embeddings or both cache_manager and embedder.")
            query_embeddings = cache_manager.get_or_compute_embeddings(
                [q.dictation for q in queries],
                embedder,
            )

        if query_embeddings.shape != (len(queries), EXPECTED_EMBEDDING_DIM):
            raise ValueError(
                f"Query embeddings shape mismatch: {query_embeddings.shape}, expected ({len(queries)}, {EXPECTED_EMBEDDING_DIM})"
            )

        # 1. Dense Semantic Similarity (dot product of unit vectors)
        dense_sim = np.clip(query_embeddings @ candidate_embeddings.T, 0.0, 1.0)

        # 2. Sparse Lexical Similarity (TF-IDF cosine similarity)
        query_tfidf = self._tfidf.transform([q.dictation for q in queries])
        keyword_sim = np.clip(cosine_similarity(query_tfidf, self._pool_tfidf), 0.0, 1.0)

        # 3. Categorical Modality Match
        modality_sim = self._metadata_matrix(queries, self.candidate_pool, "modality")

        # 4. Categorical Body Part Match
        body_part_sim = self._metadata_matrix(queries, self.candidate_pool, "body_part")

        # Linear Ensemble
        combined_scores = (
            self.weights.get("embedding_similarity", 0.0) * dense_sim
            + self.weights.get("keyword_similarity", 0.0) * keyword_sim
            + self.weights.get("modality_match", 0.0) * modality_sim
            + self.weights.get("body_part_match", 0.0) * body_part_sim
        )

        # Self-Retrieval Prevention: Training query cases cannot retrieve themselves
        for i, query in enumerate(queries):
            for j, candidate in enumerate(self.candidate_pool):
                if query.case_id == candidate.case_id:
                    combined_scores[i, j] = -np.inf

        # Rank candidates per query case
        results: list[RetrievalResult] = []
        for i, query in enumerate(queries):
            scores_row = combined_scores[i]
            # Stable sort in descending order
            top_indices = np.argsort(-scores_row, kind="stable")[:k]

            candidates_list: list[RetrievalCandidate] = []
            for rank, idx in enumerate(top_indices, start=1):
                if np.isneginf(scores_row[idx]):
                    continue
                cand_case = self.candidate_pool[idx]
                candidates_list.append(
                    RetrievalCandidate(
                        query_case_id=query.case_id,
                        rank=rank,
                        candidate_case_id=cand_case.case_id,
                        score=float(scores_row[idx]),
                        embedding_similarity=float(dense_sim[i, idx]),
                        keyword_similarity=float(keyword_sim[i, idx]),
                        modality_match=float(modality_sim[i, idx]),
                        body_part_match=float(body_part_sim[i, idx]),
                    )
                )
            results.append(RetrievalResult(query_case_id=query.case_id, candidates=candidates_list))

        return results
