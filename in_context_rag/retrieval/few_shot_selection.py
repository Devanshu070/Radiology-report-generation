"""Leakage-safe hybrid ranking for in-context demonstration selection."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import REPOSITORY_ROOT
from ..preprocess.split import training_pool
from ..schemas import CaseRecord, FewShotPair, SplitManifest
from .embeddings import DEFAULT_EMBEDDING_MODEL, load_or_create_embeddings


DEFAULT_TOP_K = 5
DEFAULT_WEIGHTS = {
    "embedding": 0.60,
    "keyword": 0.20,
    "modality": 0.10,
    "body_part": 0.10,
}


def _metadata_matches(training: list[CaseRecord], queries: list[CaseRecord], field: str) -> np.ndarray:
    training_values = [str(getattr(case, field)).strip().casefold() for case in training]
    query_values = [str(getattr(case, field)).strip().casefold() for case in queries]
    return np.asarray([
        [float(bool(query) and query == candidate) for candidate in training_values]
        for query in query_values
    ], dtype=np.float32)


def select_few_shot_examples(
    cases: Iterable[CaseRecord],
    query_cases: Iterable[CaseRecord],
    manifest: SplitManifest,
    *,
    top_k: int = DEFAULT_TOP_K,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path | None = None,
    embedding_matrices: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, list[FewShotPair]]:
    """Select at most five demonstrations, exclusively from labeled train IDs.

    ``embedding_matrices`` is an injection point for deterministic tests and
    callers with precomputed vectors; production calls use the local embedding
    cache and PubMedBERT.
    """
    all_cases = list(cases)
    queries = list(query_cases)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= DEFAULT_TOP_K:
        raise ValueError(f"top_k must be between 1 and {DEFAULT_TOP_K}.")
    if not queries:
        return {}
    if len({case.case_id for case in queries}) != len(queries):
        raise ValueError("Query cases must have unique case IDs.")
    by_id: dict[str, CaseRecord] = {}
    for case in all_cases:
        if case.case_id in by_id:
            raise ValueError(f"Duplicate case_id in retrieval input: {case.case_id}")
        by_id[case.case_id] = case
    if not set(by_id).issubset(manifest.assignments):
        raise ValueError("Every supplied case must be present in the split manifest.")
    if any(query.case_id not in manifest.assignments for query in queries):
        raise ValueError("Every query case must be present in the split manifest.")

    candidates = training_pool(all_cases, manifest)
    if not candidates:
        raise ValueError("No labeled training cases are available for few-shot selection.")
    candidate_ids = {case.case_id for case in candidates}
    if not candidate_ids.issubset(manifest.train_ids):
        raise ValueError("Few-shot candidate pool contains a non-training case.")

    training_texts = [case.dictation for case in candidates]
    query_texts = [case.dictation for case in queries]
    if embedding_matrices is None:
        embedding_matrices = load_or_create_embeddings(
            training_texts,
            query_texts,
            cache_dir=cache_dir or REPOSITORY_ROOT / "artifacts" / "in_context_rag" / "embeddings",
            model_name=model_name,
        )
    training_vectors, query_vectors = (np.asarray(matrix, dtype=np.float32) for matrix in embedding_matrices)
    if (
        training_vectors.ndim != 2
        or query_vectors.ndim != 2
        or training_vectors.shape[0] != len(candidates)
        or query_vectors.shape[0] != len(queries)
        or training_vectors.shape[1] != query_vectors.shape[1]
    ):
        raise ValueError("Embedding matrices must have one compatible vector per case.")

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    training_keywords = vectorizer.fit_transform(training_texts)
    query_keywords = vectorizer.transform(query_texts)
    components = {
        "embedding": np.clip(query_vectors @ training_vectors.T, 0.0, 1.0),
        "keyword": np.clip(cosine_similarity(query_keywords, training_keywords), 0.0, 1.0),
        "modality": _metadata_matches(candidates, queries, "modality"),
        "body_part": _metadata_matches(candidates, queries, "body_part"),
    }
    combined = sum(DEFAULT_WEIGHTS[name] * values for name, values in components.items())

    result: dict[str, list[FewShotPair]] = {}
    for query_index, query in enumerate(queries):
        scores = combined[query_index].copy()
        for candidate_index, candidate in enumerate(candidates):
            if candidate.case_id == query.case_id:
                scores[candidate_index] = -np.inf
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (-float(scores[index]), candidates[index].case_id),
        )
        selected: list[FewShotPair] = []
        for candidate_index in ranked:
            if not np.isfinite(scores[candidate_index]):
                continue
            candidate = candidates[candidate_index]
            if not candidate.report:
                continue
            selected.append(FewShotPair(
                case_id=candidate.case_id,
                modality=candidate.modality,
                body_part=candidate.body_part,
                study_description=candidate.study_description,
                patient_age_band=candidate.patient_age_band,
                patient_sex=candidate.patient_sex,
                dictation=candidate.dictation,
                template_content=candidate.template_content,
                report=candidate.report,
            ))
            if len(selected) == top_k:
                break
        result[query.case_id] = selected
    return result
