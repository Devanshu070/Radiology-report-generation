"""Stable train/validation/test assignments and leakage-safe training pools."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..schemas import CaseRecord, SplitManifest


def create_split_manifest(cases: Iterable[CaseRecord], *, seed: int = 42) -> SplitManifest:
    """Assign cases deterministically to disjoint 80/10/10 partitions.

    Case identifiers are hash-ranked rather than relying on source CSV order.
    At least three records are required so each partition is represented.
    """
    unique_cases: dict[str, CaseRecord] = {}
    for case in cases:
        if case.case_id in unique_cases:
            raise ValueError(f"Duplicate case_id in split input: {case.case_id}")
        unique_cases[case.case_id] = case
    case_ids = list(unique_cases)
    if len(case_ids) < 3:
        raise ValueError("At least three cases are required for nonempty train/validation/test splits.")

    ranked = sorted(
        case_ids,
        key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest(),
    )
    test_count = max(1, round(len(ranked) * 0.1))
    validation_count = max(1, round(len(ranked) * 0.1))
    train_count = len(ranked) - test_count - validation_count
    if train_count < 1:
        raise ValueError("Split ratios leave no training cases.")

    assignments = {case_id: "train" for case_id in ranked[:train_count]}
    assignments.update({case_id: "validation" for case_id in ranked[train_count:train_count + validation_count]})
    assignments.update({case_id: "test" for case_id in ranked[train_count + validation_count:]})
    return SplitManifest(seed=seed, assignments=assignments)


def training_pool(cases: Iterable[CaseRecord], manifest: SplitManifest) -> list[CaseRecord]:
    """Return labeled training cases only; reject manifest/data mismatches."""
    case_map: dict[str, CaseRecord] = {}
    case_order: list[str] = []
    for case in cases:
        if case.case_id in case_map:
            raise ValueError(f"Duplicate case_id in pool input: {case.case_id}")
        case_map[case.case_id] = case
        case_order.append(case.case_id)
    if set(case_map) != set(manifest.assignments):
        raise ValueError("Split manifest case IDs must exactly match the supplied records.")
    return [case_map[case_id] for case_id in case_order if case_id in manifest.train_ids and case_map[case_id].report]
