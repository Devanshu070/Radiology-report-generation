"""Load CSV data and partition into leakage-safe candidate and query sets."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..config import training_data_path
from ..schemas import ClinicalCase, SplitManifest
from .validation import validate_cases


def load_cases(
    path: str | Path | None = None,
    *,
    require_report: bool = True,
) -> list[ClinicalCase]:
    """Read cases from a CSV file into validated ClinicalCase objects.

    Args:
        path: CSV path. If None, defaults to ``CLINICAL_DATA_ROOT/train.csv``.
        require_report: Whether the 'report' column is strictly required.

    Returns:
        List of validated ClinicalCase instances.
    """
    file_path = Path(path) if path is not None else training_data_path()
    if not file_path.is_file():
        raise FileNotFoundError(f"Clinical data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or malformed: {file_path}")
        rows = [row for row in reader]

    return validate_cases(rows, require_report=require_report)


def create_deterministic_splits(
    cases: Iterable[ClinicalCase],
    *,
    seed: int = 42,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.10,
) -> tuple[SplitManifest, list[ClinicalCase], list[ClinicalCase], list[ClinicalCase]]:
    """Partition cases deterministically into train, validation, and test subsets.

    Partitioning is keyed on a SHA-256 digest of ``seed`` and ``case_id``, ensuring
    platform-independent determinism regardless of input row order.

    Args:
        cases: All clinical cases to partition.
        seed: Randomization seed.
        test_fraction: Fraction allocated to test (outer holdout).
        validation_fraction: Fraction of remaining training allocated to validation.

    Returns:
        A tuple of (SplitManifest, train_cases, validation_cases, test_cases).
    """
    case_list = list(cases)
    if len(case_list) < 3:
        raise ValueError("At least three cases are required to construct train, validation, and test splits.")

    by_id: dict[str, ClinicalCase] = {}
    for case in case_list:
        if case.case_id in by_id:
            raise ValueError(f"Duplicate case_id in split pool: {case.case_id}")
        by_id[case.case_id] = case

    # Deterministic hash ordering
    sorted_ids = sorted(
        by_id.keys(),
        key=lambda cid: hashlib.sha256(f"{seed}:{cid}".encode("utf-8")).digest(),
    )

    n_total = len(sorted_ids)
    n_test = max(1, round(n_total * test_fraction))
    n_remaining = n_total - n_test
    if n_remaining < 2:
        raise ValueError("Dataset size too small for specified split fractions.")

    n_val = max(1, round(n_remaining * validation_fraction))
    n_train = n_remaining - n_val
    if n_train < 1:
        raise ValueError("Dataset partition leaves zero training examples.")

    test_ids = sorted_ids[:n_test]
    val_ids = sorted_ids[n_test : n_test + n_val]
    train_ids = sorted_ids[n_test + n_val :]

    assignments: dict[str, str] = {}
    for cid in test_ids:
        assignments[cid] = "test"
    for cid in val_ids:
        assignments[cid] = "validation"
    for cid in train_ids:
        assignments[cid] = "train"

    manifest = SplitManifest(seed=seed, assignments=assignments)  # type: ignore[arg-type]
    train_cases = [by_id[cid] for cid in train_ids]
    val_cases = [by_id[cid] for cid in val_ids]
    test_cases = [by_id[cid] for cid in test_ids]

    return manifest, train_cases, val_cases, test_cases


def filter_training_pool(
    cases: Iterable[ClinicalCase],
    manifest: SplitManifest,
) -> list[ClinicalCase]:
    """Isolate labeled training cases, enforcing zero leakage from val/test sets.

    Raises:
        ValueError: If cases do not match manifest IDs or if any training candidate lacks a report.
    """
    by_id: dict[str, ClinicalCase] = {}
    for case in cases:
        if case.case_id in by_id:
            raise ValueError(f"Duplicate case ID in candidate pool: {case.case_id}")
        by_id[case.case_id] = case

    missing_from_cases = manifest.train_ids - set(by_id.keys())
    if missing_from_cases:
        raise ValueError(f"Cases pool missing {len(missing_from_cases)} IDs present in train manifest.")

    training_cases: list[ClinicalCase] = []
    for cid in sorted(manifest.train_ids):
        case = by_id[cid]
        if not case.report or not case.report.strip():
            raise ValueError(f"Training candidate case '{cid}' lacks reference report.")
        training_cases.append(case)

    return training_cases
