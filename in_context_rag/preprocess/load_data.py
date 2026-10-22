"""Load local CSV cases without embedding private data in the package."""

from __future__ import annotations

import csv
from pathlib import Path

from ..config import training_data_path
from ..schemas import CaseRecord
from .validation import validate_cases


REQUIRED_COLUMNS = {
    "case_id",
    "modality",
    "body_part",
    "study_description",
    "patient_age_band",
    "patient_sex",
    "template_content",
    "dictation",
}


def load_cases(
    path: str | Path | None = None, *, require_report: bool = True
) -> list[CaseRecord]:
    """Read and validate cases from a CSV file.

    By default, load ``CLINICAL_DATA_ROOT/train.csv`` (falling back to the
    ignored repository ``data/private/train.csv``). Set ``require_report`` to
    false for unlabeled inference CSVs.
    """
    source = Path(path) if path is not None else training_data_path()
    if not source.is_file():
        raise FileNotFoundError(f"Clinical data CSV not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = REQUIRED_COLUMNS | ({"report"} if require_report else set())
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows = [
            {key: (value or "") for key, value in row.items() if key is not None and key in REQUIRED_COLUMNS | {"report"}}
            for row in reader
        ]
    return validate_cases(rows, require_report=require_report)
