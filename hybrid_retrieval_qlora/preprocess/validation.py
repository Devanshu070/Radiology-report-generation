"""Validation logic for raw clinical CSV dictionaries and case records."""

from __future__ import annotations

from typing import Any

from ..schemas import ClinicalCase


REQUIRED_CASE_FIELDS = {
    "case_id",
    "modality",
    "body_part",
    "template_content",
    "dictation",
}


def validate_case(record: dict[str, Any], *, require_report: bool = False) -> ClinicalCase:
    """Validate a single case dictionary against the clinical data contract.

    Raises:
        ValueError: If any required field is missing or contains empty whitespace.
    """
    for field in REQUIRED_CASE_FIELDS:
        val = record.get(field)
        if val is None or not str(val).strip():
            raise ValueError(f"Field '{field}' is required and cannot be empty.")

    if require_report:
        report_val = record.get("report")
        if report_val is None or not str(report_val).strip():
            raise ValueError("Field 'report' is required for training cases and cannot be empty.")

    return ClinicalCase(
        case_id=str(record["case_id"]).strip(),
        modality=str(record["modality"]).strip(),
        body_part=str(record["body_part"]).strip(),
        study_description=str(record.get("study_description", "")).strip(),
        patient_age_band=str(record.get("patient_age_band", "")).strip(),
        patient_sex=str(record.get("patient_sex", "")).strip(),
        template_content=str(record["template_content"]).strip(),
        dictation=str(record["dictation"]).strip(),
        report=str(record["report"]).strip() if record.get("report") is not None and str(record["report"]).strip() else None,
    )


def validate_cases(records: list[dict[str, Any]], *, require_report: bool = False) -> list[ClinicalCase]:
    """Validate an iterable of case records, ensuring identifier uniqueness.

    Raises:
        ValueError: If records contain duplicate case IDs or invalid fields.
    """
    validated: list[ClinicalCase] = []
    seen_ids: set[str] = set()

    for idx, record in enumerate(records):
        case = validate_case(record, require_report=require_report)
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case_id encountered at row {idx}: '{case.case_id}'")
        seen_ids.add(case.case_id)
        validated.append(case)

    return validated
