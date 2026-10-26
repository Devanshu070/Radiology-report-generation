"""Validation logic for raw clinical CSV dictionaries and case records."""

from __future__ import annotations

from typing import Any

from ..schemas import ClinicalCase, TokenizedSFTRecord


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


def validate_tokenized_sft_record(record: TokenizedSFTRecord | dict[str, Any]) -> TokenizedSFTRecord:
    """Validate aligned token arrays and masked-loss boundaries for an SFT record.

    Ensures:
      - ``input_ids``, ``attention_mask``, ``completion_mask``, and ``labels`` have identical length.
      - Binary masks contain only 0 and 1.
      - Supervised positions (completion_mask == 1) have attention_mask == 1 and labels == input_ids.
      - Prompt and padding positions (completion_mask == 0 or attention_mask == 0) have labels == -100.

    Raises:
        ValueError: If any array length or loss boundary invariant is violated.
    """
    rec = record if isinstance(record, TokenizedSFTRecord) else TokenizedSFTRecord.model_validate(record)

    n_tokens = len(rec.input_ids)
    if n_tokens == 0:
        raise ValueError(f"Tokenized record for case '{rec.case_id}' contains empty input_ids.")

    if len(rec.attention_mask) != n_tokens:
        raise ValueError(
            f"Case '{rec.case_id}' length mismatch: input_ids={n_tokens} vs attention_mask={len(rec.attention_mask)}"
        )
    if len(rec.completion_mask) != n_tokens:
        raise ValueError(
            f"Case '{rec.case_id}' length mismatch: input_ids={n_tokens} vs completion_mask={len(rec.completion_mask)}"
        )
    if len(rec.labels) != n_tokens:
        raise ValueError(
            f"Case '{rec.case_id}' length mismatch: input_ids={n_tokens} vs labels={len(rec.labels)}"
        )

    for i in range(n_tokens):
        att = rec.attention_mask[i]
        comp = rec.completion_mask[i]
        label = rec.labels[i]
        tok = rec.input_ids[i]

        if att not in (0, 1):
            raise ValueError(f"Case '{rec.case_id}' invalid attention_mask value at index {i}: {att}")
        if comp not in (0, 1):
            raise ValueError(f"Case '{rec.case_id}' invalid completion_mask value at index {i}: {comp}")

        # Padding tokens must have attention_mask=0, completion_mask=0, labels=-100
        if att == 0:
            if comp != 0:
                raise ValueError(
                    f"Case '{rec.case_id}' index {i}: padding token (attention_mask=0) has completion_mask=1"
                )
            if label != -100:
                raise ValueError(
                    f"Case '{rec.case_id}' index {i}: padding token (attention_mask=0) has label={label} (expected -100)"
                )
            continue

        # Active tokens (att == 1)
        if comp == 1:
            if label != tok:
                raise ValueError(
                    f"Case '{rec.case_id}' index {i}: completion token has label={label} but input_id={tok}"
                )
        else:
            if label != -100:
                raise ValueError(
                    f"Case '{rec.case_id}' index {i}: unmasked prompt token has label={label} (expected -100)"
                )

    return rec
