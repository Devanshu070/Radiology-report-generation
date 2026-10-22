"""Extract top-level template sections and build prompt-ready case records."""

from __future__ import annotations

import re

from ..schemas import CaseRecord, PromptCase, TemplateSections


_SECTION_HEADING = re.compile(r"^[ \t]*(FINDINGS|IMPRESSION)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)


def extract_template_sections(template: str) -> TemplateSections:
    """Extract exactly one FINDINGS section followed by one IMPRESSION section."""
    matches = list(_SECTION_HEADING.finditer(template))
    names = [match.group(1).upper() for match in matches]
    if names != ["FINDINGS", "IMPRESSION"]:
        raise ValueError("Template must contain exactly one FINDINGS heading followed by one IMPRESSION heading.")
    if template[:matches[0].start()].strip():
        raise ValueError("Template content before FINDINGS is not allowed.")

    findings = template[matches[0].end():matches[1].start()].strip()
    impression = template[matches[1].end():].strip()
    if not findings:
        raise ValueError("Template FINDINGS section must not be empty.")
    if not impression:
        raise ValueError("Template IMPRESSION section must not be empty.")
    return TemplateSections(findings=findings, impression=impression)


def to_prompt_case(case: CaseRecord) -> PromptCase:
    """Convert a validated case into prompt variables without altering text."""
    sections = extract_template_sections(case.template_content)
    return PromptCase(
        case_id=case.case_id,
        modality=case.modality,
        body_part=case.body_part,
        study_description=case.study_description,
        patient_age_band=case.patient_age_band,
        patient_sex=case.patient_sex,
        template_findings=sections.findings,
        template_impression=sections.impression,
        dictation=case.dictation,
    )
