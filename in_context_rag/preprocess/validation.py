"""Input and split validation independent of model/provider code."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import ValidationError

from ..schemas import CaseRecord


def validate_cases(
    rows: Iterable[CaseRecord | Mapping[str, object]], *, require_report: bool = True
) -> list[CaseRecord]:
    """Validate case records and reject duplicate identifiers or unlabeled rows."""
    cases: list[CaseRecord] = []
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        try:
            case = row if isinstance(row, CaseRecord) else CaseRecord.model_validate(row)
        except ValidationError as error:
            raise ValueError(f"Invalid case record at row {row_number}: {error}") from error
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case_id: {case.case_id}")
        if require_report and not case.report:
            raise ValueError(f"Training case {case.case_id} has no reference report.")
        seen_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("No case records were supplied.")
    return cases
