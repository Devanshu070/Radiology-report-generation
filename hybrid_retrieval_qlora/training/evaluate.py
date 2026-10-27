"""Clinically weighted RES evaluation of generated radiology reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.res import res_proxy

from ..schemas import ClinicalCase, EvaluationScore, EvaluationSummary, GenerationResult


def evaluate_reports(
    cases: Sequence[ClinicalCase],
    generation_results: Sequence[GenerationResult],
    *,
    variant: str = "finetuned_retrieval",
    output_path: Path | str | None = None,
) -> tuple[list[EvaluationScore], EvaluationSummary]:
    """Score generated reports against reference ground-truth using the canonical RES proxy.

    Findings unchanged from the template receive weight 1, while changed reference
    fields receive weight 3. Findings and impression are weighted at 0.65 and 0.35.
    Failed generations receive the maximum error score (1.0).

    Args:
        cases: Target clinical cases containing ``template_content`` and ``report``.
        generation_results: Generated report records matching target cases.
        variant: Label distinguishing generation runs.
        output_path: Optional CSV output path to persist detailed scores.

    Returns:
        Tuple of (detailed_evaluation_scores, summary_statistics).

    Raises:
        ValueError: If any target case lacks a reference ground-truth report.
    """
    case_map = {c.case_id: c for c in cases}
    scores: list[EvaluationScore] = []

    for gen in generation_results:
        case = case_map.get(gen.case_id)
        if case is None:
            continue

        if not case.report:
            raise ValueError(
                f"Case '{case.case_id}' lacks reference report required for RES evaluation."
            )

        if gen.status == "success" and gen.generated_report:
            score_dict = res_proxy(
                template_text=case.template_content,
                reference_text=case.report,
                prediction_text=gen.generated_report,
            )
            scores.append(
                EvaluationScore(
                    case_id=case.case_id,
                    variant=variant,
                    status=gen.status,
                    generated_report=gen.generated_report,
                    expected_report=case.report,
                    res_proxy=score_dict["res_proxy"],
                    findings_proxy=score_dict["findings_proxy"],
                    impression_proxy=score_dict["impression_proxy"],
                    error=gen.error,
                    scoring_error=score_dict.get("parse_error", ""),
                )
            )
        else:
            scores.append(
                EvaluationScore(
                    case_id=case.case_id,
                    variant=variant,
                    status=gen.status or "failed",
                    generated_report=gen.generated_report,
                    expected_report=case.report,
                    res_proxy=1.0,
                    findings_proxy=1.0,
                    impression_proxy=1.0,
                    error=gen.error or "Generation failed or produced empty report.",
                    scoring_error="",
                )
            )

    total = len(scores)
    successful = sum(1 for s in scores if s.status == "success")
    mean_res = sum(s.res_proxy for s in scores) / total if total > 0 else 0.0
    mean_findings = sum(s.findings_proxy for s in scores) / total if total > 0 else 0.0
    mean_impression = sum(s.impression_proxy for s in scores) / total if total > 0 else 0.0

    summary = EvaluationSummary(
        variant=variant,
        total_cases=total,
        successful_cases=successful,
        mean_res_proxy=round(mean_res, 6),
        mean_findings_proxy=round(mean_findings, 6),
        mean_impression_proxy=round(mean_impression, 6),
    )

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd

            df = pd.DataFrame([s.model_dump() for s in scores])
            df.to_csv(out_file, index=False)
        except ImportError:
            import csv

            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "case_id",
                        "variant",
                        "status",
                        "generated_report",
                        "expected_report",
                        "res_proxy",
                        "findings_proxy",
                        "impression_proxy",
                        "error",
                        "scoring_error",
                    ],
                )
                writer.writeheader()
                for s in scores:
                    writer.writerow(s.model_dump())

    return scores, summary
