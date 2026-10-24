"""Escape-safe HTML comparison pages for local clinician review."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any

from ..schemas import CaseRecord


def _display(value: Any) -> str:
    return escape("" if value is None else str(value))


def build_review_html(
    results: list[dict[str, Any]],
    cases: list[CaseRecord],
    *,
    title: str = "Radiology report review",
) -> str:
    """Render generated reports and their source context without raw HTML injection."""
    by_id = {case.case_id: case for case in cases}
    articles: list[str] = []
    for result in results:
        case_id = str(result.get("case_id", ""))
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"Review result references unknown case_id: {case_id}")
        score = result.get("res_proxy")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) and math.isfinite(float(score)) else "not scored"
        metadata = "".join(
            f"<div class='meta'><strong>{_display(label)}</strong><br>{_display(value)}</div>"
            for label, value in (
                ("Modality", case.modality),
                ("Body part", case.body_part),
                ("Study", case.study_description),
                ("Age band", case.patient_age_band),
                ("Sex", case.patient_sex),
                ("Status", result.get("status", "")),
            )
        )
        articles.append(f"""
        <article>
          <h2>Case {_display(case_id)} <span class="score">RES proxy: {_display(score_text)}</span></h2>
          <div class="metadata">{metadata}</div>
          <section><h3>Physician dictation</h3><pre>{_display(case.dictation)}</pre></section>
          <div class="reports">
            <section><h3>Input template</h3><pre>{_display(case.template_content)}</pre></section>
            <section><h3>Generated report</h3><pre>{_display(result.get('generated_report') or '(No report generated)')}</pre></section>
            <section><h3>Reference report</h3><pre>{_display(case.report or '(No reference available)')}</pre></section>
          </div>
          <p class="error">{_display(result.get('error', ''))}</p>
        </article>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_display(title)}</title>
<style>
body{{font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:24px;max-width:1400px;color:#152238;background:#f6f8fb}}
h1{{margin-top:0}}article{{background:#fff;border:1px solid #d4dce7;border-radius:10px;padding:20px;margin:22px 0}}
h2{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}}.score{{font-size:.85em;color:#40536d}}
.metadata{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:12px 0 18px}}
.meta,section{{border:1px solid #d4dce7;border-radius:6px;padding:12px;min-width:0}}.reports{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:0}}.error{{color:#a21b1b;overflow-wrap:anywhere}}
</style></head><body><h1>{_display(title)}</h1>
<p>Local review aid only. Model output requires qualified human review and is not a clinical decision.</p>
{''.join(articles)}</body></html>"""


def write_review_html(
    results: list[dict[str, Any]],
    cases: list[CaseRecord],
    output_path: str | Path,
    *,
    title: str = "Radiology report review",
) -> Path:
    """Write the review page to a local path, creating its parent directory."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_review_html(results, cases, title=title), encoding="utf-8")
    return path
