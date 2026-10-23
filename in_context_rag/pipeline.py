"""Composable retrieval, generation, report assembly, and RES evaluation."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

from evaluation.res import res_proxy

from .generation.groq_client import GroqGenerator, ProviderError
from .generation.request_builder import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL, build_generation_request
from .generation.response_cache import ResponseCache, request_fingerprint
from .preprocess.template_sections import to_prompt_case
from .retrieval.embeddings import DEFAULT_EMBEDDING_MODEL
from .retrieval.few_shot_selection import select_few_shot_examples
from .schemas import CaseRecord, FewShotPair, GeneratedSections, PromptCase, SplitManifest


_FIELD_HEADING = re.compile(r"^([A-Za-z][A-Za-z0-9 /&()\-]{0,60}):[ \t]*", re.MULTILINE)


@dataclass(frozen=True)
class GenerationResult:
    """Safe local generation output; reference text is added only at evaluation."""

    case_id: str
    generated_report: str
    example_ids: list[str]
    status: str
    error: str = ""
    cached: bool = False
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieve_examples(
    cases: list[CaseRecord],
    manifest: SplitManifest,
    *,
    query_split: str = "test",
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path | None = None,
    embedding_matrices: tuple[Any, Any] | None = None,
) -> dict[str, list[str]]:
    """Return demonstration IDs for one held-out partition; candidates are train-only."""
    if query_split not in {"validation", "test"}:
        raise ValueError("Queries must come from validation or test, never the training partition.")
    split_ids = manifest.validation_ids if query_split == "validation" else manifest.test_ids
    by_id = {case.case_id: case for case in cases}
    if set(by_id) != set(manifest.assignments):
        raise ValueError("Split manifest case IDs must exactly match supplied cases.")
    query_cases = [case for case in cases if case.case_id in split_ids]
    if not query_cases:
        raise ValueError(f"The {query_split} partition is empty.")
    selected = select_few_shot_examples(
        cases,
        query_cases,
        manifest,
        top_k=top_k,
        model_name=model_name,
        cache_dir=cache_dir,
        embedding_matrices=embedding_matrices,
    )
    return {case_id: [example.case_id for example in examples] for case_id, examples in selected.items()}


def _field_labels(findings: str) -> list[str]:
    return [match.group(1) for match in _FIELD_HEADING.finditer(findings)]


def assemble_report(template: PromptCase, sections: GeneratedSections) -> str:
    """Assemble model bodies while enforcing the template's exact field sequence."""
    original_labels = _field_labels(template.template_findings)
    generated_labels = _field_labels(sections.findings)
    if original_labels != generated_labels:
        raise ValueError("Generated FINDINGS field labels or order do not match the supplied template.")
    return f"FINDINGS:\n{sections.findings}\n\nIMPRESSION:\n{sections.impression}"


async def generate_reports(
    cases: list[CaseRecord],
    manifest: SplitManifest,
    selections: dict[str, list[str]],
    generator: GroqGenerator,
    *,
    query_split: str = "test",
    model: str = DEFAULT_MODEL,
    cache: ResponseCache | None = None,
    use_cache: bool = True,
    concurrency: int = 2,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> list[GenerationResult]:
    """Generate reports for selected held-out cases using only train examples."""
    if query_split not in {"validation", "test"}:
        raise ValueError("Generation queries must come from validation or test.")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1.")
    by_id = {case.case_id: case for case in cases}
    if len(by_id) != len(cases) or set(by_id) != set(manifest.assignments):
        raise ValueError("Cases must have unique IDs and exactly match the split manifest.")
    queries = [
        case for case in cases
        if manifest.assignments[case.case_id] == query_split
    ]
    if not queries:
        raise ValueError(f"The {query_split} partition is empty.")
    if set(selections) != {case.case_id for case in queries}:
        raise ValueError("Selections must contain exactly one entry for every query case.")

    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(case: CaseRecord) -> GenerationResult:
        example_ids = selections[case.case_id]
        if not isinstance(example_ids, list) or any(not isinstance(example_id, str) for example_id in example_ids):
            return GenerationResult(case.case_id, "", [], "failed", "Demonstration IDs must be a list of strings.", model=model)
        if len(example_ids) > 5 or len(set(example_ids)) != len(example_ids):
            return GenerationResult(case.case_id, "", example_ids, "failed", "Invalid demonstration list.", model=model)
        if any(example_id not in manifest.train_ids for example_id in example_ids):
            return GenerationResult(case.case_id, "", example_ids, "failed", "A demonstration is outside the training split.", model=model)
        if case.case_id in example_ids:
            return GenerationResult(case.case_id, "", example_ids, "failed", "A case cannot demonstrate itself.", model=model)
        try:
            examples = [by_id[example_id] for example_id in example_ids]
            if any(not example.report for example in examples):
                raise ValueError("A demonstration lacks a reference report.")
            prompt_case = to_prompt_case(case)
            pairs = [FewShotPair(
                case_id=example.case_id,
                modality=example.modality,
                body_part=example.body_part,
                study_description=example.study_description,
                patient_age_band=example.patient_age_band,
                patient_sex=example.patient_sex,
                dictation=example.dictation,
                template_content=example.template_content,
                report=example.report or "",
            ) for example in examples]
            payload = build_generation_request(
                prompt_case,
                pairs,
                model=model,
                max_completion_tokens=max_completion_tokens,
            )
            async with semaphore:
                while True:
                    fingerprint = request_fingerprint(payload)
                    cached_sections = cache.get(fingerprint) if cache is not None and use_cache else None
                    try:
                        sections = cached_sections or await generator.generate(payload)
                        report = assemble_report(prompt_case, sections)
                        if cache is not None and use_cache and cached_sections is None:
                            cache.put(fingerprint, sections)
                        return GenerationResult(
                            case_id=case.case_id,
                            generated_report=report,
                            example_ids=example_ids,
                            status="success",
                            cached=cached_sections is not None,
                            model=model,
                        )
                    except ProviderError as error:
                        if error.status_code == 413 and len(pairs) > 3:
                            pairs.pop()
                            example_ids = example_ids[:-1]
                            payload = build_generation_request(
                                prompt_case,
                                pairs,
                                model=model,
                                max_completion_tokens=max_completion_tokens,
                            )
                            continue
                        raise
        except Exception as error:
            if isinstance(error, (ProviderError, ValueError)):
                safe_error = str(error)[:500]
            else:
                safe_error = f"{type(error).__name__} (details omitted)"
            return GenerationResult(
                case_id=case.case_id,
                generated_report="",
                example_ids=example_ids,
                status="failed",
                error=safe_error,
                model=model,
            )

    return list(await asyncio.gather(*(generate_one(case) for case in queries)))


def evaluate_results(
    cases: list[CaseRecord], results: list[GenerationResult | dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add local RES proxy scores after generation, never during retrieval/prompting."""
    by_id = {case.case_id: case for case in cases}
    output: list[dict[str, Any]] = []
    for result in results:
        row = result.as_dict() if isinstance(result, GenerationResult) else dict(result)
        case_id = str(row.get("case_id", ""))
        if case_id not in by_id:
            raise ValueError(f"Generation result references unknown case_id: {case_id}")
        case = by_id[case_id]
        score = res_proxy(case.template_content, case.report or "", str(row.get("generated_report", "")))
        row.update(score)
        output.append(row)
    return output


async def run_pipeline(
    cases: list[CaseRecord],
    manifest: SplitManifest,
    generator: GroqGenerator,
    *,
    query_split: str = "test",
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
    embedding_model: str | None = None,
    embedding_cache_dir: str | Path | None = None,
    response_cache: ResponseCache | None = None,
    use_cache: bool = True,
    concurrency: int = 2,
    embedding_matrices: tuple[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval, generation, report assembly, and local evaluation."""
    selections = retrieve_examples(
        cases,
        manifest,
        query_split=query_split,
        top_k=top_k,
        model_name=embedding_model or DEFAULT_EMBEDDING_MODEL,
        cache_dir=embedding_cache_dir,
        embedding_matrices=embedding_matrices,
    )
    generated = await generate_reports(
        cases,
        manifest,
        selections,
        generator,
        query_split=query_split,
        model=model,
        cache=response_cache,
        use_cache=use_cache,
        concurrency=concurrency,
    )
    return evaluate_results(cases, generated)
