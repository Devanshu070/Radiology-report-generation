"""Command-line interface for local In-Context RAG preparation and evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from .generation.groq_client import GroqGenerator
from .generation.request_builder import DEFAULT_MAX_COMPLETION_TOKENS, DEFAULT_MODEL
from .generation.response_cache import ResponseCache
from .pipeline import evaluate_results, generate_reports, retrieve_examples
from .preprocess.load_data import load_cases
from .preprocess.split import create_split_manifest
from .review.html_reviewer import write_review_html
from .schemas import SplitManifest


DEFAULT_ARTIFACT_DIR = Path("artifacts") / "in_context_rag"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input artifact not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at line {line_number} of {path}.") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at line {line_number} of {path}.")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _load_manifest(cases: list[Any], path: Path) -> SplitManifest:
    if path.is_file():
        manifest = SplitManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if set(manifest.assignments) != {case.case_id for case in cases}:
            raise ValueError("Saved split manifest does not match the current clinical dataset.")
        return manifest
    return create_split_manifest(cases)


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ARTIFACT_DIR / "split_manifest.json")


def _load_cli_inputs(manifest_path: Path) -> tuple[list[Any], SplitManifest]:
    cases = load_cases()
    return cases, _load_manifest(cases, manifest_path)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", default=os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", default="NeuML/pubmedbert-base-embeddings")
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_ARTIFACT_DIR / "embeddings")
    parser.add_argument("--response-cache", type=Path, default=DEFAULT_ARTIFACT_DIR / "responses")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true", help="Bypass the successful-response cache.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, retrieve, generate, evaluate, and review local radiology reports."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Validate local data and save a deterministic split manifest.")
    _add_common_data_args(prepare)

    retrieve = subparsers.add_parser("retrieve", help="Select training-only few-shot example IDs.")
    _add_common_data_args(retrieve)
    retrieve.add_argument("--split", choices=("validation", "test"), default="test")
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--embedding-model", default="NeuML/pubmedbert-base-embeddings")
    retrieve.add_argument("--embedding-cache", type=Path, default=DEFAULT_ARTIFACT_DIR / "embeddings")
    retrieve.add_argument("--output", type=Path)

    generate = subparsers.add_parser("generate", help="Generate reports from a saved retrieval selection.")
    _add_common_data_args(generate)
    _add_generation_args(generate)
    generate.add_argument("--selections", type=Path)
    generate.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Score saved generated reports with the local RES proxy.")
    evaluate.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT_DIR / "generated.jsonl")
    evaluate.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "evaluated.jsonl")

    review = subparsers.add_parser("review", help="Write an escaped local HTML report-review page.")
    review.add_argument("--input", type=Path, default=DEFAULT_ARTIFACT_DIR / "evaluated.jsonl")
    review.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "review.html")

    run = subparsers.add_parser("run", help="Run retrieval, generation, and evaluation for one held-out split.")
    _add_common_data_args(run)
    _add_generation_args(run)
    run.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "evaluated.jsonl")
    return parser


async def _generate_command(args: argparse.Namespace) -> Path:
    cases, manifest = _load_cli_inputs(args.manifest)
    split_ids = manifest.validation_ids if args.split == "validation" else manifest.test_ids
    selections_path = args.selections or DEFAULT_ARTIFACT_DIR / f"selections-{args.split}.json"
    selections = _read_json(selections_path)
    if not isinstance(selections, dict) or any(
        not isinstance(case_id, str) or not isinstance(ids, list) for case_id, ids in selections.items()
    ):
        raise ValueError("Retrieval selections must be a JSON object mapping case IDs to example-ID lists.")
    if set(selections) != set(split_ids):
        raise ValueError("Retrieval selection IDs do not match the requested held-out split.")

    generator = GroqGenerator()
    try:
        results = await generate_reports(
            cases,
            manifest,
            selections,
            generator,
            query_split=args.split,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            cache=ResponseCache(args.response_cache),
            use_cache=not args.no_cache,
            concurrency=args.concurrency,
        )
    finally:
        await generator.aclose()
    output = args.output or DEFAULT_ARTIFACT_DIR / "generated.jsonl"
    _write_jsonl(output, [result.as_dict() for result in results])
    return output


async def _run_command(args: argparse.Namespace) -> Path:
    cases, manifest = _load_cli_inputs(args.manifest)
    generator = GroqGenerator()
    try:
        selections = retrieve_examples(
            cases,
            manifest,
            query_split=args.split,
            top_k=args.top_k,
            model_name=args.embedding_model,
            cache_dir=args.embedding_cache,
        )
        results = await generate_reports(
            cases,
            manifest,
            selections,
            generator,
            query_split=args.split,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            cache=ResponseCache(args.response_cache),
            use_cache=not args.no_cache,
            concurrency=args.concurrency,
        )
    finally:
        await generator.aclose()
    evaluated = evaluate_results(cases, results)
    _write_jsonl(args.output, evaluated)
    return args.output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            cases = load_cases()
            manifest = create_split_manifest(cases)
            _write_json(args.manifest, manifest.model_dump(mode="json"))
            print(
                f"Prepared {len(cases)} cases: {len(manifest.train_ids)} train, "
                f"{len(manifest.validation_ids)} validation, {len(manifest.test_ids)} test."
            )
        elif args.command == "retrieve":
            cases, manifest = _load_cli_inputs(args.manifest)
            selections = retrieve_examples(
                cases,
                manifest,
                query_split=args.split,
                top_k=args.top_k,
                model_name=args.embedding_model,
                cache_dir=args.embedding_cache,
            )
            output = args.output or DEFAULT_ARTIFACT_DIR / f"selections-{args.split}.json"
            _write_json(output, selections)
            print(f"Saved retrieval selections for {len(selections)} cases to {output}.")
        elif args.command == "generate":
            output = asyncio.run(_generate_command(args))
            print(f"Saved generated reports to {output}.")
        elif args.command == "evaluate":
            cases = load_cases()
            results = evaluate_results(cases, _read_jsonl(args.input))
            _write_jsonl(args.output, results)
            mean_score = sum(float(row["res_proxy"]) for row in results) / max(1, len(results))
            print(f"Evaluated {len(results)} cases | mean RES proxy {mean_score:.4f}; lower is better.")
        elif args.command == "review":
            cases = load_cases()
            output = write_review_html(_read_jsonl(args.input), cases, args.output)
            print(f"Saved local review page to {output}.")
        elif args.command == "run":
            output = asyncio.run(_run_command(args))
            print(f"Saved generated and evaluated reports to {output}.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
