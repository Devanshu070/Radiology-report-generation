"""Report generation using fine-tuned QLoRA adapters or base models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import re
from typing import Any

from ..config import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_SFT_SYSTEM_PROMPT,
    DEFAULT_TOP_K,
)
from ..preprocess.chat_formatting import format_sft_example
from ..retrieval.cache_manifest import EmbeddingCacheManager
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.pubmedbert_embeddings import PubMedBERTEmbedder
from ..schemas import ClinicalCase, GenerationResult

_FINDINGS_HEADER = re.compile(r"^\s*FINDINGS\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_IMPRESSION_HEADER = re.compile(r"^\s*IMPRESSION\s*:\s*$", re.IGNORECASE | re.MULTILINE)


def _validate_report_structure(report: str) -> None:
    """Require non-empty FINDINGS and IMPRESSION sections in the expected order."""
    findings = _FINDINGS_HEADER.search(report)
    impression = _IMPRESSION_HEADER.search(report)
    if findings is None or impression is None or findings.start() >= impression.start():
        raise ValueError("Generated report must contain FINDINGS followed by IMPRESSION sections.")
    if not report[findings.end() : impression.start()].strip():
        raise ValueError("Generated report has an empty FINDINGS section.")
    if not report[impression.end() :].strip():
        raise ValueError("Generated report has an empty IMPRESSION section.")


def generate_reports(
    model: Any,
    tokenizer: Any,
    cases: Sequence[ClinicalCase],
    *,
    variant: str = "finetuned_retrieval",
    retriever: HybridRetriever | None = None,
    cache_manager: EmbeddingCacheManager | None = None,
    embedder: PubMedBERTEmbedder | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    system_prompt: str = DEFAULT_SFT_SYSTEM_PROMPT,
    output_path: Path | str | None = None,
    generate_fn: Callable[[str], str] | None = None,
    device: str | None = None,
) -> list[GenerationResult]:
    """Generate completed radiology reports for input cases using greedy decoding.

    Args:
        model: Loaded Hugging Face causal LM or mock model.
        tokenizer: Hugging Face or mock tokenizer.
        cases: Target cases requiring completion.
        variant: Label distinguishing generation runs (e.g. 'finetuned_retrieval', 'baseline').
        retriever: Optional hybrid retriever to supply demonstration cases.
        cache_manager: Optional cache manager for retrieval embeddings.
        embedder: Optional PubMedBERT embedding model.
        top_k: Number of demonstrations to retrieve if retriever is provided.
        max_input_tokens: Token budget for the prompt turn.
        max_new_tokens: Maximum tokens to generate.
        max_seq_length: Total context limit.
        system_prompt: System prompt instructions.
        output_path: Optional CSV output path to persist results.
        generate_fn: Optional injected generation callback for testing without GPU/weights.
        device: Torch device (e.g. 'cuda:0', 'cpu'). Auto-detected if None.

    Returns:
        List of GenerationResult records.
    """
    results: list[GenerationResult] = []

    # Map candidate cases by ID if retriever is available
    candidate_pool = getattr(retriever, "candidate_pool", []) if retriever else []
    training_cases_by_id = {c.case_id: c for c in candidate_pool}

    # Set model to eval mode if applicable
    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    for case in cases:
        candidate_demos: list[ClinicalCase] = []
        if retriever is not None and top_k > 0:
            retrieval_results = retriever.retrieve(
                [case],
                cache_manager=cache_manager,
                embedder=embedder,
                top_k=top_k,
            )
            if retrieval_results and retrieval_results[0].candidates:
                for cand in retrieval_results[0].candidates:
                    demo_case = training_cases_by_id.get(cand.candidate_case_id)
                    if demo_case and demo_case.case_id != case.case_id:
                        candidate_demos.append(demo_case)

        formatted = format_sft_example(
            case,
            candidate_demos,
            tokenizer=tokenizer,
            max_input_tokens=max_input_tokens,
            max_seq_length=max_seq_length,
            system_prompt=system_prompt,
            supervised=False,
        )

        prompt_dicts = [m.model_dump() for m in formatted.prompt_messages]

        try:
            if generate_fn is not None:
                # Injected generation callback (e.g., in unit tests)
                raw_prompt = (
                    tokenizer.apply_chat_template(
                        prompt_dicts,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if hasattr(tokenizer, "apply_chat_template")
                    else str(prompt_dicts)
                )
                generated_text = generate_fn(raw_prompt)
                token_count = len(generated_text.split())
            else:
                import torch

                active_device = (
                    device
                    if device is not None
                    else ("cuda:0" if torch.cuda.is_available() else "cpu")
                )

                raw_prompt = tokenizer.apply_chat_template(
                    prompt_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(raw_prompt, add_special_tokens=False, return_tensors="pt")
                inputs = {k: v.to(active_device) for k, v in inputs.items()}
                input_length = inputs["input_ids"].shape[1]

                eos_ids = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
                eos_id_list = []
                if eos_ids is not None:
                    eos_id_list.extend(eos_ids if isinstance(eos_ids, list) else [eos_ids])
                if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                    eos_id_list.append(tokenizer.eos_token_id)
                eos_id_list = list(set(eos_id_list)) or None

                pad_token_id = getattr(tokenizer, "pad_token_id", None)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=eos_id_list,
                        pad_token_id=pad_token_id,
                    )

                continuation_ids = outputs[0][input_length:]
                token_count = len(continuation_ids)
                generated_text = tokenizer.decode(
                    continuation_ids, skip_special_tokens=True
                ).strip()

            if not generated_text:
                raise ValueError("Model generated empty text.")
            _validate_report_structure(generated_text)

            results.append(
                GenerationResult(
                    case_id=case.case_id,
                    variant=variant,
                    generated_report=generated_text,
                    status="success",
                    error="",
                    generated_tokens=token_count,
                )
            )
        except Exception as exc:
            results.append(
                GenerationResult(
                    case_id=case.case_id,
                    variant=variant,
                    generated_report="",
                    status="failed",
                    error=str(exc),
                    generated_tokens=0,
                )
            )

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pandas as pd

            df = pd.DataFrame([r.model_dump() for r in results])
            df.to_csv(out_file, index=False)
        except ImportError:
            import csv

            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "case_id",
                        "variant",
                        "generated_report",
                        "status",
                        "error",
                        "generated_tokens",
                    ],
                )
                writer.writeheader()
                for r in results:
                    writer.writerow(r.model_dump())

    return results
