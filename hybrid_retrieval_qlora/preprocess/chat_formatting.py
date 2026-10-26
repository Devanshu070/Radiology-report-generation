"""Hugging Face ChatML template formatting and demonstration budgeting for Approach 2 SFT."""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
from typing import Any

from ..config import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_SFT_SYSTEM_PROMPT,
    SFT_INPUT_COLUMNS,
)
from ..schemas import ChatMessage, ClinicalCase, FormattedSFTExample

TEMPLATE_SECTIONS_PATTERN = re.compile(
    r"(?ims)^\s*FINDINGS\s*:\s*(?P<template_findings>.*?)"
    r"^\s*IMPRESSION\s*:\s*(?P<template_impression>.*)$"
)


def extract_template_sections(template_content: str) -> tuple[str, str]:
    """Extract baseline findings and impression from structured template content.

    Args:
        template_content: Full text of the baseline radiology template.

    Returns:
        Tuple of (template_findings, template_impression).
    """
    match = TEMPLATE_SECTIONS_PATTERN.search(template_content)
    if match:
        findings = match.group("template_findings").strip()
        impression = match.group("template_impression").strip()
        return findings, impression

    # Fallback if standard headers are not cleanly separated
    return template_content.strip(), ""


def format_case_input_payload(case: ClinicalCase) -> dict[str, str]:
    """Construct canonical JSON dictionary representation for a clinical case input.

    Matches the Approach 2 data contract specifying clinical context, dictation,
    and separated template findings and impression baselines.
    """
    findings, impression = extract_template_sections(case.template_content)
    raw_dict = {
        "modality": case.modality,
        "body_part": case.body_part,
        "study_description": case.study_description,
        "patient_age_band": case.patient_age_band,
        "patient_sex": case.patient_sex,
        "dictation": case.dictation,
        "template_findings": findings,
        "template_impression": impression,
    }
    return {col: str(raw_dict.get(col, "")).strip() for col in SFT_INPUT_COLUMNS}


def build_chat_messages(
    case: ClinicalCase,
    demonstrations: Sequence[ClinicalCase] | None = None,
    *,
    system_prompt: str = DEFAULT_SFT_SYSTEM_PROMPT,
    supervised: bool = True,
) -> tuple[list[ChatMessage], ChatMessage | None]:
    """Assemble conversation turns for a clinical case with demonstration examples.

    Args:
        case: Current target clinical case.
        demonstrations: Sequence of retrieved labeled demonstration cases.
        system_prompt: System instruction block for the chat model.
        supervised: Whether to include the current case's reference report as target completion.

    Returns:
        Tuple of (prompt_messages, completion_message).
    """
    prompt_turns: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt.strip())
    ]

    for demo in (demonstrations or []):
        if not demo.report or not demo.report.strip():
            raise ValueError(f"Demonstration case '{demo.case_id}' lacks a labeled reference report.")
        demo_payload = json.dumps(format_case_input_payload(demo), ensure_ascii=False)
        prompt_turns.append(ChatMessage(role="user", content=demo_payload))
        prompt_turns.append(ChatMessage(role="assistant", content=demo.report.strip()))

    case_payload = json.dumps(format_case_input_payload(case), ensure_ascii=False)
    prompt_turns.append(ChatMessage(role="user", content=case_payload))

    completion_turn: ChatMessage | None = None
    if supervised:
        if not case.report or not case.report.strip():
            raise ValueError(f"Supervised case '{case.case_id}' lacks reference report.")
        completion_turn = ChatMessage(role="assistant", content=case.report.strip())

    return prompt_turns, completion_turn


def format_sft_example(
    case: ClinicalCase,
    candidate_demonstrations: Sequence[ClinicalCase] | None = None,
    *,
    tokenizer: Any | None = None,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    system_prompt: str = DEFAULT_SFT_SYSTEM_PROMPT,
    supervised: bool = True,
) -> FormattedSFTExample:
    """Format an SFT chat example, budgeting whole demonstrations to satisfy token limits.

    Prunes demonstrations from the end if token limits are exceeded, ensuring
    no clinical text or report sections are arbitrarily truncated.

    Args:
        case: The primary clinical case.
        candidate_demonstrations: Retrieved demonstration candidates in descending score order.
        tokenizer: Model tokenizer implementing ``apply_chat_template`` or token counting.
        max_input_tokens: Maximum token length permitted for the prompt turns.
        max_seq_length: Maximum sequence length for the combined prompt + completion.
        system_prompt: Task-specific system instructions.
        supervised: Whether the example includes target completion supervision.

    Returns:
        FormattedSFTExample ready for tokenization and loss masking.

    Raises:
        ValueError: If the current case alone exceeds token limits without demonstrations.
    """
    active_demos = list(candidate_demonstrations or [])

    while True:
        prompt_messages, completion_message = build_chat_messages(
            case,
            active_demos,
            system_prompt=system_prompt,
            supervised=supervised,
        )

        if tokenizer is None:
            # Without tokenizer, accept formatted messages directly
            return FormattedSFTExample(
                case_id=case.case_id,
                prompt_messages=prompt_messages,
                completion_message=completion_message,
                demonstration_ids=[d.case_id for d in active_demos],
            )

        prompt_dicts = [m.model_dump() for m in prompt_messages]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt_dicts,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_length = len(prompt_tokens)

        if supervised and completion_message is not None:
            all_dicts = [*prompt_dicts, completion_message.model_dump()]
            full_tokens = tokenizer.apply_chat_template(
                all_dicts,
                tokenize=True,
                add_generation_prompt=False,
            )
            full_length = len(full_tokens)
        else:
            full_length = prompt_length

        if prompt_length <= max_input_tokens and (not supervised or full_length <= max_seq_length):
            return FormattedSFTExample(
                case_id=case.case_id,
                prompt_messages=prompt_messages,
                completion_message=completion_message,
                demonstration_ids=[d.case_id for d in active_demos],
            )

        if not active_demos:
            raise ValueError(
                f"Case '{case.case_id}' exceeds maximum token limit ({full_length} > {max_seq_length}) "
                "without demonstrations; increase token limits. No clinical text was truncated."
            )

        # Drop the lowest-ranked demonstration whole
        active_demos.pop()
