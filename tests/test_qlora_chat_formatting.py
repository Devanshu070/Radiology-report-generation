"""Public offline unit tests for Approach 2 Qwen ChatML formatting and budgeting."""

from __future__ import annotations

import json
import unittest

from hybrid_retrieval_qlora.preprocess.chat_formatting import (
    build_chat_messages,
    extract_template_sections,
    format_case_input_payload,
    format_sft_example,
)
from hybrid_retrieval_qlora.schemas import ClinicalCase


def make_case(
    case_id: str,
    *,
    modality: str = "CT",
    body_part: str = "Chest",
    template: str = "FINDINGS:\nLungs are clear.\n\nIMPRESSION:\nNormal examination.",
    dictation: str = "Small right pleural effusion.",
    report: str | None = "FINDINGS:\nSmall right pleural effusion.\n\nIMPRESSION:\n1. Small right pleural effusion.",
) -> ClinicalCase:
    return ClinicalCase(
        case_id=case_id,
        modality=modality,
        body_part=body_part,
        study_description="Contrast scan",
        patient_age_band="60-70",
        patient_sex="M",
        template_content=template,
        dictation=dictation,
        report=report,
    )


class MockLengthTokenizer:
    """Mock tokenizer returning token counts based on text length for offline testing."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str:
        # Simulate ~4 chars per token plus header tokens
        total_chars = sum(len(m.get("content", "")) + len(m.get("role", "")) for m in messages)
        overhead = len(messages) * 4 + (4 if add_generation_prompt else 0)
        token_count = max(1, total_chars // 4 + overhead)
        if tokenize:
            return list(range(token_count))
        return "mock_template_string"


class TemplateExtractionTests(unittest.TestCase):
    """Test parsing and section separation from baseline templates."""

    def test_extracts_findings_and_impression(self):
        template = "FINDINGS:\nNo fracture or dislocation.\n\nIMPRESSION:\nUnremarkable study."
        findings, impression = extract_template_sections(template)
        self.assertEqual(findings, "No fracture or dislocation.")
        self.assertEqual(impression, "Unremarkable study.")

    def test_fallback_when_headers_absent(self):
        template = "Unstructured template without clear headings."
        findings, impression = extract_template_sections(template)
        self.assertEqual(findings, template)
        self.assertEqual(impression, "")


class CasePayloadFormattingTests(unittest.TestCase):
    """Test canonical input payload dictionary construction."""

    def test_payload_contains_all_input_columns(self):
        case = make_case("case-01")
        payload = format_case_input_payload(case)

        self.assertEqual(payload["modality"], "CT")
        self.assertEqual(payload["body_part"], "Chest")
        self.assertEqual(payload["dictation"], "Small right pleural effusion.")
        self.assertEqual(payload["template_findings"], "Lungs are clear.")
        self.assertEqual(payload["template_impression"], "Normal examination.")


class ChatMessageAssemblyTests(unittest.TestCase):
    """Test assembling multi-turn chat message sequences."""

    def test_supervised_assembly_with_demonstrations(self):
        demo1 = make_case("demo-01", report="FINDINGS: Demo 1 finding.\n\nIMPRESSION: Demo 1 impression.")
        demo2 = make_case("demo-02", report="FINDINGS: Demo 2 finding.\n\nIMPRESSION: Demo 2 impression.")
        target = make_case("target-01")

        prompt_turns, completion_turn = build_chat_messages(
            target,
            demonstrations=[demo1, demo2],
            supervised=True,
        )

        # 1 system + (2 * 2 demos) + 1 current case user turn = 6 prompt turns
        self.assertEqual(len(prompt_turns), 6)
        self.assertEqual(prompt_turns[0].role, "system")
        self.assertEqual(prompt_turns[1].role, "user")
        self.assertEqual(prompt_turns[2].role, "assistant")
        self.assertEqual(prompt_turns[3].role, "user")
        self.assertEqual(prompt_turns[4].role, "assistant")
        self.assertEqual(prompt_turns[5].role, "user")

        self.assertIsNotNone(completion_turn)
        self.assertEqual(completion_turn.role, "assistant")
        self.assertEqual(completion_turn.content, target.report)

    def test_unsupervised_inference_assembly(self):
        target = make_case("target-eval", report=None)
        prompt_turns, completion_turn = build_chat_messages(target, supervised=False)

        self.assertEqual(len(prompt_turns), 2)  # system + user
        self.assertEqual(prompt_turns[0].role, "system")
        self.assertEqual(prompt_turns[1].role, "user")
        self.assertIsNone(completion_turn)


class DemonstrationBudgetingTests(unittest.TestCase):
    """Test pruning whole demonstrations to fit maximum sequence budgets."""

    def test_drops_demonstrations_from_end_to_fit_budget(self):
        demos = [make_case(f"demo-{i:02d}") for i in range(5)]
        target = make_case("target-case")

        tokenizer = MockLengthTokenizer()
        # Set a sequence budget that fits target + 2 demonstrations
        formatted = format_sft_example(
            target,
            candidate_demonstrations=demos,
            tokenizer=tokenizer,
            max_input_tokens=600,
            max_seq_length=650,
        )

        self.assertLess(len(formatted.demonstration_ids), 5)
        # Verify demonstrations were retained in priority order (demo-00, demo-01, ...)
        self.assertEqual(formatted.demonstration_ids, ["demo-00", "demo-01"])

    def test_raises_when_case_alone_exceeds_budget(self):
        target = make_case("target-oversized", dictation="Very long dictation " * 50)
        tokenizer = MockLengthTokenizer()

        with self.assertRaises(ValueError) as ctx:
            format_sft_example(
                target,
                candidate_demonstrations=[],
                tokenizer=tokenizer,
                max_seq_length=20,  # Far too small
            )
        self.assertIn("exceeds maximum token limit", str(ctx.exception))
        self.assertIn("No clinical text was truncated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
