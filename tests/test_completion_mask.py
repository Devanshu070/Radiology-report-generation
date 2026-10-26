"""Public offline unit tests for Approach 2 target-only SFT completion masking."""

from __future__ import annotations

import unittest

from hybrid_retrieval_qlora.preprocess.chat_formatting import format_sft_example
from hybrid_retrieval_qlora.preprocess.completion_mask import (
    IM_END_TOKEN_ID,
    IM_START_TOKEN_ID,
    PAD_TOKEN_ID,
    MockQwenTokenizer,
    tokenize_sft_dataset,
    tokenize_sft_example,
)
from hybrid_retrieval_qlora.preprocess.validation import validate_tokenized_sft_record
from hybrid_retrieval_qlora.schemas import ClinicalCase, FormattedSFTExample, TokenizedSFTRecord


def make_case(
    case_id: str,
    *,
    dictation: str = "Small right pleural effusion.",
    report: str | None = "FINDINGS:\nSmall right pleural effusion.\n\nIMPRESSION:\n1. Small right pleural effusion.",
) -> ClinicalCase:
    return ClinicalCase(
        case_id=case_id,
        modality="CT",
        body_part="Chest",
        study_description="Routine scan",
        patient_age_band="60-70",
        patient_sex="F",
        template_content="FINDINGS:\nLungs are clear.\n\nIMPRESSION:\nNo acute cardiopulmonary finding.",
        dictation=dictation,
        report=report,
    )


class MockQwenTokenizerTests(unittest.TestCase):
    """Test ChatML format rendering and tokenization in MockQwenTokenizer."""

    def setUp(self):
        self.tokenizer = MockQwenTokenizer()

    def test_renders_qwen_chatml_structure(self):
        messages = [
            {"role": "system", "content": "You are a radiologist."},
            {"role": "user", "content": "Patient has cough."},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("<|im_start|>system\nYou are a radiologist.<|im_end|>\n", rendered)
        self.assertIn("<|im_start|>user\nPatient has cough.<|im_end|>\n", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_special_tokens_have_correct_ids(self):
        tokens = self.tokenizer.tokenize_text("<|im_start|>assistant\n<|im_end|>")
        self.assertEqual(tokens[0], IM_START_TOKEN_ID)
        self.assertEqual(tokens[-1], IM_END_TOKEN_ID)


class CompletionMaskingTests(unittest.TestCase):
    """Test target-only completion mask generation and label alignment."""

    def setUp(self):
        self.tokenizer = MockQwenTokenizer()
        self.case = make_case("case-mask-01")
        self.demo = make_case("demo-01", report="FINDINGS: Normal. IMPRESSION: Clear.")

    def test_target_only_loss_and_labels(self):
        formatted = format_sft_example(
            self.case,
            candidate_demonstrations=[self.demo],
            tokenizer=self.tokenizer,
            supervised=True,
        )

        record = tokenize_sft_example(formatted, self.tokenizer)

        # Equal length invariant
        n_tokens = len(record.input_ids)
        self.assertEqual(len(record.attention_mask), n_tokens)
        self.assertEqual(len(record.completion_mask), n_tokens)
        self.assertEqual(len(record.labels), n_tokens)

        # Invariant checks
        self.assertTrue(any(m == 1 for m in record.completion_mask))
        self.assertTrue(any(m == 0 for m in record.completion_mask))

        # Check prompt tokens: completion_mask == 0 and labels == -100
        prompt_indices = [i for i, m in enumerate(record.completion_mask) if m == 0]
        for idx in prompt_indices:
            self.assertEqual(record.labels[idx], -100)
            self.assertEqual(record.attention_mask[idx], 1)

        # Check completion tokens: completion_mask == 1 and labels == input_ids
        completion_indices = [i for i, m in enumerate(record.completion_mask) if m == 1]
        for idx in completion_indices:
            self.assertEqual(record.labels[idx], record.input_ids[idx])
            self.assertEqual(record.attention_mask[idx], 1)

        # Ensure assistant generation prompt header (<|im_start|>assistant\n) is masked with -100
        # The prompt ends with the assistant header, so its tokens must have completion_mask == 0
        prompt_boundary = prompt_indices[-1]
        self.assertEqual(record.completion_mask[prompt_boundary], 0)
        self.assertEqual(record.labels[prompt_boundary], -100)

        # The end-of-turn token for the assistant completion must be supervised
        self.assertEqual(record.completion_mask[-1], 1)
        self.assertEqual(record.labels[-1], record.input_ids[-1])

    def test_unsupervised_inference_example(self):
        eval_case = make_case("eval-01", report=None)
        formatted = format_sft_example(eval_case, tokenizer=self.tokenizer, supervised=False)
        record = tokenize_sft_example(formatted, self.tokenizer)

        # In unsupervised inference, completion_mask is all 0s, labels is all -100
        self.assertTrue(all(m == 0 for m in record.completion_mask))
        self.assertTrue(all(lab == -100 for lab in record.labels))
        self.assertTrue(all(att == 1 for att in record.attention_mask))

    def test_padding_behavior(self):
        formatted = format_sft_example(self.case, tokenizer=self.tokenizer, supervised=True)
        raw_record = tokenize_sft_example(formatted, self.tokenizer)
        raw_len = len(raw_record.input_ids)

        target_len = raw_len + 15
        padded_record = tokenize_sft_example(
            formatted,
            self.tokenizer,
            max_length=target_len,
            padding="max_length",
        )

        self.assertEqual(len(padded_record.input_ids), target_len)
        self.assertEqual(len(padded_record.attention_mask), target_len)
        self.assertEqual(len(padded_record.completion_mask), target_len)
        self.assertEqual(len(padded_record.labels), target_len)

        # Verify active tokens
        self.assertEqual(padded_record.attention_mask[:raw_len], [1] * raw_len)

        # Verify padding tokens (tail)
        self.assertEqual(padded_record.input_ids[raw_len:], [PAD_TOKEN_ID] * 15)
        self.assertEqual(padded_record.attention_mask[raw_len:], [0] * 15)
        self.assertEqual(padded_record.completion_mask[raw_len:], [0] * 15)
        self.assertEqual(padded_record.labels[raw_len:], [-100] * 15)

    def test_truncation_behavior(self):
        formatted = format_sft_example(self.case, tokenizer=self.tokenizer, supervised=True)
        raw_record = tokenize_sft_example(formatted, self.tokenizer)
        raw_len = len(raw_record.input_ids)

        truncated_len = max(5, raw_len - 10)
        truncated_record = tokenize_sft_example(
            formatted,
            self.tokenizer,
            max_length=truncated_len,
        )

        self.assertEqual(len(truncated_record.input_ids), truncated_len)
        self.assertEqual(len(truncated_record.attention_mask), truncated_len)
        self.assertEqual(len(truncated_record.completion_mask), truncated_len)
        self.assertEqual(len(truncated_record.labels), truncated_len)

    def test_batch_tokenization(self):
        c1 = make_case("batch-01")
        c2 = make_case("batch-02")
        f1 = format_sft_example(c1, tokenizer=self.tokenizer, supervised=True)
        f2 = format_sft_example(c2, tokenizer=self.tokenizer, supervised=True)

        batch = tokenize_sft_dataset([f1, f2], self.tokenizer)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].case_id, "batch-01")
        self.assertEqual(batch[1].case_id, "batch-02")

    def test_longest_batch_padding_aligns_records(self):
        short = format_sft_example(make_case("short"), tokenizer=self.tokenizer, supervised=True)
        long_case = make_case(
            "long",
            dictation="Large right pleural effusion " + "with adjacent compressive atelectasis " * 20,
            report="FINDINGS: Large effusion and atelectasis. IMPRESSION: Abnormal.",
        )
        long = format_sft_example(long_case, tokenizer=self.tokenizer, supervised=True)
        records = tokenize_sft_dataset([short, long], self.tokenizer, padding="longest")

        self.assertEqual(len(records[0].input_ids), len(records[1].input_ids))
        self.assertIn(0, records[0].attention_mask)
        self.assertEqual(records[0].completion_mask[-1], 0)
        self.assertEqual(records[0].labels[-1], -100)
        self.assertTrue(any(records[0].completion_mask))

    def test_truncation_cannot_remove_entire_supervised_completion(self):
        formatted = format_sft_example(self.case, tokenizer=self.tokenizer, supervised=True)
        prompt_length = len(
            self.tokenizer.apply_chat_template(
                [message.model_dump() for message in formatted.prompt_messages],
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        with self.assertRaisesRegex(ValueError, "removed every supervised completion token"):
            tokenize_sft_example(formatted, self.tokenizer, max_length=prompt_length)


class TokenizedRecordValidationTests(unittest.TestCase):
    """Test strict validation of TokenizedSFTRecord invariants."""

    def test_detects_length_mismatch(self):
        bad_record = TokenizedSFTRecord(
            case_id="err-01",
            input_ids=[1, 2, 3],
            attention_mask=[1, 1],  # Mismatch
            completion_mask=[0, 1, 1],
            labels=[-100, 2, 3],
        )
        with self.assertRaises(ValueError) as ctx:
            validate_tokenized_sft_record(bad_record)
        self.assertIn("length mismatch", str(ctx.exception))

    def test_detects_invalid_mask_value(self):
        bad_record = TokenizedSFTRecord(
            case_id="err-02",
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            completion_mask=[0, 2, 1],  # Invalid mask value 2
            labels=[-100, 2, 3],
        )
        with self.assertRaises(ValueError) as ctx:
            validate_tokenized_sft_record(bad_record)
        self.assertIn("invalid completion_mask", str(ctx.exception))

    def test_detects_mismatched_completion_label(self):
        bad_record = TokenizedSFTRecord(
            case_id="err-03",
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            completion_mask=[0, 1, 1],
            labels=[-100, 999, 3],  # 999 != 2
        )
        with self.assertRaises(ValueError) as ctx:
            validate_tokenized_sft_record(bad_record)
        self.assertIn("completion token has label=", str(ctx.exception))

    def test_detects_unmasked_prompt_token_label(self):
        bad_record = TokenizedSFTRecord(
            case_id="err-04",
            input_ids=[1, 2, 3],
            attention_mask=[1, 1, 1],
            completion_mask=[0, 0, 1],
            labels=[1, -100, 3],  # Index 0 has completion_mask=0 but label=1
        )
        with self.assertRaises(ValueError) as ctx:
            validate_tokenized_sft_record(bad_record)
        self.assertIn("unmasked prompt token", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
