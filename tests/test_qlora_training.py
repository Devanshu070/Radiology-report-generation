"""Unit tests for Approach 2 QLoRA training, dataset preparation, and RES evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from hybrid_retrieval_qlora.preprocess.completion_mask import MockQwenTokenizer
from hybrid_retrieval_qlora.preprocess.load_data import create_deterministic_splits, filter_training_pool
from hybrid_retrieval_qlora.retrieval.hybrid_retriever import HybridRetriever
from hybrid_retrieval_qlora.schemas import ClinicalCase, GenerationResult
from hybrid_retrieval_qlora.training.evaluate import evaluate_reports
from hybrid_retrieval_qlora.training.qlora_config import QLoRAConfig
from hybrid_retrieval_qlora.training.sft_dataset import prepare_sft_records, to_hf_dataset
from hybrid_retrieval_qlora.training.train import MaskedSFTDataCollator, train_qlora


def _create_sample_cases() -> list[ClinicalCase]:
    """Create deterministic synthetic clinical cases for testing."""
    return [
        ClinicalCase(
            case_id=f"case_{i:03d}",
            modality="CT" if i % 2 == 0 else "XR",
            body_part="CHEST",
            study_description="Routine examination",
            patient_age_band="50-59",
            patient_sex="F" if i % 2 == 0 else "M",
            template_content=(
                "FINDINGS:\nLUNGS: Clear bilaterally without focal consolidation.\n"
                "PLEURA: No pleural effusion or pneumothorax.\n"
                "IMPRESSION:\nNormal examination."
            ),
            dictation=(
                f"Dictation for patient {i}: small left pleural effusion noted, "
                "lungs otherwise remain clear."
            ),
            report=(
                "FINDINGS:\nLUNGS: Clear bilaterally without focal consolidation.\n"
                f"PLEURA: Small left pleural effusion noted in case {i}.\n"
                f"IMPRESSION:\n1. Small left pleural effusion in case {i}."
            ),
        )
        for i in range(12)
    ]


class MockTrainerState:
    def __init__(self):
        self.best_model_checkpoint = "/tmp/mock_run/checkpoints/checkpoint-20"
        self.best_metric = 0.425
        self.log_history = [
            {"step": 10, "loss": 0.85, "learning_rate": 5e-5},
            {"step": 20, "eval_loss": 0.425, "epoch": 2.0},
        ]


class MockTrainer:
    """Mock TRL SFTTrainer for offline unit testing without GPU/downloads."""

    def __init__(self, model, args, train_dataset, eval_dataset, processing_class, data_collator):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.state = MockTrainerState()
        self.saved_model_path = None
        self.trained = False

    def train(self, resume_from_checkpoint=None):
        self.trained = True
        return {"train_runtime": 1.23, "train_loss": 0.5}

    def save_model(self, output_dir):
        self.saved_model_path = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "adapter_model.safetensors").write_text("mock-adapter-weights")

    def save_state(self):
        pass


class TestQLoRATraining(unittest.TestCase):
    """Test suite for QLoRA configuration, SFT dataset preparation, training, and evaluation."""

    def setUp(self):
        self.tokenizer = MockQwenTokenizer()
        self.cases = _create_sample_cases()

    def test_qlora_config_defaults_and_validation(self):
        """Verify QLoRAConfig adheres to Approach 2 canonical parameters."""
        config = QLoRAConfig()

        self.assertEqual(config.model_name, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.model_revision, "main")
        self.assertEqual(config.lora_rank, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertEqual(config.num_train_epochs, 2)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.max_seq_length, 6144)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(config.load_in_4bit)

        sig = config.get_training_signature()
        self.assertEqual(sig["model"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(sig["lora_rank"], 16)
        self.assertIn("packages", sig)

        # Validation constraints
        with self.assertRaises(ValidationError):
            QLoRAConfig(lora_rank=0)
        with self.assertRaises(ValidationError):
            QLoRAConfig(validation_fraction=1.5)

    def test_prepare_sft_records_and_loss_masking(self):
        """Confirm token alignment, loss masking, and prompt exclusion in SFT dataset."""
        train_cases = self.cases[:4]
        records, usages, formatted = prepare_sft_records(
            train_cases,
            self.tokenizer,
            supervised=True,
            top_k=0,
            max_input_tokens=2048,
            max_seq_length=4096,
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(len(usages), 4)
        self.assertEqual(len(formatted), 4)

        for rec, case in zip(records, train_cases):
            self.assertEqual(rec.case_id, case.case_id)
            # Array lengths must strictly match
            n = len(rec.input_ids)
            self.assertEqual(len(rec.attention_mask), n)
            self.assertEqual(len(rec.completion_mask), n)
            self.assertEqual(len(rec.labels), n)

            # Completion mask invariants
            ones_count = rec.completion_mask.count(1)
            zeros_count = rec.completion_mask.count(0)
            self.assertGreater(ones_count, 0)
            self.assertGreater(zeros_count, 0)

            # Target-only loss: prompt tokens must be -100, completion tokens must equal input_ids
            for tid, mask_val, label in zip(rec.input_ids, rec.completion_mask, rec.labels):
                if mask_val == 1:
                    self.assertEqual(label, tid)
                else:
                    self.assertEqual(label, -100)

        # Convert to dataset
        ds = to_hf_dataset(records)
        self.assertEqual(len(ds["case_id"]), 4)
        self.assertIn("input_ids", ds)
        self.assertIn("labels", ds)
        self.assertIn("completion_mask", ds)
        self.assertNotIn("prompt", ds)
        self.assertNotIn("completion", ds)

        import torch

        collator = MaskedSFTDataCollator(self.tokenizer)
        # Trainer may remove non-model columns before handing records to the collator.
        features = [
            {key: value for key, value in ds[index].items() if key != "completion_mask"}
            for index in range(2)
        ]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertTrue(torch.all(batch["labels"][batch["attention_mask"] == 0] == -100))
        self.assertTrue(torch.all(batch["labels"][batch["input_ids"] != batch["labels"]] == -100))

    def test_split_leakage_prevented_in_sft_demonstrations(self):
        """Verify candidate retrieval strictly enforces training split boundary."""
        manifest, train_cases, val_cases, test_cases = create_deterministic_splits(
            self.cases, seed=42, validation_fraction=0.25
        )
        training_pool = filter_training_pool(self.cases, manifest)

        retriever = HybridRetriever(training_pool, manifest=manifest, top_k=2)

        import numpy as np

        rng = np.random.default_rng(42)
        cand_embeddings = rng.standard_normal((len(training_pool), 768)).astype(np.float32)
        cand_embeddings /= np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
        query_embeddings = rng.standard_normal((len(val_cases), 768)).astype(np.float32)
        query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Prepare records for validation cases with retrieval
        records, usages, formatted = prepare_sft_records(
            val_cases,
            self.tokenizer,
            retriever=retriever,
            candidate_embeddings=cand_embeddings,
            query_embeddings=query_embeddings,
            supervised=True,
            top_k=2,
        )

        train_id_set = manifest.train_ids
        for usage in usages:
            # Demonstrations must only come from train split
            for demo_id in usage.demonstration_ids:
                self.assertIn(demo_id, train_id_set)
                self.assertNotIn(demo_id, manifest.validation_ids)
                self.assertNotIn(demo_id, manifest.test_ids)

    def test_train_qlora_mock_execution(self):
        """Verify train_qlora pipeline logic with mock model and trainer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir) / "test_run"
            config = QLoRAConfig(output_dir=temp_dir)

            train_cases = self.cases[:3]
            val_cases = self.cases[3:5]
            train_records, _, train_fmt = prepare_sft_records(train_cases, self.tokenizer)
            val_records, _, val_fmt = prepare_sft_records(val_cases, self.tokenizer)

            train_ds = to_hf_dataset(train_records)
            val_ds = to_hf_dataset(val_records)

            class MockModel:
                def __init__(self):
                    self.config = type("Config", (), {"pad_token_id": 151643, "use_cache": False})()

            result = train_qlora(
                config,
                train_ds,
                val_ds,
                self.tokenizer,
                model=MockModel(),
                trainer_cls=MockTrainer,
                run_dir=run_path,
            )

            self.assertTrue((run_path / "run_config.json").is_file())
            self.assertTrue((run_path / "adapter" / "adapter_model.safetensors").is_file())
            self.assertEqual(result["best_validation_loss"], 0.425)
            self.assertEqual(result["model"].config.pad_token_id, 151643)

            meta = json.loads((run_path / "run_config.json").read_text())
            self.assertEqual(meta["lora_rank"], 16)
            self.assertEqual(meta["best_validation_loss"], 0.425)

    def test_train_qlora_checks_mask_invariants_for_every_record(self):
        train_records, _usages, _formatted = prepare_sft_records(
            self.cases[:3], self.tokenizer, top_k=0
        )
        train_ds = to_hf_dataset(train_records)
        train_ds["labels"][1][0] = train_ds["input_ids"][1][0]
        val_records, _val_usages, _val_formatted = prepare_sft_records(
            self.cases[3:5], self.tokenizer, top_k=0
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                train_qlora(
                    QLoRAConfig(output_dir=temp_dir),
                    train_ds,
                    to_hf_dataset(val_records),
                    self.tokenizer,
                    model=object(),
                    trainer_cls=MockTrainer,
                    run_dir=Path(temp_dir) / "invalid_run",
                )

    def test_evaluate_reports_with_canonical_res(self):
        """Verify evaluation against canonical RES metric with summary aggregation."""
        cases = self.cases[:3]

        gen_results = [
            # Case 0: High agreement with expected report
            GenerationResult(
                case_id=cases[0].case_id,
                variant="finetuned_retrieval",
                generated_report=cases[0].report,
                status="success",
                error="",
                generated_tokens=50,
            ),
            # Case 1: Partial agreement
            GenerationResult(
                case_id=cases[1].case_id,
                variant="finetuned_retrieval",
                generated_report="FINDINGS:\nLUNGS: Clear.\nIMPRESSION:\nNormal study.",
                status="success",
                error="",
                generated_tokens=20,
            ),
            # Case 2: Failed generation
            GenerationResult(
                case_id=cases[2].case_id,
                variant="finetuned_retrieval",
                generated_report="",
                status="failed",
                error="Out of memory",
                generated_tokens=0,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_csv = Path(temp_dir) / "report_scores.csv"
            scores, summary = evaluate_reports(
                cases,
                gen_results,
                variant="finetuned_retrieval",
                output_path=out_csv,
            )

            self.assertEqual(len(scores), 3)
            self.assertEqual(summary.total_cases, 3)
            self.assertEqual(summary.successful_cases, 2)

            # Case 0 (exact match) should have RES score 0.0
            self.assertAlmostEqual(scores[0].res_proxy, 0.0, places=4)
            # Case 2 (failed) should receive penalty 1.0
            self.assertEqual(scores[2].res_proxy, 1.0)
            self.assertEqual(scores[2].status, "failed")

            # CSV file should be created
            self.assertTrue(out_csv.is_file())

    def test_generate_reports_mock(self):
        """Verify report generation with injected generator and error handling."""
        from hybrid_retrieval_qlora.inference.generate_reports import generate_reports

        cases = self.cases[:2]

        def mock_gen_fn(prompt: str) -> str:
            if "patient 1" in prompt:
                raise RuntimeError("Inference timeout simulation")
            return "FINDINGS:\nLUNGS: Clear bilaterally.\nIMPRESSION:\n1. Unremarkable examination."

        with tempfile.TemporaryDirectory() as temp_dir:
            out_csv = Path(temp_dir) / "reports.csv"
            results = generate_reports(
                model=None,
                tokenizer=self.tokenizer,
                cases=cases,
                generate_fn=mock_gen_fn,
                output_path=out_csv,
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].case_id, cases[0].case_id)
            self.assertEqual(results[0].status, "success")
            self.assertIn("FINDINGS", results[0].generated_report)

            self.assertEqual(results[1].case_id, cases[1].case_id)
            self.assertEqual(results[1].status, "failed")
            self.assertIn("Inference timeout simulation", results[1].error)

            self.assertTrue(out_csv.is_file())

    def test_generate_reports_rejects_missing_report_sections(self):
        from hybrid_retrieval_qlora.inference.generate_reports import generate_reports

        results = generate_reports(
            model=None,
            tokenizer=self.tokenizer,
            cases=self.cases[:1],
            generate_fn=lambda _: "The lungs are clear.",
        )
        self.assertEqual(results[0].status, "failed")
        self.assertIn("FINDINGS followed by IMPRESSION", results[0].error)


if __name__ == "__main__":
    unittest.main()
