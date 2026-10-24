"""Public offline unit tests for Approach 2 input preprocessing and split manifests."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from hybrid_retrieval_qlora.preprocess.load_data import (
    create_deterministic_splits,
    filter_training_pool,
    load_cases,
)
from hybrid_retrieval_qlora.preprocess.validation import validate_case, validate_cases
from hybrid_retrieval_qlora.schemas import ClinicalCase, SplitManifest


class ValidationTests(unittest.TestCase):
    """Test validation of clinical case attributes and contracts."""

    def test_valid_case_dict(self):
        raw = {
            "case_id": "c-01",
            "modality": "X-Ray",
            "body_part": "Chest",
            "study_description": "2 views",
            "patient_age_band": "Adult",
            "patient_sex": "M",
            "template_content": "FINDINGS:\nIMPRESSION:",
            "dictation": "Clear lungs.",
            "report": "FINDINGS: Clear. IMPRESSION: Clear.",
        }
        case = validate_case(raw, require_report=True)
        self.assertEqual(case.case_id, "c-01")
        self.assertEqual(case.modality, "X-Ray")
        self.assertEqual(case.report, "FINDINGS: Clear. IMPRESSION: Clear.")

    def test_missing_required_fields_raises(self):
        raw = {
            "case_id": "c-01",
            "modality": "",  # Empty
            "body_part": "Chest",
            "template_content": "FINDINGS:\nIMPRESSION:",
            "dictation": "Clear lungs.",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_case(raw)
        self.assertIn("modality", str(ctx.exception))

    def test_missing_report_when_required_raises(self):
        raw = {
            "case_id": "c-01",
            "modality": "CT",
            "body_part": "Chest",
            "template_content": "FINDINGS:\nIMPRESSION:",
            "dictation": "Clear lungs.",
            "report": "",
        }
        with self.assertRaises(ValueError) as ctx:
            validate_case(raw, require_report=True)
        self.assertIn("report", str(ctx.exception))

    def test_duplicate_case_id_rejected(self):
        records = [
            {
                "case_id": "c-dup",
                "modality": "CT",
                "body_part": "Chest",
                "template_content": "T",
                "dictation": "D",
            },
            {
                "case_id": "c-dup",
                "modality": "MRI",
                "body_part": "Brain",
                "template_content": "T",
                "dictation": "D2",
            },
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_cases(records)
        self.assertIn("Duplicate case_id", str(ctx.exception))


class LoadDataAndSplitTests(unittest.TestCase):
    """Test loading synthetic CSVs and generating deterministic partition manifests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "synthetic.csv"

        fieldnames = [
            "case_id", "modality", "body_part", "study_description",
            "patient_age_band", "patient_sex", "template_content",
            "dictation", "report",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "case_id": f"synth-{i:02d}",
                    "modality": "CT" if i % 2 == 0 else "MRI",
                    "body_part": "Chest" if i % 2 == 0 else "Abdomen",
                    "study_description": "Routine scan",
                    "patient_age_band": "50-60",
                    "patient_sex": "F",
                    "template_content": "FINDINGS:\nIMPRESSION:",
                    "dictation": f"Dictation content for case {i}",
                    "report": f"Report content for case {i}",
                })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_cases_from_csv(self):
        cases = load_cases(self.csv_path, require_report=True)
        self.assertEqual(len(cases), 10)
        self.assertEqual(cases[0].case_id, "synth-00")

    def test_deterministic_split_creation(self):
        cases = load_cases(self.csv_path, require_report=True)
        manifest, train_cases, val_cases, test_cases = create_deterministic_splits(
            cases,
            seed=42,
            test_fraction=0.20,
            validation_fraction=0.10,
        )

        self.assertEqual(len(test_cases), 2)
        self.assertEqual(len(val_cases), 1)
        self.assertEqual(len(train_cases), 7)

        # Disjoint verification
        self.assertTrue(manifest.train_ids.isdisjoint(manifest.validation_ids))
        self.assertTrue(manifest.train_ids.isdisjoint(manifest.test_ids))
        self.assertTrue(manifest.validation_ids.isdisjoint(manifest.test_ids))

        # Re-running with same seed produces identical assignments
        manifest2, _, _, _ = create_deterministic_splits(cases, seed=42)
        self.assertEqual(manifest.assignments, manifest2.assignments)

    def test_filter_training_pool(self):
        cases = load_cases(self.csv_path, require_report=True)
        manifest, train_cases, val_cases, test_cases = create_deterministic_splits(cases, seed=42)

        pool = filter_training_pool(cases, manifest)
        self.assertEqual(len(pool), len(train_cases))
        self.assertEqual(set(c.case_id for c in pool), manifest.train_ids)

        for case in pool:
            self.assertTrue(bool(case.report))


if __name__ == "__main__":
    unittest.main()
