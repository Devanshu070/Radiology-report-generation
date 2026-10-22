"""Synthetic tests for Approach 1's independent preprocessing pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from in_context_rag.config import DEFAULT_DATA_ROOT, clinical_data_root
from in_context_rag.preprocess.load_data import load_cases
from in_context_rag.preprocess.split import create_split_manifest, training_pool
from in_context_rag.preprocess.template_sections import extract_template_sections, to_prompt_case
from in_context_rag.preprocess.validation import validate_cases
from in_context_rag.schemas import CaseRecord


def make_case(case_id: str, *, report: str | None = "Reference report.") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        modality="X-Ray",
        body_part="Chest",
        study_description="Chest radiograph",
        patient_age_band="Adult",
        patient_sex="Unspecified",
        template_content="FINDINGS:\nLungs are clear.\n\nIMPRESSION:\nNo acute finding.",
        dictation="Small right pleural effusion.",
        report=report,
    )


class TemplateSectionTests(unittest.TestCase):
    def test_extracts_findings_and_impression(self):
        result = extract_template_sections(
            "FINDINGS:\nLUNGS: Clear.\nPLEURA: No effusion.\n\nIMPRESSION:\nNo acute finding."
        )
        self.assertEqual(result.findings, "LUNGS: Clear.\nPLEURA: No effusion.")
        self.assertEqual(result.impression, "No acute finding.")

    def test_rejects_missing_duplicate_and_misordered_headings(self):
        invalid_templates = (
            "FINDINGS:\nNormal.",
            "IMPRESSION:\nNormal.\nFINDINGS:\nNormal.",
            "FINDINGS:\nOne.\nFINDINGS:\nTwo.\nIMPRESSION:\nNormal.",
        )
        for template in invalid_templates:
            with self.subTest(template=template), self.assertRaises(ValueError):
                extract_template_sections(template)

    def test_requires_nonempty_sections(self):
        with self.assertRaisesRegex(ValueError, "FINDINGS section"):
            extract_template_sections("FINDINGS:\n\nIMPRESSION:\nNormal.")
        with self.assertRaisesRegex(ValueError, "IMPRESSION section"):
            extract_template_sections("FINDINGS:\nNormal.\nIMPRESSION:\n")

    def test_builds_prompt_variables_without_changing_template_text(self):
        case = make_case("case-1")
        prompt_case = to_prompt_case(case)
        self.assertEqual(prompt_case.template_findings, "Lungs are clear.")
        self.assertEqual(prompt_case.template_impression, "No acute finding.")
        self.assertEqual(prompt_case.dictation, case.dictation)


class CaseLoadingAndValidationTests(unittest.TestCase):
    def test_schema_rejects_missing_required_input(self):
        with self.assertRaises(ValidationError):
            CaseRecord(case_id="case-1", modality="X-Ray")

    def test_duplicate_case_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
            validate_cases([make_case("same"), make_case("same")])

    def test_training_rows_require_reference_reports(self):
        with self.assertRaisesRegex(ValueError, "no reference report"):
            validate_cases([make_case("unlabeled", report=None)])
        self.assertEqual(len(validate_cases([make_case("unlabeled", report=None)], require_report=False)), 1)

    def test_loader_checks_headers_and_ignores_unmodeled_columns(self):
        columns = [
            "case_id", "modality", "body_part", "study_description", "patient_age_band",
            "patient_sex", "template_content", "dictation", "report", "local_extra",
        ]
        row = {
            "case_id": "case-1", "modality": "X-Ray", "body_part": "Chest",
            "study_description": "Chest radiograph", "patient_age_band": "Adult",
            "patient_sex": "Unspecified", "template_content": "FINDINGS:\nClear.\nIMPRESSION:\nNormal.",
            "dictation": "No acute finding.", "report": "No acute finding.", "local_extra": "ignored",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(row)
            self.assertEqual(load_cases(path)[0].case_id, "case-1")

            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["case_id"])
                writer.writeheader()
                writer.writerow({"case_id": "case-1"})
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_cases(path)

    def test_data_root_uses_environment_override_and_private_default(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"CLINICAL_DATA_ROOT": "/tmp/clinical-data"}):
            self.assertEqual(clinical_data_root(), Path("/tmp/clinical-data"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(clinical_data_root(), DEFAULT_DATA_ROOT)


class SplitIsolationTests(unittest.TestCase):
    def test_split_is_deterministic_and_partitions_are_disjoint(self):
        cases = [make_case(f"case-{index:02d}") for index in range(30)]
        first = create_split_manifest(cases, seed=17)
        second = create_split_manifest(reversed(cases), seed=17)
        self.assertEqual(first.assignments, second.assignments)
        self.assertFalse(first.train_ids & first.validation_ids)
        self.assertFalse(first.train_ids & first.test_ids)
        self.assertFalse(first.validation_ids & first.test_ids)
        self.assertEqual(first.train_ids | first.validation_ids | first.test_ids, {case.case_id for case in cases})

    def test_training_pool_excludes_validation_and_test(self):
        cases = [make_case(f"case-{index:02d}") for index in range(30)]
        manifest = create_split_manifest(cases)
        pool = training_pool(cases, manifest)
        pool_ids = {case.case_id for case in pool}
        self.assertEqual(pool_ids, set(manifest.train_ids))
        self.assertFalse(pool_ids & set(manifest.validation_ids | manifest.test_ids))

    def test_split_rejects_too_few_cases_and_manifest_mismatch(self):
        with self.assertRaisesRegex(ValueError, "At least three"):
            create_split_manifest([make_case("one"), make_case("two")])
        cases = [make_case(f"case-{index}") for index in range(5)]
        manifest = create_split_manifest(cases)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            training_pool(cases[:-1], manifest)


if __name__ == "__main__":
    unittest.main()
