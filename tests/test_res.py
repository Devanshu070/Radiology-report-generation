"""Synthetic unit tests for the shared local RES-style evaluator."""

import unittest

from evaluation.res import parse_report, res_proxy, score_tokens, weighted_word_edit


TEMPLATE = """FINDINGS:
LUNGS: Lungs are clear.
PLEURA: No pleural effusion or pneumothorax.
OTHER FINDINGS:

IMPRESSION:
No acute cardiopulmonary abnormality."""

REFERENCE = """FINDINGS:
LUNGS: Lungs are clear.
PLEURA: Small right pleural effusion. No pneumothorax.
OTHER FINDINGS:

IMPRESSION:
1. Small right pleural effusion."""


class ResProxyTests(unittest.TestCase):
    def test_identical_report_scores_zero(self):
        result = res_proxy(TEMPLATE, TEMPLATE, TEMPLATE)
        self.assertEqual(result["res_proxy"], 0.0)
        self.assertEqual(result["parse_error"], "")

    def test_correct_prediction_scores_zero(self):
        self.assertEqual(res_proxy(TEMPLATE, REFERENCE, REFERENCE)["res_proxy"], 0.0)

    def test_negation_and_laterality_are_scored_as_important(self):
        no_effusion = weighted_word_edit("No right pleural effusion", "Right pleural effusion")
        wrong_side = weighted_word_edit("Small left pleural effusion", "Small right pleural effusion")
        self.assertGreater(no_effusion, 0.0)
        self.assertGreater(wrong_side, 0.0)
        self.assertGreater(
            weighted_word_edit("No right pleural effusion", "Right pleural effusion"),
            weighted_word_edit("The right pleural effusion", "Right pleural effusion"),
        )

    def test_measurements_and_unit_aliases_are_normalized(self):
        self.assertEqual(score_tokens("1. Air-space opacity, 2 millimeters."), ["airspace", "opacity", "2", "mm"])
        self.assertEqual(weighted_word_edit("Opacity measures 2 millimeters", "Opacity measures 2 mm"), 0.0)
        self.assertGreater(weighted_word_edit("Lesion measures +2 mm", "Lesion measures -2 mm"), 0.0)

    def test_numbered_impression_prefix_is_ignored_for_scoring(self):
        self.assertEqual(weighted_word_edit("Small right pleural effusion.", "1. Small right pleural effusion."), 0.0)

    def test_hierarchical_empty_heading_is_detected_as_parent(self):
        report = "FINDINGS:\nTENDONS:\nSUPRASPINATUS: Intact.\nINFRASPINATUS: Intact.\n\nIMPRESSION:\nNormal."
        parsed = parse_report(report)
        self.assertIn("TENDONS", parsed.parents)
        self.assertEqual(parsed.fields["SUPRASPINATUS"], "Intact.")

    def test_unexpected_field_adds_error(self):
        prediction = REFERENCE.replace("OTHER FINDINGS:\n", "UNREQUESTED FIELD: Extra finding.\nOTHER FINDINGS:\n")
        result = res_proxy(TEMPLATE, REFERENCE, prediction)
        self.assertGreater(result["findings_proxy"], 0.0)

    def test_malformed_prediction_returns_maximum_score_and_error(self):
        result = res_proxy(TEMPLATE, REFERENCE, "FINDINGS:\nLUNGS: Clear.")
        self.assertEqual(result["res_proxy"], 1.0)
        self.assertEqual(result["findings_proxy"], 1.0)
        self.assertTrue(result["parse_error"])

    def test_malformed_template_raises(self):
        with self.assertRaisesRegex(ValueError, "FINDINGS.*IMPRESSION"):
            res_proxy("not a report", REFERENCE, REFERENCE)

    def test_critical_findings_and_impression_are_weighted(self):
        altered = REFERENCE.replace("Small right pleural effusion", "Large left pleural effusion")
        result = res_proxy(TEMPLATE, REFERENCE, altered)
        self.assertGreater(result["findings_proxy"], 0.0)
        self.assertGreater(result["impression_proxy"], 0.0)
        self.assertAlmostEqual(
            result["res_proxy"],
            0.65 * result["findings_proxy"] + 0.35 * result["impression_proxy"],
        )


if __name__ == "__main__":
    unittest.main()
