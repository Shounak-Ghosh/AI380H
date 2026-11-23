from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment2_smoke import fictional_cases
from llm_verification import Answer, Driver, deterministic_answer, expected_drivers, verify_attribution
from narrative_verifier import verify_narrative


class NarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = fictional_cases()
        cls.case = cls.cases[1]

    def audit(self, prose, case=None, drivers=()):
        return verify_narrative(case or self.case, Answer(prose, drivers))

    def test_explicit_directions_and_approved_aliases(self):
        result = self.audit(
            "Age increases the model’s predicted risk. Serum creatinine increases predicted risk. "
            "Sodium is associated with a lower model prediction.", case=self.cases[6],
        )
        self.assertFalse(result.passed)
        self.assertEqual([(c.feature, c.status) for c in result.claims],
                         [("age", "supported"), ("creatinine", "supported"), ("sodium", "contradicted")])
        result = self.audit("prior_admissions increases predicted risk. Prior hospital admissions are the strongest contributor.")
        self.assertTrue(result.passed)
        self.assertEqual([c.code for c in result.claims], ["direction", "ranking"])

    def test_negative_association(self):
        result = self.audit("Sodium is associated with a lower model prediction.", case=self.cases[0])
        self.assertTrue(result.passed)
        self.assertEqual(result.claims[0].status, "supported")

    def test_correct_json_does_not_hide_wrong_narrative(self):
        drivers = expected_drivers(self.case)
        self.assertTrue(verify_attribution(self.case, drivers).passed)
        result = self.audit("Age decreases the predicted risk.", drivers=drivers)
        self.assertFalse(result.passed)
        self.assertEqual({c.code for c in result.claims}, {"direction", "driver_conflict"})

    def test_rank_claims_and_ties(self):
        result = self.audit("Age is the strongest driver. Prior admissions are the strongest contributor.")
        self.assertEqual([c.status for c in result.claims], ["contradicted", "supported"])
        self.assertTrue(self.audit("Creatinine is the strongest contributor.", case=self.cases[7]).passed)
        self.assertFalse(self.audit("Sodium is a top three factor.", case=self.cases[7]).passed)

    def test_unsupported_influence_and_clinical_claims(self):
        for prose, code in (
            ("Weight influences the model's prediction.", "unsupported_feature"),
            ("Sodium missing is an important driver.", "negligible"),
            ("Creatinine shows kidney disease.", "clinical"),
            ("Prior admissions cause readmission.", "causal"),
            ("The age SHAP contribution is 10 percentage points.", "scale"),
            ("Age contributes 10% to predicted risk.", "scale"),
        ):
            with self.subTest(prose=prose):
                result = self.audit(prose)
                self.assertFalse(result.passed)
                self.assertEqual(result.claims[0].status, "contradicted")
                self.assertEqual(result.claims[0].code, code)

    def test_missing_versus_observed_lab(self):
        missing = self.cases[2]
        for prose in ("Sodium was 138 mEq/L.", "Sodium was normal.",
                      "Sodium of 138 mEq/L decreases predicted risk.",
                      "Elevated sodium decreases predicted risk."):
            with self.subTest(prose=prose):
                self.assertIn("missing_observation", [claim.code for claim in self.audit(prose, missing).claims])
        self.assertTrue(self.audit("Sodium was missing.", missing).passed)
        self.assertTrue(self.audit("Sodium was 138 mEq/L.").passed)
        self.assertFalse(self.audit("Sodium was missing.").passed)
        self.assertFalse(self.audit("Sodium was 139 mEq/L.").passed)
        self.assertEqual(self.audit("Sodium missing decreases the model's predicted risk.", missing).claims[0].status,
                         "supported")

    def test_non_attribution_and_ambiguous_prose_remain_distinct(self):
        prediction = f"The model estimates a {self.case.predicted_risk:.1%} readmission probability."
        result = self.audit(prediction + " Age may be playing a complex role.")
        self.assertEqual([c.status for c in result.claims], ["non_attribution", "ambiguous"])
        self.assertFalse(result.passed)
        self.assertTrue(self.audit(prediction + " The drivers are uncertain.").passed)
        self.assertFalse(self.audit("The model estimates a 99% readmission probability.").passed)
        self.assertFalse(self.audit("Age increases predicted risk because treatment was delayed.").passed)

    def test_live_style_probability_and_clear_risk_antecedent(self):
        case = self.cases[2]
        probability = f"The model predicts a {case.predicted_risk:.1%} risk of readmission."
        result = self.audit(
            probability + " Sodium missing decreases predicted risk, while age also increases it, "
            "and creatinine decreases it.", case,
        )
        self.assertTrue(result.passed, result.claims)
        self.assertEqual([claim.status for claim in result.claims], ["non_attribution", "supported", "supported", "supported"])
        self.assertFalse(self.audit(probability + " Creatinine decreases it.", case).passed)
        self.assertFalse(self.audit("Sodium missing decreases predicted risk, while age decreases it.", case).passed)

    def test_correct_scale_description_is_descriptive(self):
        result = self.audit("SHAP contributions describe changes in raw log-odds, not percentage-point changes.")
        self.assertTrue(result.passed)
        self.assertEqual(result.claims[0].status, "non_attribution")

    def test_deterministic_text_is_rechecked_for_all_fixtures(self):
        for case in self.cases:
            with self.subTest(case=case.case_id):
                answer = deterministic_answer(case)
                result = verify_narrative(case, answer)
                self.assertTrue(result.passed, result.claims)
