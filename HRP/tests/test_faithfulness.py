import json
import unittest
from unittest.mock import Mock

from src.hrp_faithfulness import (
    local_request, parse_answer, run_benchmark, score_answer, write_verified_explanation,
)


class FaithfulnessTests(unittest.TestCase):
    def setUp(self):
        self.signed = {"age": 0.6, "pulse": -0.4, "sodium": 0.2, "other": -0.01}
        self.correct = {
            "explanation": (
                "Age increases predicted risk. Pulse decreases predicted risk. "
                "Sodium increases predicted risk."
            ),
            "drivers": [
                {"feature": feature, "direction": "increases" if value > 0 else "decreases"}
                for feature, value in list(self.signed.items())[:3]
            ],
        }
        self.case = {
            "patient_id": 1, "scenario": "Synthetic", "risk": "High", "sign": "Negative",
            "prediction": 0.6, "values": dict.fromkeys(self.signed, 2),
            "shap": self.signed, "reversed_features": ("pulse",),
        }

    def test_json_validation(self):
        raw = json.dumps(self.correct)
        self.assertEqual(parse_answer(f"```json\n{raw}\n```", self.signed), self.correct)
        for invalid in (
            "before " + raw, raw + raw, "[]",
            '{"explanation":"one","explanation":"two","drivers":[]}',
            json.dumps({**self.correct, "drivers": [self.correct["drivers"][0]] * 2}),
            json.dumps({**self.correct, "drivers": [{"feature": [], "direction": "increases"}]}),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_answer(invalid, self.signed)

    def test_rank_and_prose_are_independent(self):
        scored = score_answer({
            "explanation": "Age decreases predicted risk. Pulse causes readmission.",
            "drivers": list(reversed(self.correct["drivers"])),
        }, self.signed)
        self.assertEqual(scored["top3_overlap"], 3)
        self.assertEqual(scored["rank_positions"], 1)
        self.assertFalse(scored["exact_top3_order"])
        self.assertEqual(scored["signed_top3_matches"], 3)
        self.assertEqual(scored["prose_missing"], 2)
        self.assertEqual(scored["prose_wrong_direction"], 1)
        self.assertEqual(scored["prose_unsupported"], 1)

    def test_revision_uses_grounded_draft_and_checks_result(self):
        draft = {**self.correct, "explanation": "Pulse increases predicted risk."}
        infer = Mock(side_effect=[json.dumps(draft), json.dumps(draft), json.dumps(self.correct)])
        records, calls = run_benchmark([self.case], infer, call_limit=3, wall_seconds=20)
        self.assertEqual(calls, 3)
        self.assertNotIn("Signed SHAP values", infer.call_args_list[0].args[0])
        self.assertIn("Correct the direction", infer.call_args_list[2].args[0])
        self.assertEqual(records.prose_wrong_direction.tolist(), [1, 1, 0])
        self.assertEqual(records.iloc[2].verification, "revised_clean")
        self.assertEqual(records.iloc[2].reversed_prose_correct, 1)

    def test_parse_failure_and_budget_are_visible(self):
        infer = Mock(side_effect=["invalid", json.dumps(self.correct)])
        records, calls = run_benchmark([self.case], infer, call_limit=2, wall_seconds=20)
        self.assertEqual(calls, 2)
        self.assertEqual(records.status.tolist(), ["parse_error", "ok", "ok"])
        self.assertEqual(records.iloc[2].verification, "accepted")
        infer = Mock(return_value="invalid")
        records, calls = run_benchmark([self.case], infer, call_limit=2, wall_seconds=20)
        self.assertEqual(calls, 2)
        self.assertEqual(records.iloc[2].verification, "revision_budget_skipped_retained_draft")

    def test_local_transport_rejects_lookalike_hosts(self):
        with self.assertRaises(ValueError):
            local_request("GET", "http://127.0.0.1:11434@example.org/api/tags", timeout=1)

    def test_writer_receives_corrected_rank_and_signs(self):
        infer = Mock(return_value=json.dumps({"explanation": self.correct["explanation"]}))
        result = write_verified_explanation(self.case, list(reversed(self.correct["drivers"])), infer)
        self.assertTrue(result["draft_corrected"])
        self.assertEqual(result["drivers"], self.correct["drivers"])
        self.assertEqual(result["writer_status"], "accepted")
        self.assertIn(json.dumps(self.correct["drivers"]), infer.call_args.args[0])

    def test_writer_falls_back_on_wrong_direction_or_extra_claim(self):
        for prose in (
            self.correct["explanation"].replace("Pulse decreases", "Pulse increases"),
            self.correct["explanation"] + " Age causes readmission.",
        ):
            with self.subTest(prose=prose):
                infer = Mock(return_value=json.dumps({"explanation": prose}))
                result = write_verified_explanation(self.case, [], infer)
                self.assertEqual(result["writer_status"], "fallback")
                self.assertEqual(result["explanation"], self.correct["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
