from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment2_scoring import SCORER_VERSION, score_answer, score_experiment
from experiment2_smoke import fictional_cases
from llm_runner import run_experiment, verify_and_revise
from llm_verification import VERIFIER_VERSION, Answer, Driver, deterministic_answer


def raw(answer):
    return json.dumps(asdict(answer))


class CompletenessTests(unittest.TestCase):
    def test_zero_to_three_drivers_and_exact_b_reuse(self):
        case = fictional_cases()[0]
        complete = deterministic_answer(case)
        prediction, tail = complete.explanation.split(". ", 1)
        clauses = tail.removesuffix(".").split(", while ")
        for count in range(4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                prose = ", while ".join(clauses[:count]) if count else "The drivers are uncertain"
                answer = Answer(f"{prediction}. {prose}.", complete.drivers[:count])
                response = "\n " + raw(answer) + " \n"
                infer = Mock(side_effect=[response, response, raw(complete)])
                state = run_experiment([case], infer, directory)
                a, b, c = state["records"].values()
                self.assertEqual(a["status"], "ok")
                self.assertEqual(b["status"], "ok")
                self.assertEqual(a["call"]["raw"], response)
                self.assertEqual(b["call"]["raw"], response)
                self.assertEqual(c["verification"]["original_raw"], response)
                audit = c["verification"]["original"]
                self.assertEqual(audit["status"], "verified_complete" if count == 3 else "correct_incomplete")
                self.assertTrue(audit["correct"])
                self.assertEqual(audit["complete"], count == 3)
                self.assertEqual(len(audit["structured"]["missing"]), 3 - count)
                self.assertEqual(len(audit["missing_prose_drivers"]), 3 - count)
                self.assertEqual(infer.call_count, 2 if count == 3 else 3)
                if count < 3:
                    self.assertTrue(c["verification"]["revision_accepted"])
                    self.assertIn("missing_drivers", infer.call_args.args[0])
                self.assertEqual(c["verification"]["status"], "verified_complete")

    def test_prose_omissions_and_incomplete_revision_require_fallback(self):
        case = fictional_cases()[0]
        complete = deterministic_answer(case)
        incomplete = replace(complete, explanation="The drivers are uncertain.")
        infer = Mock(return_value=raw(incomplete))
        outcome = verify_and_revise(case, raw(incomplete), infer)
        self.assertTrue(outcome["original"]["structured"]["complete"])
        self.assertEqual(outcome["original"]["status"], "correct_incomplete")
        self.assertEqual(outcome["revision"]["status"], "correct_incomplete")
        self.assertEqual(outcome["source"], "deterministic_fallback")
        self.assertEqual(outcome["status"], "verified_complete")
        self.assertIn("missing_prose_drivers", infer.call_args.args[0])

    def test_empty_reference_can_be_complete(self):
        case = fictional_cases()[4]
        draft = deterministic_answer(case)
        infer = Mock()
        outcome = verify_and_revise(case, raw(draft), infer)
        self.assertTrue(outcome["draft_accepted"])
        self.assertEqual(outcome["status"], "verified_complete")
        infer.assert_not_called()


class ScoringTests(unittest.TestCase):
    def test_hand_calculated_scores_for_order_sign_and_narrative(self):
        case = fictional_cases()[0]
        answer = Answer(
            "Age increases predicted risk. Weight influences the model's prediction. "
            "Creatinine might play a complex role.",
            (Driver("prior_admissions", "negative"), Driver("age", "positive"), Driver("sodium", "negative")),
        )
        metrics = score_answer(case, answer)
        expected = {
            "top_three_overlap": 2, "signed_top_three_agreement": 1, "rank_position_agreement": 0,
            "exact_top_three_ordering": False, "missing_influential_drivers": 1,
            "incorrect_feature_directions": 1, "narrative_contradictions": 2,
            "unsupported_model_claims": 1, "narrative_claims_checked": 2, "narrative_claims_unparsed": 1,
        }
        self.assertEqual({name: metrics[name] for name in expected}, expected)
        wrong_sign = replace(answer, drivers=(Driver("age", "negative"), Driver("prior_admissions", "positive"),
                                               Driver("creatinine", "negative")))
        scored = score_answer(case, wrong_sign)
        self.assertTrue(scored["exact_top_three_ordering"])
        self.assertEqual(scored["rank_position_agreement"], 3)
        self.assertEqual(scored["signed_top_three_agreement"], 2)
        self.assertEqual(scored["narrative_contradictions"], 2)  # Direction and JSON conflict share a clause.

    def test_pairs_failures_and_source_strata(self):
        cases = fictional_cases()[:3]
        good = [raw(deterministic_answer(case)) for case in cases]
        empty = raw(Answer("The drivers are uncertain.", ()))
        infer = Mock(side_effect=[
            empty, good[0],                       # A abstains, C accepts B.
            "malformed", empty, good[1],          # A fails parsing, C corrects B.
            good[2], empty, TimeoutError("test"), # C's failed revision produces fallback.
        ])
        with tempfile.TemporaryDirectory() as directory:
            state = run_experiment(cases, infer, directory)
            report = score_experiment(cases, state)
            repeat = run_experiment(cases, infer, directory)
        self.assertEqual(state, repeat)
        self.assertEqual(infer.call_count, 8)
        self.assertEqual(report["operations"]["calls"], 8)
        self.assertEqual(report["operations"]["parse_failures"], 1)
        self.assertEqual(report["operations"]["inference_failures"], {"timeout": 1})
        self.assertGreaterEqual(report["operations"]["latency_seconds"], 0)
        self.assertEqual(report["by_condition"]["A"]["evaluated"], 2)
        self.assertEqual(report["by_condition"]["C"]["original_verification_statuses"],
                         {"verified_complete": 1, "correct_incomplete": 2})
        self.assertEqual(set(report["by_condition_and_source"]["C"]),
                         {"grounded_draft", "model_revision", "deterministic_fallback"})
        self.assertEqual(list(report["by_condition_and_source"]["A"]), ["independent_generation"])
        self.assertEqual([report["paired"][pair]["pairs"] for pair in ("A_to_B", "B_to_C", "A_to_C")], [2, 3, 2])
        self.assertEqual(report["paired"]["A_to_B"]["excluded_pairs"], 1)
        self.assertEqual(report["paired"]["A_to_B"]["metrics"]["signed_top_three_agreement"]["mean_change"], 0)
        self.assertEqual(report["paired"]["B_to_C"]["metrics"]["signed_top_three_agreement"]["mean_change"], 2)
        self.assertEqual(report["paired"]["A_to_C"]["metrics"]["signed_top_three_agreement"]["mean_change"], 1.5)
        with self.assertRaises(ValueError):
            score_experiment(reversed(cases), state)
        list(state["records"].values())[2]["verification"]["original_raw"] = "altered"
        with self.assertRaisesRegex(ValueError, "exact original"):
            score_experiment(cases, state)

    def test_checkpoint_versions_provenance_and_pending_scores(self):
        case = fictional_cases()[0]
        infer = Mock(return_value=raw(deterministic_answer(case)))
        with tempfile.TemporaryDirectory() as directory:
            state = run_experiment([case], infer, directory, max_calls=1, provenance={"id": "fictional"})
            self.assertEqual(state["manifest"]["scorer_version"], SCORER_VERSION)
            self.assertEqual(state["manifest"]["verifier_version"], VERIFIER_VERSION)
            self.assertEqual(score_experiment([case], state)["operations"]["statuses"], {"ok": 1, "pending": 2})
            for constant in ("SCORER_VERSION", "VERIFIER_VERSION", "SHAP_ZERO_TOLERANCE"):
                with self.subTest(constant=constant), patch(f"llm_runner.{constant}", "changed"):
                    with self.assertRaisesRegex(ValueError, "Incompatible"):
                        run_experiment([case], infer, directory, provenance={"id": "fictional"})
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                run_experiment([case], infer, directory, provenance={"id": "different"})
            self.assertEqual(infer.call_count, 1)

    def test_benchmark_sized_fictional_run_preserves_terminal_failures(self):
        template = fictional_cases()[0]
        cases = tuple(replace(template, case_id=f"fixture-{index:02}") for index in range(72))
        infer = Mock(side_effect=["invalid JSON", ConnectionError("fixture unavailable")] * len(cases))
        with tempfile.TemporaryDirectory() as directory:
            state = run_experiment(cases, infer, directory, max_calls=216)
            report = score_experiment(cases, state)
        self.assertIsNone(state["pause_reason"])
        self.assertEqual(report["operations"]["records"], 216)
        self.assertEqual(report["operations"]["statuses"], {"parse_error": 72, "inference_error": 72, "ok": 72})
        self.assertEqual(report["operations"]["intervention_counts"]["fallback_used"], 72)
        self.assertEqual(report["operations"]["calls"], 144)
        self.assertEqual(report["paired"]["B_to_C"]["excluded_pairs"], 72)
