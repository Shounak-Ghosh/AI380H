from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment2_smoke import fictional_cases, run_smoke
from llm_runner import InferenceError, ModelConfig, Ollama, checkpoint_key, run_experiment, verify_and_revise
from llm_verification import (
    COMMON_INSTRUCTIONS, Driver, build_prompt, deterministic_answer, expected_drivers, feature_keys,
    parse_response, verify_attribution,
)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = fictional_cases()
        cls.case = cls.cases[0]
        cls.good = deterministic_answer(cls.case)
        cls.raw = json.dumps(asdict(cls.good))

    def test_input_isolation(self):
        injected = replace(self.case, features=(
            {**self.case.features[0], "subject_id": 12345, "observed_outcome": 1},
            *self.case.features[1:],
        ))
        baseline, grounded = [build_prompt(injected, condition) for condition in "AB"]
        a, b = [json.loads(prompt.split("Input JSON:\n")[1]) for prompt in (baseline, grounded)]
        self.assertEqual(a, {key: b[key] for key in a})
        self.assertEqual(set(a), {"features", "predicted_risk"})
        self.assertIn("ranked_influential_features", b["shap"])
        for prompt in (baseline, grounded):
            self.assertTrue(prompt.startswith(COMMON_INSTRUCTIONS))
            for token in (injected.case_id, "subject_id", "observed_outcome", "12345"):
                self.assertNotIn(token, prompt)
        self.assertNotIn("shap", baseline.lower())
        with self.assertRaises(ValueError):
            build_prompt(injected, "C")

    def test_fixture_additivity(self):
        probabilities = []
        for case in self.cases:
            reference = case.shap["expected_value"] + sum(row["shap"] for row in case.shap["contributions"])
            self.assertAlmostEqual(reference, math.log(case.predicted_risk / (1 - case.predicted_risk)))
            probabilities.append(case.predicted_risk)
        self.assertLess(min(probabilities), 0.1)
        self.assertGreater(max(probabilities), 0.5)

    def test_strict_json(self):
        allowed = feature_keys(self.case)
        self.assertEqual(parse_response(self.raw, allowed).answer, self.good)
        for raw in ("{", "```json\n" + self.raw + "\n```", self.raw + self.raw,
                    '{"explanation":"one","explanation":"two","drivers":[]}',
                    '{"explanation":"x","drivers":[{"feature":"age","feature":"sodium","direction":"positive"}]}',
                    '{"explanation":NaN,"drivers":[]}', '{"explanation":Infinity,"drivers":[]}'):
            with self.subTest(raw=raw):
                result = parse_response(raw, allowed)
                self.assertEqual(result.error.code, "invalid_json")
                self.assertEqual(result.raw, raw)

    def test_schema_and_style(self):
        body = json.loads(self.raw)
        driver = body["drivers"][0]
        for invalid in (
            {"explanation": " "}, {"explanation": "word " * 91}, {"extra": True},
            {"drivers": {}}, {"drivers": [driver] * 4}, {"drivers": [driver, driver]},
            {"drivers": [{"feature": "weight", "direction": "positive"}]},
            {"drivers": [{"feature": "age", "direction": "increases"}]},
            {"drivers": [{"feature": []}]}, {"drivers": [{**driver, "extra": 0}]},
        ):
            with self.subTest(invalid=invalid):
                result = parse_response(json.dumps({**body, **invalid}), feature_keys(self.case))
                self.assertEqual(result.error.code, "invalid_schema")
        result = parse_response('{"explanation":"The drivers are uncertain.","drivers":[]}', feature_keys(self.case))
        self.assertEqual(result.warnings[0].code, "sentence_count")
        self.assertEqual(result.answer.drivers, ())

    def test_structure_ranking_sign_abstentions_and_ties(self):
        expected = expected_drivers(self.case)
        self.assertEqual([d.feature for d in expected], ["age", "prior_admissions", "creatinine"])
        for drivers in (expected, expected[1:], ()):
            result = verify_attribution(self.case, drivers)
            self.assertTrue(result.passed)
            self.assertEqual(result.expected, expected)
            self.assertEqual(result.observed, drivers)
        invalid = (expected[2], Driver("age", "positive"), Driver("sodium", "negative"),
                   Driver("sodium_missing", "positive"), Driver("bogus", "positive"), expected[2])
        result = verify_attribution(self.case, invalid)
        self.assertFalse(result.passed)
        self.assertEqual({issue.code for issue in result.issues}, {
            "driver_count", "ranking", "direction", "top_three", "negligible", "unknown_feature",
            "duplicate_feature",
        })
        self.assertEqual(expected_drivers(self.cases[4]), ())
        for feature in ("age", "sodium"):
            self.assertEqual(verify_attribution(self.cases[4], (Driver(feature, "positive"),)).issues[0].code,
                             "negligible")
        self.assertEqual([d.feature for d in expected_drivers(self.cases[7])],
                         ["age", "prior_admissions", "creatinine"])

    def test_revision_actions_and_raw_trace(self):
        infer = Mock()
        accepted = verify_and_revise(self.case, self.raw, infer)
        self.assertTrue(accepted["draft_accepted"])
        infer.assert_not_called()
        self.assertEqual(accepted["source"], "grounded_draft")
        bad = self.raw.replace("age decreases", "age increases")
        infer.return_value = self.raw
        revised = verify_and_revise(self.case, bad, infer)
        infer.assert_called_once()
        self.assertIn("driver_conflict", infer.call_args.args[0])
        self.assertEqual(revised["original_raw"], bad)
        self.assertTrue(revised["revision_requested"])
        self.assertTrue(revised["revision_accepted"])
        self.assertEqual(revised["source"], "model_revision")
        for mid, code in (("invalid", "invalid_json"), (TimeoutError("test"), "timeout")):
            with self.subTest(code=code):
                infer = Mock(side_effect=[mid])
                fallback = verify_and_revise(self.case, bad, infer)
                infer.assert_called_once()
                self.assertTrue(fallback["revision_rejected"])
                self.assertTrue(fallback["fallback_used"])
                self.assertEqual(fallback["fallback_reason"][0]["code"], code)
                self.assertEqual(fallback["answer"], asdict(self.good))
                self.assertTrue(fallback["fallback_validation"]["narrative"]["passed"])

    def test_smoke_and_private_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_smoke(directory)
            self.assertEqual(report["outputs"], dict.fromkeys("ABC", 8))
            self.assertEqual(report["valid_outputs"], {"A": 8, "B": 7, "C": 8})
            self.assertEqual(report["calls"], 21)
            self.assertEqual(report["accepted_originals"], 3)
            self.assertEqual(report["revisions_requested"], 5)
            self.assertEqual(report["revisions_accepted"], 2)
            self.assertEqual(report["fallbacks"], 3)
            self.assertEqual(report["parser_failures"], 1)
            self.assertEqual(report["narrative_failures"], 6)
            self.assertEqual(report["inference_failures"], 1)
            self.assertEqual(stat.S_IMODE(Path(report["checkpoint"]).stat().st_mode), 0o600)

    def test_interrupted_run_resumes_without_repeating_completed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            infer = Mock(side_effect=[self.raw, KeyboardInterrupt(), self.raw])
            with self.assertRaises(KeyboardInterrupt):
                run_experiment([self.case], infer, directory)
            state = run_experiment([self.case], infer, directory)
            self.assertEqual(state["calls"], 2)
            self.assertEqual(infer.call_count, 3)
            self.assertEqual(infer.call_args_list[1].args[0], infer.call_args_list[2].args[0])
            self.assertEqual([row["status"] for row in state["records"].values()], ["ok"] * 3)

    def test_saved_response_survives_interrupted_finalization(self):
        with tempfile.TemporaryDirectory() as directory:
            infer = Mock(return_value=self.raw)
            with patch("llm_runner.parse_response", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiment([self.case], infer, directory)
            state = run_experiment([self.case], infer, directory)
            self.assertEqual(infer.call_count, 2)
            self.assertEqual(state["calls"], 2)

    def test_saved_revision_survives_interrupted_recheck(self):
        import llm_runner

        with tempfile.TemporaryDirectory() as directory:
            bad = self.raw.replace("age decreases", "age increases")
            infer = Mock(side_effect=[self.raw, bad, self.raw])
            audit = llm_runner._audit

            def interrupt_revision(case, call):
                if call["raw"] == self.raw:
                    raise KeyboardInterrupt
                return audit(case, call)

            with patch("llm_runner._audit", side_effect=interrupt_revision):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiment([self.case], infer, directory)
            state = run_experiment([self.case], infer, directory)
            self.assertEqual(infer.call_count, 3)
            result = list(state["records"].values())[-1]["verification"]
            self.assertTrue(result["revision_accepted"])

    def test_transport_failure_is_recorded_and_c_has_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            infer = Mock(side_effect=[self.raw, ConnectionError("local endpoint down")])
            state = run_experiment([self.case], infer, directory)
            a, b, c = state["records"].values()
            self.assertEqual(a["status"], "ok")
            self.assertEqual(b["call"]["error"]["code"], "connection_failure")
            self.assertEqual(c["verification"]["source"], "deterministic_fallback")
            self.assertFalse(c["verification"]["revision_requested"])
            self.assertEqual(state, run_experiment([self.case], infer, directory))
            self.assertEqual(infer.call_count, 2)

    def test_runtime_budget_and_checkpoint_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            infer = Mock(return_value=self.raw)
            with patch("llm_runner.time.monotonic", side_effect=[0, 2]):
                paused = run_experiment([self.case], infer, directory, runtime_seconds=1)
            self.assertEqual(paused["pause_reason"], "runtime_limit")
            infer.assert_not_called()
            state = run_experiment([self.case], infer, directory)
            config = state["manifest"]["configuration"]
            self.assertEqual(set(state["records"]), {
                checkpoint_key(self.case.case_id, condition, state["manifest"]["prompt_version"], config)
                for condition in "ABC"
            })
            for kwargs in ({"prompt_version": "changed"}, {"request_timeout": 99},
                           {"model_config": ModelConfig(digest="new")}, {"model_config": ModelConfig(seed=1)}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Incompatible"):
                    run_experiment([self.case], infer, directory, **kwargs)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                run_experiment([replace(self.case, predicted_risk=0.7)], infer, directory)


class OllamaTests(unittest.TestCase):
    def test_endpoint_is_numeric_loopback(self):
        for endpoint in ("https://127.0.0.1:11434", "http://localhost:11434", "http://127.0.0.1",
                         "http://127.0.0.1:11434@example.org", "http://192.0.2.1:11434",
                         "http://127.0.0.1:11434/other"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                Ollama(ModelConfig(endpoint=endpoint))

    def test_discovery_and_generation_without_proxy_or_redirects(self):
        with patch("hrp_faithfulness.requests.Session") as factory:
            session = factory.return_value.__enter__.return_value
            response = session.request.return_value
            response.status_code = 200
            response.json.side_effect = [
                {"models": [{"name": "qwen3:8b", "digest": "sha256:fictional"}]},
                {"done": True, "response": "answer", "model": "qwen3:8b"},
            ]
            ollama = Ollama.discover()
            self.assertEqual(ollama.config.digest, "sha256:fictional")
            self.assertEqual(ollama("prompt", 25), "answer")
            self.assertFalse(session.trust_env)
            kwargs = session.request.call_args.kwargs
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(kwargs["timeout"], 25)
            self.assertEqual(kwargs["json"], {
                "model": "qwen3:8b", "prompt": "prompt", "format": "json", "think": False, "stream": False,
                "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
            })
            response.status_code = 302
            with self.assertRaises(InferenceError) as context:
                ollama("prompt", 25)
            self.assertEqual(context.exception.code, "invalid_ollama_response")

    def test_transport_errors_and_truncated_raw_text(self):
        ollama = Ollama()
        for raised, code in ((requests.ConnectionError("offline"), "connection_failure"),
                             (requests.Timeout("time"), "timeout"), (ValueError("bad envelope"), "invalid_ollama_response")):
            with self.subTest(code=code), patch("llm_runner.local_request", side_effect=raised):
                with self.assertRaises(InferenceError) as context:
                    ollama("prompt", 1)
                self.assertEqual(context.exception.code, code)
        with patch("llm_runner.local_request", return_value={"done": True, "done_reason": "length", "response": "{"}):
            with self.assertRaises(InferenceError) as context:
                ollama("prompt", 1)
            self.assertEqual(context.exception.code, "truncated_generation")
            self.assertEqual(context.exception.raw, "{")
