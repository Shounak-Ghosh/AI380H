from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_runner import _fictional_manifest, _provenance, notebook_run, scripted_inference
from adversarial_scenarios import build_manifest, explanation_cases
from llm_runner import ModelConfig, run_experiment
from llm_verification import build_prompt


class AdversarialRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = _fictional_manifest(build_manifest())
        cls.cases = explanation_cases(cls.manifest)

    def test_prompt_isolation(self):
        for case in self.cases:
            a, b = [build_prompt(case, c) for c in "AB"]
            for prompt in (a, b):
                for forbidden in ("intuitive", "counterintuitive", "adversarial", "reversal", "coefficients", case.case_id):
                    self.assertNotIn(forbidden, prompt)
            a, b = [json.loads(p.split("Input JSON:\n")[1]) for p in (a, b)]
            self.assertEqual(set(a), {"features", "predicted_risk"})
            self.assertEqual(a, {k: b[k] for k in a})
            self.assertEqual(b["shap"]["expected_value"], case.shap["expected_value"])
            for r in b["shap"]["contributions"]:
                self.assertIn(r, [{k: row[k] for k in r} for row in case.shap["contributions"]])

    def test_resume_and_offline_review(self):
        with TemporaryDirectory() as directory, patch("adversarial_runner.Ollama.discover") as discover:
            paused = notebook_run("fictional", directory=directory, max_calls=5, replicates=20)
            self.assertEqual(paused["state"]["calls"], 5)
            self.assertEqual(paused["state"]["pause_reason"], "call_limit")
            self.assertEqual(notebook_run(directory=directory)["mode"], "setup")
            done = notebook_run("fictional", directory=directory, replicates=20)
            self.assertEqual(done["state"]["calls"], 10)
            paths = [Path(directory) / name for name in ("checkpoint.json", "scenario_manifest.json")]
            before = [sha256(path.read_bytes()).hexdigest() for path in paths]
            with patch("adversarial_runner.build_manifest") as build, patch("adversarial_runner.Ollama._request") as request:
                reviewed = notebook_run(directory=directory, replicates=20)
                build.assert_not_called()
                request.assert_not_called()
            self.assertEqual(reviewed["state"], done["state"])
            self.assertEqual(reviewed["result"], done["result"])
            self.assertEqual(reviewed["checkpoint_sha256"], before[0])
            self.assertEqual([sha256(path.read_bytes()).hexdigest() for path in paths], before)
            self.assertEqual(notebook_run("fictional", directory=directory, replicates=20)["state"], done["state"])
            records = {(r["case_id"], r["condition"]): r for r in done["state"]["records"].values()}
            for case in self.cases:
                self.assertEqual(records[case.case_id, "B"]["call"]["raw"],
                                 records[case.case_id, "C"]["verification"]["original_raw"])
            self.assertEqual(done["result"]["operations"]["intervention_counts"]["fallback_used"], 1)
            discover.assert_not_called()

    def test_checkpoint_incompatibility_and_inference_failure(self):
        with TemporaryDirectory() as directory:
            provenance = _provenance(self.manifest, "fictional")
            infer = scripted_inference(self.manifest)
            calls = 0

            def one_timeout(prompt, timeout):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise TimeoutError("scripted")
                return infer(prompt, timeout)

            state = run_experiment(self.cases, one_timeout, directory, provenance=provenance)
            self.assertEqual(calls, state["calls"])
            self.assertEqual(sum(r["status"] == "inference_error" for r in state["records"].values()), 1)
            for changes in ({"prompt_version": "changed"}, {"provenance": {**provenance, "runner_version": "changed"}},
                            {"provenance": {**provenance, "scenario_manifest_sha256": "changed"}},
                            {"model_config": ModelConfig(digest="changed")}):
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    run_experiment(self.cases, one_timeout, directory, **({"provenance": provenance} | changes))
            self.assertEqual(calls, state["calls"])
