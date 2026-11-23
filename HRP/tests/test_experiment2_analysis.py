from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment2_analysis import (
    analyze, inference_costs, notebook_run, paired_intervals, read_checkpoint, rejection_rows, save_analysis,
)
from experiment2_smoke import fictional_cases
from llm_runner import ModelConfig, run_experiment
from llm_verification import case_fingerprints, deterministic_answer


class AnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_pairs_within_case(self):
        rows = self.paired_fixture()
        first = paired_intervals(rows, seed=7, replicates=2000)
        self.assertEqual(first, paired_intervals(rows, seed=7, replicates=2000))
        a_b = self.signed(first, "complete_case")
        self.assertEqual((a_b["pairs"], a_b["excluded_pairs"]), (2, 1))
        self.assertEqual((a_b["mean_difference"], a_b["ci95_low"], a_b["ci95_high"]), (1, 1, 1))

    def test_failure_inclusive_differences_include_invalid_a_as_zero(self):
        rows = self.paired_fixture()
        a_b = self.signed(paired_intervals(rows), "failure_inclusive")
        self.assertEqual((a_b["pairs"], a_b["observed_valid_pairs"], a_b["excluded_pairs"]), (3, 2, 0))
        self.assertAlmostEqual(a_b["mean_difference"], (1 + 1 + 3) / 3)

    @staticmethod
    def signed(result, method):
        return next(row for row in result if row["comparison"] == "A→B" and row["analysis"] == method
                    and row["metric"] == "signed_top_three_agreement")

    @staticmethod
    def paired_fixture():
        return [{"case_id": str(i), "condition": condition,
                 "quality": {"signed_top_three_agreement": value} if value is not None else None}
                for i, case in enumerate(((0, 1, 3), (2, 3, 3), (None, 3, 3)))
                for condition, value in zip("ABC", case)]

    def test_rejection_categories_overlap_and_distinguish_missing_parse(self):
        original = {
            "status": "verification_failed", "narrative": {"claims": [
                {"code": "ambiguous_subject", "status": "ambiguous"},
                {"code": "ambiguous_subject", "status": "ambiguous"},
                {"code": "clinical", "status": "contradicted"},
            ]},
            "structured": {"issues": [{"code": "ranking"}], "missing": []},
            "missing_prose_drivers": ["age"],
        }
        traces = [{"original": original, "revision_rejected": True,
                   "revision": {"status": "parse_error", "narrative": None}}]
        counts = {(r["stage"], r["reason"]): (r["count"], r["percent"]) for r in rejection_rows(traces)}
        for reason in ("Unparsed or ambiguous wording", "Missing explicit directional prose",
                       "Unsupported clinical interpretation", "Ranking error"):
            self.assertEqual(counts["original B", reason], (1, 100))
        self.assertEqual(counts["original B", "Incorrect SHAP direction"], (0, 0))
        self.assertEqual(counts["rejected revision", "Other"], (1, 100))

    def test_cost_increments_and_full_delivery_do_not_double_count(self):
        def row(condition, field, seconds):
            return {"condition": condition, field: {"latency_seconds": seconds}}

        output = inference_costs([row("A", "call", 3), row("B", "call", 2),
                                  row("B", "call", 4), row("C", "revision_call", 1),
                                  {"condition": "C"}])
        costs = {r["stage"]: (r["calls"], r["seconds"], r["median_seconds"]) for r in output}
        self.assertEqual(costs["B"], (2, 6, 3))
        self.assertEqual(costs["C"], (1, 1, 1))
        self.assertEqual(costs["C delivery (B + C)"], (3, 7, 2))
        self.assertEqual(costs["Total unique"], (4, 10, 2.5))

    def test_review_and_derived_export_preserve_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path, cases = self.saved_run(root)
            payload = path.read_bytes()
            modified = path.stat().st_mtime_ns
            with patch("experiment2_inputs.load_explanation_cases", return_value=({"id": "fictional"}, cases)), \
                 patch("experiment2_analysis.Ollama.discover") as discover, \
                 patch("experiment2_analysis.run_experiment") as inference:
                run = notebook_run("review", checkpoint=path)
            discover.assert_not_called()
            inference.assert_not_called()
            analysis = analyze(run)
            directory = Path(root) / "derived"
            save_analysis(directory, analysis, run["checkpoint_sha256"], run["state"]["manifest"], {})
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_mtime_ns, modified)
            self.assertEqual(read_checkpoint(path)[1], run["checkpoint_sha256"])
            manifest = json.loads((directory / "experiment2_analysis_manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_sha256"], run["checkpoint_sha256"])
            with self.assertRaises(FileExistsError):
                save_analysis(directory, analysis, run["checkpoint_sha256"], run["state"]["manifest"], {})

    def test_review_with_missing_checkpoint_does_not_load_cases_or_discover(self):
        with tempfile.TemporaryDirectory() as root, \
             patch("experiment2_inputs.load_explanation_cases") as loader, \
             patch("experiment2_analysis.Ollama.discover") as discover:
            with self.assertRaises(FileNotFoundError):
                notebook_run("review", checkpoint=Path(root) / "missing.json")
            loader.assert_not_called()
            discover.assert_not_called()

    def test_real_requires_dedicated_directory(self):
        with tempfile.TemporaryDirectory() as root, patch("experiment2_analysis.Ollama.discover") as discover:
            checkpoint = Path(root) / "checkpoint.json"
            for directory in (None, root, "relative"):
                with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "dedicated"):
                    notebook_run("real", checkpoint=checkpoint, run_directory=directory)
            discover.assert_not_called()

    def test_completed_dedicated_real_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as root:
            path, cases = self.saved_run(root)
            legacy = Path(root) / "checkpoint.json"
            with patch("experiment2_inputs.load_explanation_cases", return_value=({"id": "fictional"}, cases)), \
                 patch("experiment2_analysis.Ollama.discover") as discover, \
                 patch("experiment2_analysis.run_experiment") as inference:
                run = notebook_run("real", checkpoint=legacy, run_directory=path.parent)
                reviewed = notebook_run("review", checkpoint=path)
            discover.assert_not_called()
            inference.assert_not_called()
            self.assertEqual(run["score"], reviewed["score"])

    @staticmethod
    def saved_run(root):
        case = fictional_cases()[0]
        directory = Path(root) / "dedicated"
        raw = json.dumps(asdict(deterministic_answer(case)))
        state = run_experiment([case], Mock(return_value=raw), directory,
                               model_config=ModelConfig(digest="fixture"), provenance={"id": "fictional"})
        cases = tuple(replace(case, case_id=f"fixture-{i}") for i in range(72))
        state["manifest"]["cases"] = case_fingerprints(cases)
        state["manifest"]["configuration"]["inference"] = "ollama"
        records = state["records"].values()
        state["records"] = {f"{c.case_id}-{r['condition']}": {**r, "case_id": c.case_id}
                            for c in cases for r in records}
        state["calls"] *= len(cases)
        path = directory / "checkpoint.json"
        path.write_text(json.dumps(state))
        return path, cases
