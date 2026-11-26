"""Offline regression tests for provenance, synthesis and public projections."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import results_synthesis as synthesis
from results_synthesis_figures import fictional_figure, figure_bytes


class SourceValidationTests(unittest.TestCase):
    def test_indexed_artifact_checksum_and_missing_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "aggregate.json"
            artifact.write_text('{"complete": true}')
            index = {artifact.name: synthesis.digest(artifact)}
            synthesis.indexed_files(root, index, (artifact.name,))
            artifact.write_text("tampered")
            with self.assertRaisesRegex(synthesis.ValidationError, "Checksum mismatch"):
                synthesis.indexed_files(root, index)
            artifact.unlink()
            with self.assertRaisesRegex(synthesis.ValidationError, "Missing indexed artifact"):
                synthesis.indexed_files(root, index)

    def test_index_rejects_traversal_and_missing_required_entry(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthesis.ValidationError, "artifact name"):
                synthesis.indexed_files(Path(directory), {"../aggregate.json": "a" * 64})
            with self.assertRaisesRegex(synthesis.ValidationError, "missing required"):
                synthesis.indexed_files(Path(directory), {"index.json": "a" * 64}, ("aggregate.json",))

    def test_human_snapshot_loads_and_rejects_bad_version(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "results-v1"
            snapshot.mkdir(mode=0o700)
            fixture = {"explanations": 18, "clusters": 6}
            files = {"aggregate.json": json.dumps(fixture), "01_faithfulness.png": "fig",
                     "02_text_ratings.png": "fig", "04_matched_changes.png": "fig"}
            index = {}
            for name, value in files.items():
                path = snapshot / name
                path.write_text(value)
                path.chmod(0o600)
                index[name] = synthesis.digest(path)
            (snapshot / "index.json").write_text(json.dumps(index))
            (snapshot / "index.json").chmod(0o600)
            result, loaded, _ = synthesis._human_snapshot(root, "results-v1")
            self.assertEqual((result, loaded), (fixture, index))
            with self.assertRaisesRegex(synthesis.ValidationError, "versioned"):
                synthesis._human_snapshot(root, "../results-v1")
            (snapshot / "aggregate.json").write_text("changed")
            with self.assertRaisesRegex(synthesis.ValidationError, "Checksum mismatch"):
                synthesis._human_snapshot(root, "results-v1")

    def test_missing_sources_are_separate_and_paths_are_redacted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            options = synthesis.Sources(cohort_root=root, model_root=root, experiment3_run=root,
                                        human_directory=root)
            with patch.object(synthesis, "_experiment4", side_effect=synthesis.ValidationError("sentinel")):
                study = synthesis.run(sources=options)
            self.assertFalse(study.validated)
            self.assertEqual([row["experiment"] for row in study.provenance], list(range(1, 6)))
            self.assertEqual(study.provenance[0]["status"], "unavailable")
            self.assertIn("pinned model/cohort", study.provenance[1]["diagnostic"])
            self.assertNotIn(directory, json.dumps(study.provenance))
            self.assertEqual(study.provenance[3]["diagnostic"], "sentinel")

    def test_live_summary_validation_preserves_delivery_denominators(self):
        from verifier_live_annotation import VERSION, aggregate

        records = [
            dict(stratum=stratum, faithfulness="faithful" if index != 9 else "unfaithful",
                 validity="valid", completeness="complete", accepted=index < 5, fallback=index >= 5,
                 ambiguous=index >= 5, explicit=False, issues=[], errors=[])
            for stratum in ("math_error", "faithful_prose", "incomplete") for index in range(10)
        ]
        manifest = {"challenge_sha256": "a" * 64, "files": {"live_selection.json": "b" * 64},
                    "frozen_verifier": {"version": "scripted", "code_sha256": "c" * 64}}
        provenance = dict(challenge_sha256="a" * 64, live_selection_sha256="b" * 64,
                          frozen_verifier_version="scripted", frozen_verifier_code_sha256="c" * 64,
                          live_checkpoint_sha256="d" * 64, annotation_sha256="e" * 64,
                          crosswalk_sha256="f" * 64, reserved_requests=30)
        summary = aggregate(records, 1, provenance)
        result = synthesis.validate_live_summary(summary, manifest)
        self.assertEqual(result["version"], VERSION)
        self.assertEqual(result["rates"]["false_rejection_faithful_complete_valid"]["denominator"], 27)
        self.assertEqual(result["final_delivery"], {"model_revision": 15, "deterministic_fallback": 15})
        summary["rates"]["fallback"]["numerator"] = 14
        with self.assertRaisesRegex(synthesis.ValidationError, "denominators"):
            synthesis.validate_live_summary(summary, manifest)

    def test_no_inference_or_fit_when_fixture_or_review_is_requested(self):
        from llm_runner import Ollama
        from model_explanations import Attribution
        from run_models import Experiment

        with (patch.object(Ollama, "discover", side_effect=AssertionError("Ollama called")),
              patch.object(Attribution, "__init__", side_effect=AssertionError("SHAP recalculated")),
              patch.object(Experiment, "__init__", side_effect=AssertionError("Model fitting invoked")),
              TemporaryDirectory() as directory):
            root = Path(directory)
            study = synthesis.run("fictional", output_root=root)
            self.assertTrue(study.fictional)
            self.assertFalse(study.validated)
            self.assertEqual(study.data[5]["explanations"], 18)
            with patch.object(synthesis, "load_sources", return_value=synthesis.Study()) as loader:
                synthesis.run(output_root=root)
                loader.assert_called_once()
            self.assertEqual(list(root.iterdir()), [])


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.study = synthesis.fictional_study()
        self.result = self.study.data[5]

    def test_human_counts_and_matched_differences(self):
        for row in synthesis.human_condition_rows(self.result):
            self.assertEqual(row["selected"], 6)
            self.assertEqual(row["faithful"] + row["unfaithful"], row["determinate"])
        paired = self.result["paired"]["C − B"][synthesis.READ]
        distribution = paired["change_distribution"]
        expected = sum(int(change) * n for change, n in distribution.items()) / paired["eligible_n"]
        self.assertEqual(paired["mean_change"], expected)
        self.assertEqual(paired["matched_n"], 6)
        corrupted = deepcopy(self.result)
        corrupted["paired"]["C − B"][synthesis.READ]["mean_change_ci95"] = [5, 6]
        with self.assertRaisesRegex(synthesis.ValidationError, "interval"):
            synthesis.validate_human_summary(corrupted, 6)

    def test_hypothesis_mapping_uses_primary_endpoint_and_right_minus_left(self):
        primary = {"comparison": "A→B", "analysis": "complete_case", "metric": synthesis.SIGNED,
                   "mean_difference": 1.2, "pairs": 6, "ci95_low": .2, "ci95_high": 2.1}
        verified = dict(primary, comparison="B→C", mean_difference=.1, ci95_low=-.1, ci95_high=.3)
        other_endpoint = dict(primary, metric="top_three_overlap", mean_difference=9)
        study = synthesis.Study(data={2: {"paired": [other_endpoint, primary, verified],
                                         "interventions": [{"outcome": "verified_complete", "count": 6}]}})
        h1, h2 = synthesis.hypothesis_rows(study)
        self.assertEqual((h1["hypothesis"], h1["result"], h1["ci95"], h1["assessment"]),
                         ("H1", "+1.200 (6 pairs)", "[+0.200, +2.100]", "Supported"))
        self.assertEqual((h2["comparison"], h2["assessment"]), ("B→C", "Inconclusive"))
        self.assertEqual(synthesis.direction_status({"mean_change_ci95": [-.8, -.2]}), "Unsupported")

    def test_public_human_rows_have_no_differencing_inputs(self):
        self.result["condition_c_by_final_source"]["model_revision"]["private_notes"] = "REDACT-ME"
        self.result["by_condition"]["A"]["subject_id"] = 12345
        rows = synthesis.coarsened_human_rows(self.result)
        self.assertEqual(set(rows[0]), {"condition", "selected", "faithful_share_band",
                                        "clarity_rounded_half_point", "readability_rounded_half_point"})
        self.assertNotIn("REDACT-ME", json.dumps(rows))
        self.assertNotIn("12345", json.dumps(rows))
        self.assertEqual(synthesis.rate_band(1), "80–100%")
        self.assertEqual(synthesis.rate_band(0), "0–20%")

    def test_automated_agreement_preserves_ambiguous_and_nonapplicable(self):
        rows = synthesis.human_verifier_rows(self.result)
        release_a = next(row for row in rows if row["condition"] == "A" and row["check"] == "release")
        self.assertEqual((release_a["concordant"], release_a["outside_binary"]), (0, 6))
        self.assertNotIn("kappa", json.dumps(rows))
        self.assertEqual(sum(release_a["labels"].values()), 6)

    def test_repeat_progress_never_invents_agreement(self):
        result, _ = __import__("human_explanation_evaluation").fictional_results()
        self.assertEqual(result["repeat_consistency"]["target"], 12)
        from human_evaluation_analysis import repeat_consistency
        pending = repeat_consistency([], [])
        self.assertEqual(pending["completed"], 0)
        self.assertIsNone(pending[synthesis.FAITH]["exact_agreement_percent"])
        self.assertIn("pending", synthesis.PENDING)

    def test_fictional_figure_is_deterministic(self):
        a, b = (figure_bytes(fictional_figure(self.study)) for _ in range(2))
        self.assertTrue(a.startswith(b"\x89PNG"))
        self.assertEqual(a, b)

    def test_detection_and_release_remain_separate(self):
        def row(tp, fp, tn, fn):
            return {"n": tp + fp + tn + fn, "confusion": dict(TP=tp, FP=fp, TN=tn, FN=fn)}
        data = {"primary": {"explicit": row(3, 0, 4, 1), "release": row(4, 2, 2, 0)},
                "all_eligible_sensitivity": {"release": row(5, 3, 3, 0)}}
        rows = synthesis.verifier_rows(data)
        self.assertEqual([(r["decision"], r["n"], r["FP"]) for r in rows],
                         [("explicit", 8, 0), ("release", 8, 2), ("release", 11, 3)])


class ExportTests(unittest.TestCase):
    def test_immutable_export_output_integrity_and_no_private_fields(self):
        study = synthesis.Study(data=dict.fromkeys(synthesis.UNITS, {}), provenance=[
            synthesis.source(i, "fixture-v1", "fixture-analysis-v1", i, i, {"logical.json": "a" * 64})
            for i in synthesis.UNITS
        ])
        study.data[5] = {"patient_id": "private-id", "raw_explanation": "private text"}
        tables = {"07_hypothesis_results": [{"hypothesis": "H5a", "assessment": "Inconclusive"}]}

        def figures(_, __, directory):
            (directory / "fixture.png").write_bytes(b"fixture")
            return {"fixture": "Fictional unit test."}

        with (TemporaryDirectory() as directory, patch.object(synthesis, "report_tables", return_value=tables),
              patch("results_synthesis_figures.public_figures", side_effect=figures)):
            root = Path(directory)
            target = synthesis.export_study(study, root)
            before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            synthesis.validate_export(target)
            with self.assertRaises(FileExistsError):
                synthesis.export_study(study, root)
            after = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertNotIn(b"private", before[Path("master_results.json")])
            (target / "master_results.json").write_text("tampered")
            with self.assertRaisesRegex(synthesis.ValidationError, "checksum"):
                synthesis.validate_export(target)

    def test_partial_or_fictional_source_cannot_export(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(synthesis.ValidationError, "all five"):
                synthesis.export_study(synthesis.fictional_study(), directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
