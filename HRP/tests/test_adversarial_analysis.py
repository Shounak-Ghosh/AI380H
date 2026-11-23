from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_analysis import feature_claim, paired_intervals, predicted_risk_sensitivity
from adversarial_runner import notebook_run
from adversarial_scenarios import explanation_cases
from llm_verification import Answer, Driver
from narrative_verifier import verify_narrative


class AdversarialAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with TemporaryDirectory() as directory:
            cls.experiment = notebook_run("fictional", directory=directory, replicates=100)

    def test_expected_matched_contrasts_and_outcomes(self):
        result = self.experiment["result"]
        contrasts = [r for r in result["paired"] if r["stratum"] == "all"
                     and r["metric"] == "signed_top_three_agreement" and r["policy"] == "valid_pairs"]
        self.assertEqual([r["pairs"] for r in contrasts], [2, 2, 2])
        self.assertEqual([r["mean_difference"] for r in contrasts], [0, -1, 0])
        self.assertEqual(sum(r["outcome"] == "clinical_expectation_substitution"
                             for r in result["feature_claims"] if r["condition"] == "B"), 2)
        self.assertEqual(sum(r["outcome"] == "retained_intuitive_direction"
                             for r in result["switches"] if r["condition"] == "B"), 2)
        self.assertEqual(sum(r["outcome"] == "correct_switch"
                             for r in result["switches"] if r["condition"] == "C"), 4)

    def test_failed_output_in_sensitivity_not_complete_case(self):
        rows = [dict(r) for r in self.experiment["result"]["case_metrics"]]
        row = next(r for r in rows if r["condition"] == "B" and r["variant"] == "counterintuitive")
        row.update(status="parse_error", signed_top_three_agreement=None, exact_top_three_ordering=None)
        intervals = paired_intervals(rows, replicates=200)
        self.assertEqual(intervals, paired_intervals(rows[::-1], replicates=200))
        selected = [r for r in intervals if r["stratum"] == "all" and r["condition"] == "B"
                    and r["metric"] == "signed_top_three_agreement"]
        self.assertEqual([r["pairs"] for r in selected], [1, 2])
        self.assertEqual([r["mean_difference"] for r in selected], [-1, -2])

    def test_ambiguous_prose_is_unverifiable(self):
        manifest = self.experiment["manifest"]
        case = explanation_cases(manifest)[1]
        feature = next(r for r in case.shap["contributions"] if r["direction"] == "negative"
                       and r["feature"] in manifest["records"][1]["targeted_features"])
        answer = Answer(f"{feature['label']}, which raises predicted risk, may be relevant.",
                        (Driver(feature["feature"], "negative"),))
        narrative = verify_narrative(case, answer)
        claim = feature_claim(case, answer, narrative, feature, "positive", "ok")
        self.assertEqual(claim["outcome"], "unverifiable")
        self.assertTrue(claim["ambiguous_feature_prose"])
        self.assertEqual(claim["structured_outcome"], "model_faithful")

    def test_risk_sensitivity_preserves_pairs_and_missing_scores(self):
        result = self.experiment["result"]
        summary = predicted_risk_sensitivity(result)
        self.assertEqual(summary, predicted_risk_sensitivity({
            "case_metrics": result["case_metrics"][::-1],
            "prediction_changes": result["prediction_changes"][::-1],
        }))
        for row in summary["pairs"]:
            self.assertLess(row["probability_change"], 0)
            self.assertEqual(row["absolute_probability_change"], -row["probability_change"])
            self.assertEqual((row["A_difference"], row["B_difference"]), (0, -1))
            self.assertEqual(set(row["mixed_direction"]), {"intuitive", "counterintuitive"})

        scores = [dict(row) for row in result["case_metrics"]]
        failed = next(row for row in scores if row["condition"] == "A" and row["variant"] == "counterintuitive")
        failed.update(status="parse_error")
        summary = predicted_risk_sensitivity({**result, "case_metrics": scores})
        pair = next(row for row in summary["pairs"] if row["profile_id"] == failed["profile_id"])
        self.assertIsNone(pair["A_difference"])
        self.assertEqual(pair["B_difference"], -1)
        self.assertEqual(sum(r["complete_pairs"] for r in summary["strata"] if r["condition"] == "A"), 1)

    def test_risk_sensitivity_band_boundaries_and_empty_strata(self):
        result = self.experiment["result"]
        first, second = result["prediction_changes"]
        prediction_changes = [
            dict(first, intuitive_probability=.15, counterintuitive_probability=0.),
            dict(second, intuitive_probability=.25, counterintuitive_probability=0.),
        ]
        summary = predicted_risk_sensitivity({**result, "prediction_changes": prediction_changes})
        bands = {row["profile_id"]: row["band"] for row in summary["pairs"]}
        self.assertEqual(bands, {first["profile_id"]: "15–<25 pp", second["profile_id"]: "≥25 pp"})
        for row in summary["strata"]:
            self.assertIsNone(row["paired_difference_sd"])
            if row["band"] == "<15 pp":
                self.assertEqual(row["complete_pairs"], 0)
                self.assertIsNone(row["mean_difference"])

    def test_bootstrap_resamples_whole_profiles(self):
        rows = [dict(row) for row in self.experiment["result"]["case_metrics"]]
        profiles = sorted({row["profile_id"] for row in rows})
        for row in rows:
            if row["condition"] == "A":
                baseline = 0 if row["profile_id"] == profiles[0] else 2
                row["signed_top_three_agreement"] = baseline + (row["variant"] == "counterintuitive")
        replay = paired_intervals(rows, seed=380, replicates=500)
        self.assertEqual(replay, paired_intervals(rows[::-1], seed=380, replicates=500))
        pooled = next(row for row in replay if row["condition"] == "A" and row["stratum"] == "all"
                      and row["metric"] == "signed_top_three_agreement" and row["policy"] == "valid_pairs")
        self.assertEqual([pooled[k] for k in ("mean_difference", "ci95_low", "ci95_high")], [1, 1, 1])
