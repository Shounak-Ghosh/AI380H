import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import verifier_evaluation as ev
from verifier_challenge_set import load_challenge


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_challenge()
        cls.predictions = ev.evaluate(cls.bundle)
        cls.rows = ev.join_results(cls.bundle, cls.predictions)
        cls.analysis = ev.analyze(cls.bundle, cls.predictions, replicates=200)

    def test_confusion_definitions_and_zero_division(self):
        rows = [dict(base_reference_id=str(i), gold=g, pred=p)
                for i, (g, p) in enumerate(((True, True), (True, False), (False, True), (False, False)))]
        result = ev.binary_metrics(rows, lambda r: r["gold"], lambda r: r["pred"])
        self.assertEqual(result["confusion"], dict(TP=1, FP=1, TN=1, FN=1))
        self.assertEqual(result["metrics"]["recall"]["estimate"], .5)
        empty = ev.binary_metrics([], bool, bool)
        self.assertIsNone(empty["metrics"]["precision"]["estimate"])
        self.assertIsNone(empty["metrics"]["balanced_accuracy"]["ci95_low"])

    def test_ambiguity_schema_and_eligible_denominators(self):
        faithful = next(r for r in self.rows if r["group"] == "A" and r["outcome"] == ev.AMBIGUOUS)
        wrong = next(r for r in self.rows if r["group"] == "B" and r["outcome"] == ev.AMBIGUOUS)
        for r in (faithful, wrong):
            self.assertFalse(r["explicit_error"])
            self.assertFalse(r["release_accepted"])
        self.assertEqual(len(ev.primary_cohort(self.rows)), 120)
        eligible = ev.primary_cohort(self.rows, True)
        self.assertEqual(len(eligible), 148)
        self.assertTrue(all(r["faithfulness"] != "indeterminate" and r["outcome"] != ev.SCHEMA for r in eligible))
        malformed = [r for r in self.rows if r["outcome"] == ev.SCHEMA]
        self.assertEqual(len(malformed), 24)
        self.assertTrue(all(not r["explicit_error"] for r in malformed))

    def test_error_rows_count_examples_once(self):
        for row in self.analysis["error_matrix"]:
            self.assertEqual(sum(v["numerator"] for v in row.values() if isinstance(v, dict)), row["n"])
        direction = next(row for row in self.analysis["error_matrix"] if row["category"] == "incorrect_feature_direction")
        self.assertEqual(direction["n"], sum("incorrect_feature_direction" in row["errors"]
                                            for row in self.rows if row["group"] == "B"))

    def test_cluster_bootstrap_and_pair_alignment(self):
        records = [("p1", (1, 1)), ("p1", (0, 1)), ("p2", (1, 1)), ("p2", (0, 1))]
        fn = lambda v: (v[0], v[1])
        stat = ev.bootstrap(records, fn)
        self.assertEqual(stat, ev.bootstrap(list(reversed(records)), fn))
        self.assertEqual((stat["ci95_low"], stat["ci95_high"], stat["clusters"]), (.5, .5, 2))
        by_id = {r["challenge_id"]: r for r in self.rows}
        self.assertEqual(sum(len(r["pairs"]) for r in self.analysis["paraphrases"]), 48)
        for style in self.analysis["paraphrases"]:
            for pair in style["pairs"]:
                self.assertEqual(by_id[pair["control_id"]]["case_id"], by_id[pair["paraphrase_id"]]["case_id"])

    def test_review_is_offline_and_checks_frozen_hashes(self):
        with patch("llm_runner.Ollama.discover", side_effect=AssertionError):
            result = ev.notebook_run("review")
        self.assertEqual(result["manifest"]["frozen_verifier"], ev.frozen_verifier())
        self.assertEqual(result["analysis"]["primary"]["explicit"]["n"], 120)
        with patch.object(ev, "VERIFIER_VERSION", "modified"):
            with self.assertRaisesRegex(ValueError, "frozen"):
                ev.frozen_verifier()
