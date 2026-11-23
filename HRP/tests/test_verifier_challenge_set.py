import copy
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifier_challenge_set import (blinded_worksheet, build_challenge, checksum, load_challenge,
                                    validate_challenge, values)


class ChallengeSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_challenge()

    def test_stable_frozen_design_and_gold_counts(self):
        self.assertEqual(checksum(self.bundle), checksum(build_challenge()))
        counts = validate_challenge(self.bundle)
        self.assertEqual(counts["groups"], {"A": 60, "B": 60, "C": 60})
        self.assertEqual(counts["gold_counts"], {"faithful": 104, "unfaithful": 68, "indeterminate": 8})
        self.assertEqual(self.bundle["inputs"][0]["challenge_id"], "exp4-A-01-01")
        self.assertEqual(self.bundle["inputs"][-1]["challenge_id"], "exp4-C-12-05")

    def test_reference_pairs_ties_missingness_and_zero(self):
        refs = list(self.bundle["references"].values())
        for first, second in zip(refs[::2], refs[1::2], strict=True):
            self.assertEqual(first["base_reference_id"], second["base_reference_id"])
            self.assertEqual(first["case"]["features"], second["case"]["features"])
            self.assertEqual(values(first), {k: -v for k, v in values(second).items()})
        self.assertTrue(any(len({abs(v) for v in list(values(r).values())[:3]}) == 2 for r in refs))
        self.assertTrue(any(f["value"] is None for r in refs for f in r["case"]["features"]))
        self.assertLess(max(validate_challenge(self.bundle)["max_residuals"].values()), 1e-12)

    def test_single_mutations_and_paraphrase_equivalence(self):
        by_id = {g["challenge_id"]: g for g in self.bundle["labels"]}
        for gold in by_id.values():
            if gold["group"] == "B":
                expected = 2 if gold["mutation"]["recipe"] == "both_layers" else 1
                self.assertEqual(sum(c["status"] == "unfaithful" for c in gold["claims"]), expected)
            if gold["group"] == "A":
                direct = by_id[gold["direct_control_id"]]
                self.assertEqual(gold["structured_drivers"], direct["structured_drivers"])
                actual = {c["feature"]: c["claimed"] for c in gold["claims"] if c["kind"] == "direction"}
                expected = {c["feature"]: c["claimed"] for c in direct["claims"] if c["kind"] == "direction"}
                self.assertEqual(actual, expected)

    def test_gold_generation_is_independent_and_blinded(self):
        with patch("llm_verification.parse_response", side_effect=AssertionError), patch(
                "narrative_verifier.verify_narrative", side_effect=AssertionError):
            self.assertEqual(checksum(build_challenge()), checksum(self.bundle))
        self.assertTrue(all(set(i) == {"challenge_id", "case_id", "raw"} for i in self.bundle["inputs"]))
        worksheet, key = blinded_worksheet(self.bundle)
        self.assertEqual(len(worksheet), 30)
        self.assertEqual(Counter(uid.split("-")[1] for uid in key.values()), {"A": 10, "B": 10, "C": 10})
        self.assertTrue(all("faithfulness" not in row and "challenge_id" not in row for row in worksheet))

    def test_modified_direction_gold_fails_validation(self):
        bundle = copy.deepcopy(self.bundle)
        gold = next(c for c in bundle["labels"][0]["claims"] if c["kind"] == "direction")
        gold["reference"] = "impossible"
        with self.assertRaisesRegex(ValueError, "direction"):
            validate_challenge(bundle)
