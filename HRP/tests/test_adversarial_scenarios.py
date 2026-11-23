import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_scenarios import (
    FAMILIES, NAMES, TOLERANCE, ScenarioConfig, attribution_attributes, build_manifest, validate_manifest,
)
from llm_verification import fingerprint


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest()

    def test_reproducibility_and_counts(self):
        manifest = self.manifest
        self.assertEqual(fingerprint(manifest), fingerprint(build_manifest()))
        self.assertEqual(len(manifest["records"]), 96)
        self.assertEqual(len({row["case_id"] for row in manifest["records"]}), 96)
        for family in FAMILIES:
            rows = [r for r in manifest["records"] if r["family"] == family]
            self.assertEqual(len({r["profile_id"] for r in rows}), 24)
            self.assertEqual({r["risk_band"] for r in rows}, {"low", "high"})
            for name in ("mixed_direction", "conflicting_importance"):
                self.assertGreater(sum(r["attributes"][name] for r in rows), 0)

    def test_local_reversals_and_additivity(self):
        validate_manifest(json.loads(json.dumps(self.manifest, sort_keys=True, allow_nan=False)))
        records = self.manifest["records"]
        for left, right in zip(records[::2], records[1::2], strict=True):
            self.assertEqual(left["features"], right["features"])
            self.assertEqual([f["name"] for f in left["features"]], list(NAMES))
            for a, b in zip(left["shap"]["contributions"], right["shap"]["contributions"], strict=True):
                target = a["feature"] in left["targeted_features"]
                self.assertAlmostEqual(a["shap"], -b["shap"] if target else b["shap"])
                if target:
                    self.assertGreaterEqual(a["shap"], self.manifest["configuration"]["reversal_min"])
            self.assertGreater(left["predicted_risk"], right["predicted_risk"])
        self.assertLess(max(m["max_errors"]["additivity"] for m in self.manifest["models"].values()), TOLERANCE)

    def test_invalid_references_are_rejected(self):
        for field, value in (("predicted_risk", float("nan")), ("margin", 10), ("risk_band", "invalid")):
            manifest = copy.deepcopy(self.manifest)
            manifest["records"][0][field] = value
            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_overlapping_attributes(self):
        attrs = attribution_attributes([.15, .5, -.4, -.14, .13, .6], "age", ScenarioConfig())
        self.assertTrue(attrs["conflicting_importance"])
        self.assertTrue(attrs["mixed_direction"])
        self.assertEqual(attrs["leading_positive"], ["heart_rate", "prior_admissions"])
        self.assertEqual(attrs["leading_negative"], ["creatinine"])
