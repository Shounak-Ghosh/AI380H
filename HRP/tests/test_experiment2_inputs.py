import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment2_inputs import MODEL_RUN_ID, _prepare_case, load_explanation_cases


class ExplanationInputTests(unittest.TestCase):
    def test_baseline_and_grounded_share_the_same_features_and_risk(self):
        row = {"age": 60, "sodium": float("nan"), "predicted_risk": 0.15,
               "expected_value": -2.4, "shap_output_scale": "raw_log_odds",
               "subject_id": 123, "hadm_id": 456, "outcome": 1}
        rows = pd.DataFrame([
            {"feature": "sodium", "shap": -0.1, "direction": "negative", "rank": 2},
            {"feature": "age", "shap": 0.3, "direction": "positive", "rank": 1},
        ])
        metadata = {"age": {"label": "Age", "unit": "years"},
                    "sodium": {"label": "Sodium", "unit": "mEq/L"}}
        case = _prepare_case(1, row, rows, metadata)
        self.assertEqual(case.case_id, "case-001")
        baseline, grounded = case.inputs("A"), case.inputs("B")
        self.assertEqual(grounded, case.inputs("C"))
        self.assertEqual(set(baseline), {"features", "predicted_risk"})
        self.assertEqual(baseline, {key: grounded[key] for key in baseline})
        self.assertIsNone(baseline["features"][1]["value"])
        self.assertEqual([entry["feature"] for entry in grounded["shap"]["contributions"]],
                         ["age", "sodium"])
        for name in ("subject_id", "hadm_id", "outcome"):
            self.assertNotIn(name, json.dumps(grounded, allow_nan=False))
        with self.assertRaises(ValueError):
            case.inputs("other")

    def test_loader_pins_directory_and_rejects_a_different_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("experiment2_inputs.verify_run", return_value={}) as verify, \
                    patch("experiment2_inputs.sha256", return_value="different"):
                with self.assertRaisesRegex(ValueError, "pinned"):
                    load_explanation_cases(root)
                verify.assert_called_once_with(Path(root) / MODEL_RUN_ID)


if __name__ == "__main__":
    unittest.main()
