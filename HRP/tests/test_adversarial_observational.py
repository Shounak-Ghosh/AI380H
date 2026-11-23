from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_observational import screen_saved_cases
from experiment2_smoke import fictional_cases


class ObservationalScreenTests(unittest.TestCase):
    def test_emits_only_aggregate_counts(self):
        cases = fictional_cases()
        with patch("adversarial_observational.load_explanation_cases",
                   return_value=({"model_family": "xgboost", "model_id": "fictional-model"}, cases)):
            result = screen_saved_cases()
        self.assertEqual(result["screened"], len(cases))
        self.assertGreater(result["flags"]["mixed_leading_contributions"], 0)
        for case in cases:
            self.assertNotIn(case.case_id, str(result))
