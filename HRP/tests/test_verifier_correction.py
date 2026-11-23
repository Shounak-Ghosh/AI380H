import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifier_challenge_set import base, load_challenge, response
from verifier_correction import HISTORICAL_ROOT, live_revision, scripted_corrections, select_live
from verifier_evaluation import evaluate


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_challenge()
        cls.scripted = scripted_corrections(cls.bundle)
        cls.by_name = {r["scenario"]: r for r in cls.scripted["rows"]}
        cls.plan = select_live(cls.bundle, evaluate(cls.bundle))

    def test_scripted_passes_and_fallbacks_have_independent_gold(self):
        good = self.by_name["fully_corrected"]
        self.assertTrue(good["trace"]["revision_accepted"] and good["revision_gold"] == "faithful")
        for name in ("partially_corrected", "unchanged", "new_error", "invalid_json", "timeout",
                     "faithful_ambiguous_revision", "correct_drivers_wrong_prose"):
            with self.subTest(name=name):
                row = self.by_name[name]
                self.assertTrue(row["trace"]["revision_rejected"] and row["trace"]["fallback_used"])
                self.assertTrue(row["final_mathematically_faithful"])
                self.assertEqual(row["requests"], 1)
        wrong = self.by_name["correct_drivers_wrong_prose"]
        self.assertIsNotNone(wrong["trace"]["revision"]["parse"]["answer"])
        self.assertEqual(wrong["revision_gold"], "unfaithful")
        self.assertEqual(self.by_name["timeout"]["trace"]["revision_call"]["error"]["code"], "timeout")
        self.assertIsNone(self.by_name["invalid_json"]["trace"]["revision"]["parse"]["answer"])
        self.assertEqual(self.by_name["original_accepted"]["requests"], 0)
        incomplete = self.by_name["accepted_omits_probability"]
        self.assertTrue(incomplete["trace"]["revision_accepted"])
        self.assertFalse(incomplete["final_complete"])

    def test_live_resume_cap_and_owner_only_permissions(self):
        inputs = {r["challenge_id"]: r for r in self.bundle["inputs"]}
        replies = []
        for row in self.plan[:3]:
            case = inputs[row["challenge_id"]]["case_id"]
            prose, drivers, _ = base(self.bundle["references"][case])
            replies.append(response(prose, drivers))
        adapter = Mock(side_effect=replies)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "live"
            first = live_revision(self.bundle, self.plan, directory, max_calls=1, adapter=adapter)
            second = live_revision(self.bundle, self.plan, directory, max_calls=3, adapter=adapter)
            with patch("llm_runner.Ollama.discover", side_effect=AssertionError):
                replay = live_revision(self.bundle, self.plan, directory, max_calls=3)
            self.assertEqual(replay, second)
            self.assertEqual(adapter.call_count, 3)
            self.assertEqual(first["rows"][0], second["rows"][0])
            self.assertEqual(second["summary"]["pending_gold"], 3)
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            path = Path(second["checkpoint"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "owner-only"):
                live_revision(self.bundle, self.plan, directory, max_calls=3, adapter=adapter)

    def test_interrupted_reserved_request_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            interrupted = Mock(side_effect=KeyboardInterrupt)
            with self.assertRaises(KeyboardInterrupt):
                live_revision(self.bundle, self.plan, tmp, max_calls=1, adapter=interrupted)
            untouched = Mock()
            state = live_revision(self.bundle, self.plan, tmp, max_calls=1, adapter=untouched)
            untouched.assert_not_called()
            self.assertEqual(state["summary"]["reserved_requests"], 1)
            self.assertEqual(state["rows"][0]["trace"]["revision_call"]["error"]["code"], "interrupted_request")

    def test_available_historical_checkpoints_unchanged(self):
        files = {
            "checkpoint.json": "42038ff9ceef439a3bb8797b419869ed7638610173c36e18f4a9631f122a047a",
            "llm-runs/experiment3-adversarial/qwen3-v1/checkpoint.json":
                "52ecf7c17a9cfd9108c264af7e8f0886a1754be23edfb261597deb1cf6052a1b",
        }
        if not all((HISTORICAL_ROOT / relative).is_file() for relative in files):
            self.skipTest("Historical private checkpoints are not available on this machine.")
        for relative, expected in files.items():
            self.assertEqual(hashlib.sha256((HISTORICAL_ROOT / relative).read_bytes()).hexdigest(), expected)
