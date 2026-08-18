from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from scripts.audio_diagnosis_quiz import normalize_case, normalize_guess


ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "data" / "audio_diagnosis_lab"


class AudioDiagnosisLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public = json.loads((LAB_DIR / "manifest.json").read_text(encoding="utf-8"))
        cls.answers = json.loads((LAB_DIR / "answer_key.json").read_text(encoding="utf-8"))
        cls.by_issue = {item["issue"]: item for item in cls.answers["cases"]}

    def test_twenty_four_blind_cases_and_clean_controls_exist(self) -> None:
        self.assertEqual(len(self.public["cases"]), 24)
        self.assertEqual(len(self.answers["cases"]), 24)
        self.assertFalse(any("issue" in item for item in self.public["cases"]))
        self.assertIn("clean", self.by_issue)
        self.assertIn("clean_stereo", self.by_issue)

    def test_difficulty_split_is_stable(self) -> None:
        counts = {
            level: sum(item["difficulty"] == level for item in self.public["cases"])
            for level in ("beginner", "intermediate", "advanced")
        }
        self.assertEqual(counts, {"beginner": 7, "intermediate": 9, "advanced": 8})

    def test_key_faults_have_measurable_evidence(self) -> None:
        clipped, _ = sf.read(LAB_DIR / self.by_issue["hard_clipping"]["file"], dtype="float64")
        dc, _ = sf.read(LAB_DIR / self.by_issue["dc_offset"]["file"], dtype="float64")
        polarity, _ = sf.read(
            LAB_DIR / self.by_issue["stereo_polarity_inversion"]["file"],
            dtype="float64",
            always_2d=True,
        )
        silent, _ = sf.read(
            LAB_DIR / self.by_issue["right_channel_silent"]["file"],
            dtype="float64",
            always_2d=True,
        )
        _, wrong_rate = sf.read(
            LAB_DIR / self.by_issue["wrong_sample_rate_metadata"]["file"],
            dtype="float64",
        )
        self.assertGreater(np.mean(np.abs(clipped) >= 0.999), 0.005)
        self.assertGreater(np.mean(dc), 0.1)
        self.assertTrue(np.allclose(polarity[:, 0] + polarity[:, 1], 0.0, atol=1/32768))
        self.assertTrue(np.allclose(silent[:, 1], 0.0))
        self.assertEqual(wrong_rate, 8_000)

    def test_quiz_accepts_short_case_ids_and_chinese_aliases(self) -> None:
        self.assertEqual(normalize_case("7"), "case_07")
        self.assertEqual(normalize_case("case-24"), "case_24")
        self.assertEqual(normalize_guess("削波"), "hard_clipping")
        self.assertEqual(normalize_guess("采样率错误"), "wrong_sample_rate_metadata")


if __name__ == "__main__":
    unittest.main()
