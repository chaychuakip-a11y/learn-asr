from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "audio_software_lab"


class AudioSoftwareLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.guide = (ROOT / "AUDIO_SOFTWARE_GUIDE.md").read_text(encoding="utf-8")
        cls.manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))

    def test_guide_covers_editing_speech_and_cross_tool_verification(self) -> None:
        for marker in (
            "Audacity",
            "Praat",
            "Sonic Visualiser",
            "Adobe Audition",
            "Cool Edit",
            "dBFS",
            "Formant",
            "TextGrid",
            "manifest.json",
            "最终通关门禁",
        ):
            self.assertIn(marker, self.guide)

    def test_manifest_has_seven_readable_assets(self) -> None:
        assets = self.manifest["assets"]
        self.assertEqual(len(assets), 7)
        for item in assets:
            info = sf.info(ASSET_DIR / item["file"])
            self.assertEqual(info.samplerate, item["sample_rate_hz"])
            self.assertEqual(info.channels, item["channels"])

    def test_calibration_tone_has_expected_peak_and_frequency(self) -> None:
        audio, sample_rate = sf.read(
            ASSET_DIR / "01_calibration_440hz_peak_minus12dbfs.wav",
            dtype="float64",
        )
        peak_dbfs = 20 * np.log10(np.max(np.abs(audio)))
        frequencies = np.fft.rfftfreq(len(audio), d=1 / sample_rate)
        dominant = frequencies[np.argmax(np.abs(np.fft.rfft(audio)))]
        self.assertAlmostEqual(peak_dbfs, -12.0, places=5)
        self.assertAlmostEqual(dominant, 440.0, places=6)

    def test_stereo_asset_has_exact_sixteen_sample_delay(self) -> None:
        audio, _ = sf.read(
            ASSET_DIR / "06_stereo_right_delayed_1ms.wav",
            dtype="float64",
            always_2d=True,
        )
        left, right = audio[:, 0], audio[:, 1]
        self.assertTrue(np.allclose(right[:16], 0.0))
        self.assertTrue(np.allclose(right[16:], left[:-16]))


if __name__ == "__main__":
    unittest.main()
