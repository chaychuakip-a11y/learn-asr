from __future__ import annotations

from pathlib import Path
import unittest

from external_evaluation.data import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    AUDIOMNIST_REVISION,
    EXPECTED_SPEAKERS,
    parse_recording,
)


class AudioMnistDataContractTests(unittest.TestCase):
    def test_source_is_content_addressed_and_complete(self) -> None:
        self.assertEqual(len(AUDIOMNIST_REVISION), 40)
        int(AUDIOMNIST_REVISION, 16)
        self.assertEqual(len(ARCHIVE_SHA256), 64)
        int(ARCHIVE_SHA256, 16)
        self.assertEqual(ARCHIVE_BYTES, 996_621_372)
        self.assertEqual(EXPECTED_SPEAKERS, tuple(f"{i:02d}" for i in range(1, 61)))

    def test_filename_parser_checks_directory_speaker_and_index(self) -> None:
        row = parse_recording(Path("data/07/3_07_49.wav"))
        self.assertEqual((row.digit, row.speaker, row.index), ("3", "07", 49))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            parse_recording(Path("data/08/3_07_49.wav"))
        with self.assertRaisesRegex(ValueError, "0..49"):
            parse_recording(Path("data/07/3_07_50.wav"))


if __name__ == "__main__":
    unittest.main()
