from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import torch

from acoustic_engine.streaming import StreamingAcousticEngine
from acoustic_engine.features import load_mono_audio

from fsdd_generalization.data import (
    ARCHIVE_SHA256,
    EXPECTED_SPEAKERS,
    Recording,
    SplitSpec,
    parse_recording,
    split_recordings,
)
from fsdd_generalization.training import augment_waveform, build_sequence_specs, edit_distance


class SplitTests(unittest.TestCase):
    def synthetic_recordings(self) -> list[Recording]:
        rows = []
        for speaker in EXPECTED_SPEAKERS:
            for digit in "0123456789":
                for index in range(50):
                    rows.append(
                        Recording(Path(f"{digit}_{speaker}_{index}.wav"), digit, speaker, index)
                    )
        return rows

    def test_default_split_is_speaker_disjoint_and_complete(self) -> None:
        splits = split_recordings(self.synthetic_recordings(), SplitSpec())
        speakers = {name: {row.speaker for row in rows} for name, rows in splits.items()}
        self.assertFalse(speakers["train"] & speakers["dev"])
        self.assertFalse(speakers["train"] & speakers["test"])
        self.assertFalse(speakers["dev"] & speakers["test"])
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 2000, "dev": 500, "test": 500})

    def test_overlap_is_rejected(self) -> None:
        spec = SplitSpec(
            train_speakers=("george", "jackson", "lucas", "nicolas"),
            dev_speakers=("theo",),
            test_speakers=("theo", "yweweler"),
        )
        with self.assertRaisesRegex(ValueError, "leakage"):
            spec.validate()

    def test_filename_parser_rejects_unknown_shape(self) -> None:
        parsed = parse_recording(Path("7_jackson_32.wav"))
        self.assertEqual((parsed.digit, parsed.speaker, parsed.index), ("7", "jackson", 32))
        with self.assertRaises(ValueError):
            parse_recording(Path("7_jackson.wav"))

    def test_archive_checksum_is_full_sha256(self) -> None:
        self.assertEqual(len(ARCHIVE_SHA256), 64)
        int(ARCHIVE_SHA256, 16)


class TrainingUtilityTests(unittest.TestCase):
    def test_augmentation_is_deterministic_and_changes_signal(self) -> None:
        waveform = torch.linspace(-0.5, 0.5, 800)
        first = augment_waveform(waveform, 12)
        second = augment_waveform(waveform, 12)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, waveform))
        self.assertLessEqual(float(first.abs().max()), 1.0)

    def test_edit_distance_counts_insertions_and_deletions(self) -> None:
        self.assertEqual(edit_distance("12", "12"), 0)
        self.assertEqual(edit_distance("12", "1"), 1)
        self.assertEqual(edit_distance("1", "123"), 2)

    def test_sequence_generation_stays_within_one_speaker_and_includes_repeats(self) -> None:
        rows = []
        for digit in "0123456789":
            for index in range(5):
                rows.append(Recording(Path(f"{digit}_george_{index}.wav"), digit, "george", index))
        specs = build_sequence_specs(
            rows, count=20, min_digits=2, max_digits=4, seed=9, prefix="test"
        )
        self.assertTrue(
            all({item.speaker for item in spec.recordings} == {spec.speaker} for spec in specs)
        )
        self.assertTrue(all(2 <= len(spec.text) <= 4 for spec in specs))
        self.assertTrue(any(spec.text[0] == spec.text[1] for spec in specs))


class PublishedArtifactTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_manifest_and_results_preserve_blind_split_contract(self) -> None:
        manifest = json.loads(
            (self.ROOT / "artifacts" / "fsdd_speaker_disjoint_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        results = json.loads(
            (self.ROOT / "artifacts" / "fsdd_speaker_disjoint_results.json").read_text(
                encoding="utf-8"
            )
        )
        speaker_sets = {
            split: {row["speaker"] for row in manifest["entries"] if row["split"] == split}
            for split in ("train", "dev", "test")
        }
        self.assertFalse(speaker_sets["train"] & speaker_sets["dev"])
        self.assertFalse(speaker_sets["train"] & speaker_sets["test"])
        self.assertFalse(speaker_sets["dev"] & speaker_sets["test"])
        for experiment in ("baseline", "augmented"):
            history = results[experiment]["history"]
            self.assertTrue(all(set(row) == {"epoch", "train_loss", "dev_cer", "dev_exact_rate"} for row in history))
            expected_best = min(history, key=lambda row: (row["dev_cer"], -row["dev_exact_rate"]))["epoch"]
            self.assertEqual(results[experiment]["best_epoch"], expected_best)
        self.assertLess(results["augmented"]["test"]["cer"], results["baseline"]["test"]["cer"])

    def test_published_checkpoint_loads_and_is_chunk_invariant(self) -> None:
        engine = StreamingAcousticEngine.load(
            self.ROOT / "artifacts" / "fsdd_speaker_disjoint_ctc.pt"
        )
        waveform = load_mono_audio(
            self.ROOT / "data" / "spoken_digits_parts" / "7_jackson_0.wav",
            engine.frontend_config.sample_rate,
        )
        predictions = {
            engine.recognize_waveform(waveform, chunk_samples=size).text
            for size in (1, 137, 400, 800, 1600)
        }
        self.assertEqual(len(predictions), 1)
        self.assertTrue(next(iter(predictions)))


if __name__ == "__main__":
    unittest.main()
