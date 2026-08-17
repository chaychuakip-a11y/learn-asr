from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import soundfile as sf
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
from fsdd_generalization.loso import (
    aggregate_outer_folds,
    build_loso_folds,
    select_candidate,
)
from fsdd_generalization.training import (
    ExperimentConfig,
    augment_waveform,
    build_sequence_specs,
    edit_distance,
    prepare_experiment_features,
    run_training,
)


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

    def test_candidate_training_can_be_physically_isolated_from_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits: dict[str, list[Recording]] = {"train": [], "dev": []}
            for split, speaker in (("train", "george"), ("dev", "jackson")):
                for digit in "0123456789":
                    path = root / f"{digit}_{speaker}_0.wav"
                    waveform = torch.sin(
                        2 * torch.pi * (180 + int(digit) * 20) * torch.arange(800) / 8_000
                    ).mul(0.1).numpy()
                    sf.write(path, waveform, 8_000)
                    splits[split].append(Recording(path, digit, speaker, 0))
            config = ExperimentConfig(
                epochs=1,
                batch_size=2,
                hidden_dim=4,
                num_layers=1,
                train_sequences=4,
                dev_sequences=2,
                test_sequences=2,
            )
            prepared = prepare_experiment_features(
                splits["train"], splits["dev"], [], config
            )
            self.assertEqual(prepared.test_specs, ())
            result = run_training(
                splits["train"],
                splits["dev"],
                [],
                config,
                prepared=prepared,
                evaluate_test=False,
            )
            self.assertFalse(result["test_evaluated"])
            self.assertNotIn("test", result)


class LosoProtocolTests(unittest.TestCase):
    def test_folds_are_disjoint_and_rotate_every_speaker(self) -> None:
        folds = build_loso_folds()
        self.assertEqual(len(folds), 6)
        self.assertEqual({fold.test_speaker for fold in folds}, set(EXPECTED_SPEAKERS))
        self.assertEqual({fold.dev_speaker for fold in folds}, set(EXPECTED_SPEAKERS))
        for fold in folds:
            groups = [set(fold.train_speakers), {fold.dev_speaker}, {fold.test_speaker}]
            self.assertFalse(groups[0] & groups[1])
            self.assertFalse(groups[0] & groups[2])
            self.assertFalse(groups[1] & groups[2])
            self.assertEqual(set.union(*groups), set(EXPECTED_SPEAKERS))

    def test_selection_uses_dev_metrics_and_prefers_simplicity_on_tie(self) -> None:
        tied = {
            "clean": {"dev": {"cer": 0.4, "exact_rate": 0.5}},
            "gain_noise": {"dev": {"cer": 0.4, "exact_rate": 0.5}},
        }
        self.assertEqual(select_candidate(tied), "clean")
        improved = {
            **tied,
            "gain_noise": {"dev": {"cer": 0.39, "exact_rate": 0.4}},
        }
        self.assertEqual(select_candidate(improved), "gain_noise")

    def test_aggregate_distinguishes_micro_and_speaker_macro(self) -> None:
        folds = []
        for index, speaker in enumerate(EXPECTED_SPEAKERS):
            sequence = {
                "errors": index + 1,
                "reference_characters": 100 + index * 10,
                "exact": 10,
                "total": 20,
                "cer": (index + 1) / (100 + index * 10),
                "exact_rate": 0.5,
            }
            folds.append(
                {
                    "test_speaker": speaker,
                    "selected_candidate": "clean" if index < 3 else "gain_noise",
                    "outer_test": {"sequence": sequence, "single_digit": sequence},
                }
            )
        aggregate = aggregate_outer_folds(folds)
        self.assertEqual(aggregate["sequence"]["total_errors"], 21)
        self.assertEqual(aggregate["selected_candidate_counts"], {"clean": 3, "gain_noise": 3})
        self.assertEqual(
            aggregate["sequence"]["macro_speaker_cer"]["speaker_count"], 6
        )


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


class PublishedLosoArtifactTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_loso_results_recompute_selection_and_speaker_aggregate(self) -> None:
        results = json.loads(
            (self.ROOT / "artifacts" / "fsdd_loso_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            results["interpretation"],
            "exploratory_resampling_estimate_not_a_new_untouched_benchmark",
        )
        folds = results["folds"]
        self.assertEqual(len(folds), 6)
        self.assertEqual({row["test_speaker"] for row in folds}, set(EXPECTED_SPEAKERS))
        for row in folds:
            self.assertEqual(
                row["selected_candidate"], select_candidate(row["candidate_results"])
            )
            self.assertTrue(row["outer_test_constructed_after_selection"])
            for candidate in row["candidate_results"].values():
                self.assertFalse(candidate["test_evaluated"])
                self.assertNotIn("test", candidate)
                self.assertEqual(candidate["test_sequences"], 0)
        recomputed = aggregate_outer_folds(folds)
        self.assertEqual(
            recomputed["selected_candidate_counts"],
            results["aggregate"]["selected_candidate_counts"],
        )
        self.assertAlmostEqual(
            recomputed["sequence"]["micro_cer"],
            results["aggregate"]["sequence"]["micro_cer"],
        )
        self.assertAlmostEqual(
            recomputed["sequence"]["macro_speaker_cer"]["sample_std"],
            results["aggregate"]["sequence"]["macro_speaker_cer"]["sample_std"],
        )

    def test_all_selected_loso_checkpoints_load(self) -> None:
        results = json.loads(
            (self.ROOT / "artifacts" / "fsdd_loso_results.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint_dir = self.ROOT / "artifacts" / "fsdd_loso_checkpoints"
        names = {row["selected_checkpoint"] for row in results["folds"]}
        self.assertEqual(len(names), 6)
        for name in names:
            engine = StreamingAcousticEngine.load(checkpoint_dir / name)
            self.assertEqual(engine.frontend_config.sample_rate, 8_000)
            self.assertEqual(engine.tokens, tuple("0123456789"))


if __name__ == "__main__":
    unittest.main()
