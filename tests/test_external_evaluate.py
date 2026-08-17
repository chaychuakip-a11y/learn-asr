from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from external_evaluation.data import AudioMnistRecording, EXPECTED_SPEAKERS
from external_evaluation.evaluate import (
    _age_band,
    aggregate_speakers,
    build_external_sequence_specs,
    edit_operation_counts,
    run_external_evaluation,
)
from external_evaluation.protocol import protocol_sha256


class ExternalEvaluationUtilityTests(unittest.TestCase):
    def test_edit_operations_sum_to_known_distance(self) -> None:
        self.assertEqual(
            edit_operation_counts("8817", "8855"),
            {"correct": 2, "substitution": 2, "deletion": 0, "insertion": 0},
        )
        deletion = edit_operation_counts("123", "13")
        self.assertEqual(deletion["deletion"], 1)
        insertion = edit_operation_counts("13", "123")
        self.assertEqual(insertion["insertion"], 1)

    def test_external_sequence_registry_has_100_per_speaker_and_repeats(self) -> None:
        recordings = [
            AudioMnistRecording(Path(f"data/{speaker}/{digit}_{speaker}_{index}.wav"), digit, speaker, index)
            for speaker in EXPECTED_SPEAKERS
            for digit in "0123456789"
            for index in range(2)
        ]
        specs = build_external_sequence_specs(recordings)
        counts = {speaker: sum(spec.speaker == speaker for spec in specs) for speaker in EXPECTED_SPEAKERS}
        self.assertEqual(set(counts.values()), {100})
        self.assertTrue(any(len(spec.text) >= 2 and spec.text[0] == spec.text[1] for spec in specs))

    def test_age_bands_are_exhaustive_at_boundaries(self) -> None:
        self.assertEqual(_age_band(29), "under_30")
        self.assertEqual(_age_band(30), "30_to_39")
        self.assertEqual(_age_band(39), "30_to_39")
        self.assertEqual(_age_band(40), "40_and_over")

    def test_aggregate_requires_all_60_speakers(self) -> None:
        with self.assertRaisesRegex(ValueError, "60"):
            aggregate_speakers({})

    def test_first_and_only_result_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "already_scored.json"
            output_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already been scored"):
                run_external_evaluation(
                    Path("unused-audiomnist"),
                    Path("unused-checkpoint.pt"),
                    Path("unused-manifest.json"),
                    output_path,
                )


class PublishedExternalEvaluationTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(
            (cls.ROOT / "artifacts" / "audiomnist_external_results.json").read_text(
                encoding="utf-8"
            )
        )

    def test_result_is_bound_to_preregistered_protocol_and_frozen_model(self) -> None:
        self.assertEqual(
            self.result["status"],
            "completed_first_and_only_untouched_audiomnist_score",
        )
        self.assertEqual(self.result["protocol_sha256"], protocol_sha256())
        self.assertEqual(
            self.result["frozen_checkpoint_sha256"],
            "5c74192bfa1b3bbddbf614a08a60b592082e7af0981af8facd3370e84d80e337",
        )
        self.assertEqual(
            self.result["preregistered_interpretation_band"],
            "weak_transfer_requires_separately_designed_adaptation",
        )

    def test_full_external_population_and_streaming_contract_were_scored(self) -> None:
        single = self.result["single_digit"]
        sequence = self.result["multi_digit"]
        self.assertEqual(single["aggregate"]["total_utterances"], 30_000)
        self.assertEqual(sequence["aggregate"]["total_utterances"], 6_000)
        self.assertEqual(len(single["per_speaker"]), 60)
        self.assertEqual(len(sequence["per_speaker"]), 60)
        self.assertEqual(single["aggregate"]["speaker_macro_cer"]["speaker_count"], 60)
        self.assertEqual(self.result["streaming"]["invariant_speakers"], 60)
        self.assertEqual(self.result["streaming"]["checked_speakers"], 60)

    def test_aggregate_counts_recompute_from_speakers_and_predictions(self) -> None:
        for track in ("single_digit", "multi_digit"):
            per_speaker = self.result[track]["per_speaker"]
            recomputed = aggregate_speakers(per_speaker)
            stored = self.result[track]["aggregate"]
            self.assertEqual(recomputed["total_errors"], stored["total_errors"])
            self.assertEqual(recomputed["total_exact"], stored["total_exact"])
            self.assertEqual(recomputed["edit_operations"], stored["edit_operations"])
            self.assertAlmostEqual(recomputed["micro_cer"], stored["micro_cer"])
            self.assertAlmostEqual(
                recomputed["speaker_macro_cer"]["sample_std"],
                stored["speaker_macro_cer"]["sample_std"],
            )

    def test_public_result_contains_relative_names_only(self) -> None:
        serialized = json.dumps(self.result, ensure_ascii=False)
        self.assertNotIn("G:\\\\", serialized)
        self.assertNotIn("C:\\\\Users", serialized)


if __name__ == "__main__":
    unittest.main()
