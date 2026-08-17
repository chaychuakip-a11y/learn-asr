from __future__ import annotations

import json
from pathlib import Path
import statistics
import unittest

from external_evaluation.protocol import load_protocol, protocol_sha256, round_half_up


class FrozenAudioMnistProtocolTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_protocol_is_frozen_before_audio_contact(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["status"], "frozen_before_audio_download_or_scoring")
        self.assertEqual(protocol["source"]["license"], "MIT")
        self.assertEqual(protocol["source"]["expected_speakers"], 60)
        self.assertEqual(protocol["evaluation_tracks"]["single_digit"]["recordings"], 30_000)
        self.assertEqual(protocol["evaluation_tracks"]["multi_digit"]["total_sequences"], 6_000)
        self.assertEqual(len(protocol_sha256()), 64)

    def test_final_epoch_rule_matches_published_loso_evidence(self) -> None:
        protocol = load_protocol()
        loso = json.loads(
            (self.ROOT / "artifacts" / "fsdd_loso_results.json").read_text(
                encoding="utf-8"
            )
        )
        observed = [
            row["candidate_results"][row["selected_candidate"]]["best_epoch"]
            for row in loso["folds"]
        ]
        frozen = protocol["final_fit"]["epoch_derivation"]["fold_best_epochs"]
        self.assertEqual(observed, frozen)
        self.assertEqual(statistics.median(observed), 9.5)
        self.assertEqual(round_half_up(statistics.median(observed)), 10)
        self.assertEqual(protocol["final_fit"]["epochs"], 10)

    def test_score_once_and_publish_regardless_rules_are_explicit(self) -> None:
        rules = load_protocol()["stopping_and_disclosure"]
        self.assertTrue(rules["score_once"])
        self.assertTrue(rules["publish_regardless_of_quality"])
        self.assertTrue(rules["no_audiomnist_driven_model_change_can_retain_untouched_label"])


if __name__ == "__main__":
    unittest.main()
