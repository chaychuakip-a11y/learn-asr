from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

from external_evaluation.final_fit import PREREGISTRATION_COMMIT, frozen_final_config
from external_evaluation.protocol import protocol_sha256
from acoustic_engine.streaming import StreamingAcousticEngine


class FrozenFinalFitTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_config_matches_preregistered_final_fit(self) -> None:
        config = frozen_final_config()
        self.assertEqual(config.epochs, 10)
        self.assertEqual(config.train_sequences, 3_000)
        self.assertEqual(config.augmentation_copies, 1)
        self.assertEqual((config.hidden_dim, config.num_layers, config.kernel_size), (48, 5, 5))
        self.assertEqual((config.dev_sequences, config.test_sequences), (0, 0))
        self.assertEqual(len(PREREGISTRATION_COMMIT), 40)

    def test_final_fit_module_has_no_audiomnist_data_dependency(self) -> None:
        source_path = self.ROOT / "external_evaluation" / "final_fit.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(
            any(name.endswith("external_evaluation.data") or name == ".data" for name in imported)
        )

    def test_published_final_fit_contains_no_selection_metric(self) -> None:
        evidence = json.loads(
            (self.ROOT / "artifacts" / "fsdd_final_fit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(evidence["audio_mnist_accessed_by_final_fit"])
        self.assertEqual(evidence["protocol_sha256"], protocol_sha256())
        self.assertEqual(evidence["preregistration_commit"], PREREGISTRATION_COMMIT)
        self.assertEqual(evidence["training_recordings"], 3_000)
        self.assertEqual(len(evidence["history"]), 10)
        self.assertTrue(
            all(set(row) == {"epoch", "train_loss"} for row in evidence["history"])
        )
        self.assertFalse(any("dev" in key or "test" in key for key in evidence))
        checkpoint = self.ROOT / "artifacts" / evidence["checkpoint"]
        engine = StreamingAcousticEngine.load(checkpoint)
        self.assertEqual(engine.model.config.hidden_dim, 48)
        self.assertEqual(engine.model.config.num_layers, 5)


if __name__ == "__main__":
    unittest.main()
