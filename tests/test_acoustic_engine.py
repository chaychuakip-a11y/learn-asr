from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from itertools import product
from collections import defaultdict
import math
import json

import torch
from fastapi.testclient import TestClient

from acoustic_engine.api import create_app
from acoustic_engine.decoder import StreamingGreedyDecoder, StreamingPrefixBeamDecoder, ctc_collapse, greedy_decode, prefix_beam_search
from acoustic_engine.engine import AcousticEngine
from acoustic_engine.features import LogMelConfig, LogMelFrontend, StreamingLogMelFrontend
from acoustic_engine.model import CausalConvCTCAcousticModel, CausalConvCTCConfig, StreamingCTCAcousticModel, StreamingCTCConfig, TinyCTCAcousticModel, TinyCTCConfig
from acoustic_engine.language_model import AddKBigramLanguageModel, ShallowFusionScorer
from acoustic_engine.streaming import StreamingAcousticEngine
from acoustic_engine.benchmark import summarize
from acoustic_engine.tutor import QUESTIONS, current_question, is_correct, load_progress, record_attempt


class FrontendTests(unittest.TestCase):
    def test_log_mel_contract(self) -> None:
        config = LogMelConfig(
            sample_rate=8_000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=24,
        )
        time = torch.arange(4_000) / config.sample_rate
        waveform = torch.sin(2 * torch.pi * 440 * time)
        features = LogMelFrontend(config)(waveform)
        expected_frames = 1 + (waveform.numel() - config.n_fft) // config.hop_length
        self.assertEqual(features.shape, (expected_frames, config.n_mels))
        self.assertTrue(torch.isfinite(features).all())

    def test_peak_normalization_is_shared_by_training_and_inference(self) -> None:
        config = LogMelConfig(
            sample_rate=8_000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=24,
        )
        frontend = LogMelFrontend(config)
        waveform = torch.randn(4_000)
        self.assertTrue(torch.allclose(frontend(waveform), frontend(waveform * 0.2), atol=1e-5))

    def test_streaming_frontend_matches_whole_waveform(self) -> None:
        config = LogMelConfig(
            sample_rate=8_000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=24,
            peak_normalize=False,
        )
        waveform = torch.randn(4_321)
        expected = LogMelFrontend(config)(waveform)
        streaming = StreamingLogMelFrontend(config)
        pieces = []
        boundaries = [137, 990, 1_045, 2_900, waveform.numel()]
        start = 0
        for end in boundaries:
            pieces.append(streaming.accept(waveform[start:end], final=end == waveform.numel()))
            start = end
        actual = torch.cat(pieces)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))


class DecoderTests(unittest.TestCase):
    def test_blank_separates_repeated_tokens(self) -> None:
        text, token_ids = ctc_collapse([1, 1, 0, 1, 2, 2, 0], tuple("0123456789"))
        self.assertEqual(text, "001")
        self.assertEqual(token_ids, [0, 0, 1])

    def test_greedy_decode_obeys_lengths(self) -> None:
        logits = torch.full((1, 5, 3), -10.0)
        logits[0, torch.arange(5), torch.tensor([1, 1, 0, 2, 1])] = 10.0
        self.assertEqual(greedy_decode(logits, torch.tensor([4]), ("a", "b")), ["ab"])

    def test_prefix_beam_matches_exhaustive_ctc_sum(self) -> None:
        probabilities = torch.tensor(
            [[0.50, 0.30, 0.20], [0.35, 0.40, 0.25], [0.45, 0.15, 0.40]],
            dtype=torch.float64,
        )
        totals: defaultdict[str, float] = defaultdict(float)
        for path in product(range(3), repeat=probabilities.shape[0]):
            text, _ = ctc_collapse(path, ("a", "b"))
            path_probability = math.prod(
                float(probabilities[frame, class_id])
                for frame, class_id in enumerate(path)
            )
            totals[text] += path_probability
        expected_text, expected_probability = max(totals.items(), key=lambda item: item[1])
        hypotheses = prefix_beam_search(probabilities.log(), ("a", "b"), beam_size=20)
        self.assertEqual(hypotheses[0].text, expected_text)
        self.assertAlmostEqual(math.exp(hypotheses[0].score), expected_probability, places=6)

    def test_language_model_can_change_an_ambiguous_choice(self) -> None:
        logits = torch.tensor([[math.log(0.01), math.log(0.51), math.log(0.48)]])
        acoustic_only = prefix_beam_search(logits, ("0", "1"), beam_size=3)[0]
        language_model = AddKBigramLanguageModel.fit(["0"] + ["1"] * 10, ("0", "1"))
        scorer = ShallowFusionScorer(language_model=language_model, lm_weight=1.0)
        fused = prefix_beam_search(
            logits,
            ("0", "1"),
            beam_size=3,
            extension_scorer=scorer,
        )[0]
        self.assertEqual(acoustic_only.text, "0")
        self.assertEqual(fused.text, "1")

    def test_hotword_bonus_applies_when_prefix_completes(self) -> None:
        scorer = ShallowFusionScorer(hotwords=((1, 2),), hotword_bonus=3.0)
        self.assertEqual(scorer.score_extension((), 1), 0.0)
        self.assertEqual(scorer.score_extension((1,), 2), 3.0)

    def test_token_insertion_bonus_applies_to_each_extension(self) -> None:
        scorer = ShallowFusionScorer(token_bonus=0.7)
        self.assertAlmostEqual(scorer.score_extension((), 1), 0.7)
        self.assertAlmostEqual(scorer.score_extension((1,), 2), 0.7)

    def test_language_model_round_trip_preserves_scores(self) -> None:
        model = AddKBigramLanguageModel.fit(["12", "12", "10"], tuple("012"))
        before = model.sequence_log_probability((1, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digits.bigram.json"
            model.save(path)
            restored = AddKBigramLanguageModel.load(path)
        self.assertAlmostEqual(restored.sequence_log_probability((1, 2)), before)


class ModelAndEngineTests(unittest.TestCase):
    def make_engine(self) -> AcousticEngine:
        frontend = LogMelFrontend(
            LogMelConfig(
                sample_rate=8_000,
                n_fft=256,
                win_length=200,
                hop_length=80,
                n_mels=24,
            )
        )
        model = TinyCTCAcousticModel(
            TinyCTCConfig(feature_dim=24, hidden_dim=8, num_classes=3)
        )
        return AcousticEngine(frontend, model, ("a", "b"))

    def test_model_preserves_time_lengths(self) -> None:
        engine = self.make_engine()
        features = torch.randn(2, 17, 24)
        lengths = torch.tensor([17, 11])
        logits, output_lengths = engine.model(features, lengths)
        self.assertEqual(logits.shape, (2, 17, 3))
        self.assertTrue(torch.equal(output_lengths, lengths))

    def test_valid_logits_do_not_depend_on_batch_padding(self) -> None:
        torch.manual_seed(11)
        engine = self.make_engine()
        short = torch.randn(1, 11, 24)
        alone, _ = engine.model(short, torch.tensor([11]))
        padded = torch.zeros(2, 19, 24)
        padded[0, :11] = short[0]
        padded[1] = torch.randn(19, 24)
        batched, _ = engine.model(padded, torch.tensor([11, 19]))
        self.assertTrue(torch.allclose(alone[0], batched[0, :11], atol=1e-6))

    def test_checkpoint_round_trip_preserves_prediction(self) -> None:
        torch.manual_seed(7)
        engine = self.make_engine()
        waveform = torch.randn(4_000)
        before = engine.recognize_waveform(waveform)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "engine.pt"
            engine.save(checkpoint)
            after = AcousticEngine.load(checkpoint).recognize_waveform(waveform)
        self.assertEqual(after.text, before.text)
        self.assertEqual(after.frame_ids, before.frame_ids)
        self.assertEqual(after.feature_frames, before.feature_frames)

    def test_streaming_model_chunk_logits_match_whole_utterance(self) -> None:
        torch.manual_seed(19)
        model = StreamingCTCAcousticModel(
            StreamingCTCConfig(feature_dim=24, hidden_dim=8, num_layers=2, num_classes=3)
        ).eval()
        features = torch.randn(1, 23, 24)
        whole, _ = model(features, torch.tensor([23]))
        state = None
        chunks = []
        for start, end in [(0, 3), (3, 11), (11, 12), (12, 23)]:
            logits, state = model.forward_chunk(features[:, start:end], state)
            chunks.append(logits)
        streamed = torch.cat(chunks, dim=1)
        self.assertTrue(torch.allclose(streamed, whole, atol=5e-6))

    def test_causal_conv_chunk_logits_match_whole_utterance(self) -> None:
        torch.manual_seed(29)
        model = CausalConvCTCAcousticModel(
            CausalConvCTCConfig(
                feature_dim=24,
                hidden_dim=8,
                num_layers=3,
                kernel_size=3,
                num_classes=3,
            )
        ).eval()
        features = torch.randn(1, 23, 24)
        whole, _ = model(features, torch.tensor([23]))
        state = None
        chunks = []
        for start, end in [(0, 3), (3, 11), (11, 12), (12, 23)]:
            logits, state = model.forward_chunk(features[:, start:end], state)
            chunks.append(logits)
        streamed = torch.cat(chunks, dim=1)
        self.assertTrue(torch.allclose(streamed, whole, atol=5e-6))

    def test_streaming_ctc_collapse_keeps_state_across_chunks(self) -> None:
        decoder = StreamingGreedyDecoder(("a", "b"))
        self.assertEqual(decoder.accept([1, 1]), "a")
        self.assertEqual(decoder.accept([1, 0]), "a")
        self.assertEqual(decoder.accept([1, 2, 2]), "aab")

    def test_streaming_prefix_beam_matches_uninterrupted_beam(self) -> None:
        torch.manual_seed(23)
        logits = torch.randn(17, 4)
        expected = prefix_beam_search(logits, ("a", "b", "c"), beam_size=12)[0]
        decoder = StreamingPrefixBeamDecoder(("a", "b", "c"), beam_size=12)
        decoder.accept_logits(logits[:2])
        decoder.accept_logits(logits[2:9])
        actual = decoder.accept_logits(logits[9:])[0]
        self.assertEqual(actual.text, expected.text)
        self.assertAlmostEqual(actual.score, expected.score, places=6)


class ApiTests(unittest.TestCase):
    def make_client(self) -> TestClient:
        torch.manual_seed(31)
        frontend_config = LogMelConfig(
            sample_rate=8_000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=24,
            peak_normalize=False,
        )
        model = CausalConvCTCAcousticModel(
            CausalConvCTCConfig(
                feature_dim=24,
                hidden_dim=8,
                num_layers=2,
                kernel_size=3,
                num_classes=3,
            )
        )
        engine = StreamingAcousticEngine(
            frontend_config,
            model,
            ("a", "b"),
            torch.zeros(24),
            torch.ones(24),
        )
        return TestClient(create_app(engine))

    def test_http_and_websocket_share_the_same_engine_contract(self) -> None:
        client = self.make_client()
        waveform = torch.randn(1_337)
        response = client.post(
            "/recognize",
            json={"samples": waveform.tolist(), "sample_rate": 8_000, "chunk_samples": 211},
        )
        self.assertEqual(response.status_code, 200)
        expected_text = response.json()["text"]
        with client.websocket_connect("/stream") as websocket:
            final = None
            for start in range(0, waveform.numel(), 211):
                end = min(start + 211, waveform.numel())
                websocket.send_json(
                    {
                        "samples": waveform[start:end].tolist(),
                        "sample_rate": 8_000,
                        "final": end == waveform.numel(),
                    }
                )
                final = websocket.receive_json()
        self.assertIsNotNone(final)
        self.assertEqual(final["text"], expected_text)
        self.assertTrue(final["is_final"])

    def test_http_rejects_wrong_sample_rate(self) -> None:
        client = self.make_client()
        response = client.post(
            "/recognize",
            json={"samples": [0.0] * 300, "sample_rate": 16_000},
        )
        self.assertEqual(response.status_code, 422)


class BenchmarkTests(unittest.TestCase):
    def test_summary_uses_audio_duration_for_rtf(self) -> None:
        summary = summarize([0.1, 0.2, 0.3, 0.4], audio_seconds=2.0)
        self.assertAlmostEqual(summary.mean_seconds, 0.25)
        self.assertAlmostEqual(summary.p50_seconds, 0.25)
        self.assertAlmostEqual(summary.p95_seconds, 0.4)
        self.assertAlmostEqual(summary.real_time_factor, 0.125)


class TutorTests(unittest.TestCase):
    def test_answer_check_normalizes_beginner_input(self) -> None:
        question = next(item for item in QUESTIONS if item.id == "sampling-count")
        self.assertTrue(is_correct(question, " 4000 个 "))
        self.assertFalse(is_correct(question, "8000"))

    def test_progress_preserves_learning_hub_fields(self) -> None:
        question = QUESTIONS[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning_progress.json"
            path.write_text(
                json.dumps({"diagnostic": {"声音与特征": 1}, "mastery": {"CTC": 0}}, ensure_ascii=False),
                encoding="utf-8",
            )
            record_attempt(path, question, "4000", True)
            restored = load_progress(path)
        self.assertEqual(restored["diagnostic"], {"声音与特征": 1})
        self.assertEqual(restored["mastery"], {"CTC": 0})
        self.assertIs(restored["tutor"]["attempts"][question.id]["passed"], True)

    def test_wrong_answer_does_not_unlock_next_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learning_progress.json"
            record_attempt(path, QUESTIONS[0], "8000", False)
            progress = load_progress(path)
        self.assertEqual(current_question(progress), QUESTIONS[0])


if __name__ == "__main__":
    unittest.main()
