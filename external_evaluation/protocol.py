from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path


PROTOCOL_PATH = Path(__file__).with_name("audiomnist_protocol.json")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def protocol_sha256(path: Path = PROTOCOL_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate_protocol(protocol: dict[str, object]) -> None:
    if protocol.get("format_version") != 1:
        raise ValueError("unsupported external-evaluation protocol version")
    if protocol.get("status") != "frozen_before_audio_download_or_scoring":
        raise ValueError("protocol is not in the pre-contact frozen state")
    source = protocol["source"]
    if not isinstance(source, dict):
        raise TypeError("source must be a dictionary")
    if source.get("revision") != "630d7dab4c040882834de6fa21baf9a60372accd":
        raise ValueError("unexpected AudioMNIST revision")
    if source.get("license") != "MIT" or source.get("expected_recordings") != 30_000:
        raise ValueError("unexpected AudioMNIST license or recording count")

    prior = protocol["prior_contact"]
    if not isinstance(prior, dict) or any(
        prior.get(field) is not False
        for field in (
            "audiomnist_audio_downloaded",
            "audiomnist_audio_listened_to",
            "audiomnist_labels_scored",
        )
    ):
        raise ValueError("protocol must be frozen before AudioMNIST audio contact")

    final_fit = protocol["final_fit"]
    if not isinstance(final_fit, dict):
        raise TypeError("final_fit must be a dictionary")
    derivation = final_fit["epoch_derivation"]
    if not isinstance(derivation, dict):
        raise TypeError("epoch_derivation must be a dictionary")
    epochs = derivation["fold_best_epochs"]
    if not isinstance(epochs, list) or len(epochs) != 6:
        raise ValueError("epoch derivation needs six LOSO folds")
    ordered = sorted(int(value) for value in epochs)
    median = (ordered[2] + ordered[3]) / 2
    if median != float(derivation["median"]):
        raise ValueError("recorded LOSO epoch median is inconsistent")
    if round_half_up(median) != int(final_fit["epochs"]):
        raise ValueError("final-fit epoch count does not follow the frozen rule")

    tracks = protocol["evaluation_tracks"]
    if not isinstance(tracks, dict):
        raise TypeError("evaluation_tracks must be a dictionary")
    if tracks["single_digit"]["recordings"] != 30_000:  # type: ignore[index]
        raise ValueError("single-digit track must score the full dataset")
    if tracks["multi_digit"]["total_sequences"] != 6_000:  # type: ignore[index]
        raise ValueError("multi-digit track must preserve 100 sequences per speaker")
    stopping = protocol["stopping_and_disclosure"]
    if not isinstance(stopping, dict) or not (
        stopping.get("score_once")
        and stopping.get("publish_regardless_of_quality")
        and stopping.get("no_audiomnist_driven_model_change_can_retain_untouched_label")
    ):
        raise ValueError("external score-once disclosure rules are incomplete")


def main() -> None:
    protocol = load_protocol()
    print("protocol_id:", protocol["protocol_id"])
    print("sha256:", protocol_sha256())
    print("status:", protocol["status"])


if __name__ == "__main__":
    main()
