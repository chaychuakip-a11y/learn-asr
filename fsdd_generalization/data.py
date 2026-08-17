from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable
from urllib.request import Request, urlopen
import zipfile

import soundfile as sf


FSDD_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
ARCHIVE_URL = (
    "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/"
    + FSDD_REVISION
)
ARCHIVE_SHA256 = "beaca4fb849cad13103b2cffba913b3d7fdb5f15222da4ddd07187b3c38a757a"
EXPECTED_SPEAKERS = ("george", "jackson", "lucas", "nicolas", "theo", "yweweler")
EXPECTED_RECORDINGS = 3_000
EXPECTED_SAMPLE_RATE = 8_000


@dataclass(frozen=True)
class Recording:
    path: Path
    digit: str
    speaker: str
    index: int

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class SplitSpec:
    train_speakers: tuple[str, ...] = ("george", "jackson", "lucas", "nicolas")
    dev_speakers: tuple[str, ...] = ("theo",)
    test_speakers: tuple[str, ...] = ("yweweler",)

    def validate(self) -> None:
        groups = [set(self.train_speakers), set(self.dev_speakers), set(self.test_speakers)]
        if any(not group for group in groups):
            raise ValueError("train/dev/test speaker sets must all be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("speaker leakage: train/dev/test speaker sets overlap")
        all_speakers = groups[0] | groups[1] | groups[2]
        if all_speakers != set(EXPECTED_SPEAKERS):
            raise ValueError(
                f"split must cover exactly {EXPECTED_SPEAKERS}, got {sorted(all_speakers)}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"fsdd-{FSDD_REVISION[:8]}.zip"
    if archive.exists():
        actual = sha256(archive)
        if actual != ARCHIVE_SHA256:
            raise ValueError(
                f"cached archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}; "
                f"remove only this file and retry: {archive}"
            )
        return archive

    partial = archive.with_suffix(".zip.part")
    if partial.exists():
        raise FileExistsError(
            f"partial download already exists; inspect or remove it before retrying: {partial}"
        )
    request = Request(ARCHIVE_URL, headers={"User-Agent": "learn-asr-course/1.0"})
    try:
        with urlopen(request, timeout=90) as response, partial.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256(partial)
        if actual != ARCHIVE_SHA256:
            raise ValueError(
                f"downloaded archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
            )
        os.replace(partial, archive)
    except Exception:
        # Keep a partial file for inspection; never silently use it on the next run.
        raise
    return archive


def _selected_archive_name(member_name: str) -> str | None:
    parts = Path(member_name).parts
    if len(parts) == 3 and parts[1] == "recordings" and parts[2].endswith(".wav"):
        return parts[2]
    if len(parts) == 2 and parts[1] == "README.md":
        return "UPSTREAM_README.md"
    return None


def extract_recordings(archive: Path, cache_dir: Path) -> Path:
    dataset_root = cache_dir / f"fsdd-{FSDD_REVISION[:8]}"
    recordings_dir = dataset_root / "recordings"
    if recordings_dir.exists():
        if len(list(recordings_dir.glob("*.wav"))) != EXPECTED_RECORDINGS:
            raise ValueError(
                f"existing extraction is incomplete: {recordings_dir}; "
                "move it aside or remove this exact dataset directory before retrying"
            )
        return recordings_dir
    if dataset_root.exists():
        raise FileExistsError(f"dataset target exists but has no valid recordings directory: {dataset_root}")

    with tempfile.TemporaryDirectory(prefix="fsdd-extract-", dir=cache_dir) as temporary:
        temporary_root = Path(temporary) / "dataset"
        temporary_recordings = temporary_root / "recordings"
        temporary_recordings.mkdir(parents=True)
        selected = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination_name = _selected_archive_name(member.filename)
                if destination_name is None:
                    continue
                destination = (
                    temporary_root / destination_name
                    if destination_name == "UPSTREAM_README.md"
                    else temporary_recordings / destination_name
                )
                if destination.exists():
                    raise ValueError(f"duplicate archive member: {destination_name}")
                with bundle.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                if destination.suffix == ".wav":
                    selected += 1
        if selected != EXPECTED_RECORDINGS:
            raise ValueError(f"archive contains {selected} selected WAV files, expected {EXPECTED_RECORDINGS}")
        temporary_root.rename(dataset_root)
    return recordings_dir


def parse_recording(path: Path) -> Recording:
    pieces = path.stem.split("_")
    if len(pieces) != 3:
        raise ValueError(f"unexpected FSDD filename: {path.name}")
    digit, speaker, index_text = pieces
    if digit not in tuple("0123456789") or speaker not in EXPECTED_SPEAKERS:
        raise ValueError(f"unexpected FSDD label or speaker: {path.name}")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"unexpected FSDD recording index: {path.name}") from exc
    return Recording(path=path, digit=digit, speaker=speaker, index=index)


def scan_recordings(recordings_dir: Path, validate_audio: bool = True) -> list[Recording]:
    recordings = [parse_recording(path) for path in sorted(recordings_dir.glob("*.wav"))]
    if len(recordings) != EXPECTED_RECORDINGS:
        raise ValueError(f"found {len(recordings)} recordings, expected {EXPECTED_RECORDINGS}")
    if len({item.name for item in recordings}) != EXPECTED_RECORDINGS:
        raise ValueError("duplicate FSDD recording names")
    speakers = {item.speaker for item in recordings}
    if speakers != set(EXPECTED_SPEAKERS):
        raise ValueError(f"speaker set mismatch: {sorted(speakers)}")
    for speaker in EXPECTED_SPEAKERS:
        for digit in "0123456789":
            group = [item for item in recordings if item.speaker == speaker and item.digit == digit]
            if len(group) != 50 or {item.index for item in group} != set(range(50)):
                raise ValueError(f"expected indexes 0..49 for speaker={speaker}, digit={digit}")
    if validate_audio:
        for item in recordings:
            info = sf.info(item.path)
            if info.samplerate != EXPECTED_SAMPLE_RATE or info.channels != 1 or info.frames <= 0:
                raise ValueError(
                    f"audio contract failed for {item.name}: "
                    f"sr={info.samplerate}, channels={info.channels}, frames={info.frames}"
                )
    return recordings


def split_recordings(
    recordings: Iterable[Recording],
    spec: SplitSpec = SplitSpec(),
) -> dict[str, list[Recording]]:
    spec.validate()
    mapping = {
        **{speaker: "train" for speaker in spec.train_speakers},
        **{speaker: "dev" for speaker in spec.dev_speakers},
        **{speaker: "test" for speaker in spec.test_speakers},
    }
    splits: dict[str, list[Recording]] = {"train": [], "dev": [], "test": []}
    for item in recordings:
        try:
            splits[mapping[item.speaker]].append(item)
        except KeyError as exc:
            raise ValueError(f"recording speaker is not assigned to a split: {item.speaker}") from exc
    speaker_sets = {name: {item.speaker for item in rows} for name, rows in splits.items()}
    if speaker_sets["train"] & speaker_sets["dev"] or speaker_sets["train"] & speaker_sets["test"] or speaker_sets["dev"] & speaker_sets["test"]:
        raise AssertionError("speaker leakage after split construction")
    expected_counts = {"train": 2_000, "dev": 500, "test": 500}
    actual_counts = {name: len(rows) for name, rows in splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"split counts mismatch: expected {expected_counts}, got {actual_counts}")
    return splits


def build_manifest(splits: dict[str, list[Recording]], spec: SplitSpec) -> dict[str, object]:
    entries = []
    for split_name in ("train", "dev", "test"):
        for item in splits[split_name]:
            entries.append(
                {
                    "file": item.name,
                    "digit": item.digit,
                    "speaker": item.speaker,
                    "index": item.index,
                    "split": split_name,
                }
            )
    return {
        "format_version": 1,
        "dataset": "Free Spoken Digit Dataset",
        "upstream": "https://github.com/Jakobovski/free-spoken-digit-dataset",
        "revision": FSDD_REVISION,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC BY-SA 4.0",
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "split_spec": asdict(spec),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "entries": entries,
    }


def prepare_fsdd(cache_dir: Path, validate_audio: bool = True) -> tuple[Path, dict[str, list[Recording]]]:
    archive = download_archive(cache_dir)
    recordings_dir = extract_recordings(archive, cache_dir)
    recordings = scan_recordings(recordings_dir, validate_audio=validate_audio)
    return recordings_dir, split_recordings(recordings)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / ".local_data"
    recordings_dir, splits = prepare_fsdd(cache_dir)
    manifest = build_manifest(splits, SplitSpec())
    print(f"recordings: {recordings_dir}")
    print(json.dumps({k: manifest[k] for k in ["revision", "archive_sha256", "counts", "split_spec"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
