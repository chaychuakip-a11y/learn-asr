from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile

import soundfile as sf


AUDIOMNIST_REVISION = "630d7dab4c040882834de6fa21baf9a60372accd"
ARCHIVE_URL = f"https://codeload.github.com/soerenab/AudioMNIST/zip/{AUDIOMNIST_REVISION}"
ARCHIVE_SHA256 = "f0420069d684baff1688658d1ba53316f9b3f742a7d0ab2963a064c591f9c573"
ARCHIVE_BYTES = 996_621_372
EXPECTED_AUDIO_BLOB_BYTES = 1_851_988_532
EXPECTED_SPEAKERS = tuple(f"{index:02d}" for index in range(1, 61))
EXPECTED_RECORDINGS = 30_000
EXPECTED_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class AudioMnistRecording:
    path: Path
    digit: str
    speaker: str
    index: int

    @property
    def relative_name(self) -> str:
        return f"{self.speaker}/{self.path.name}"


@dataclass(frozen=True)
class SpeakerMetadata:
    speaker: str
    accent: str
    age: int
    gender: str
    native_speaker: bool
    origin: str
    recording_date: str
    recording_room: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"audiomnist-{AUDIOMNIST_REVISION[:8]}.zip"
    if archive.exists():
        if archive.stat().st_size != ARCHIVE_BYTES:
            raise ValueError(
                f"AudioMNIST archive size mismatch: expected {ARCHIVE_BYTES}, "
                f"got {archive.stat().st_size}"
            )
        actual = sha256(archive)
        if actual != ARCHIVE_SHA256:
            raise ValueError(
                f"AudioMNIST archive SHA256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
            )
        return archive

    partial = archive.with_suffix(".zip.part")
    if partial.exists():
        raise FileExistsError(f"partial AudioMNIST download needs inspection: {partial}")
    request = Request(ARCHIVE_URL, headers={"User-Agent": "learn-asr-course/1.0"})
    with urlopen(request, timeout=120) as response, partial.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size != ARCHIVE_BYTES or sha256(partial) != ARCHIVE_SHA256:
        raise ValueError("downloaded AudioMNIST archive failed frozen size/SHA256 checks")
    os.replace(partial, archive)
    return archive


def _selected_member(member_name: str) -> Path | None:
    parts = Path(member_name).parts
    if len(parts) == 4 and parts[1] == "data" and parts[2] in EXPECTED_SPEAKERS:
        filename = parts[3]
        pieces = Path(filename).stem.split("_")
        if Path(filename).suffix.lower() == ".wav" and len(pieces) == 3:
            return Path("data") / parts[2] / filename
    if len(parts) == 3 and parts[1:] == ("data", "audioMNIST_meta.txt"):
        return Path("audioMNIST_meta.json")
    if len(parts) == 2 and parts[1] == "LICENSE":
        return Path("UPSTREAM_LICENSE")
    if len(parts) == 2 and parts[1] == "README.md":
        return Path("UPSTREAM_README.md")
    return None


def extract_dataset(archive: Path, cache_dir: Path) -> Path:
    dataset_root = cache_dir / f"audiomnist-{AUDIOMNIST_REVISION[:8]}"
    data_dir = dataset_root / "data"
    if data_dir.exists():
        if len(list(data_dir.glob("*/*.wav"))) != EXPECTED_RECORDINGS:
            raise ValueError(f"existing AudioMNIST extraction is incomplete: {dataset_root}")
        return dataset_root
    if dataset_root.exists():
        raise FileExistsError(
            f"AudioMNIST target exists without a valid data directory: {dataset_root}"
        )

    with tempfile.TemporaryDirectory(prefix="audiomnist-extract-", dir=cache_dir) as temp:
        temporary_root = Path(temp) / "dataset"
        temporary_root.mkdir()
        selected_wavs = 0
        selected_bytes = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = _selected_member(member.filename)
                if relative is None:
                    continue
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"duplicate AudioMNIST archive member: {relative}")
                with bundle.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if destination.suffix.lower() == ".wav":
                    selected_wavs += 1
                    selected_bytes += member.file_size
        if selected_wavs != EXPECTED_RECORDINGS:
            raise ValueError(
                f"archive supplied {selected_wavs} AudioMNIST WAVs, expected {EXPECTED_RECORDINGS}"
            )
        if selected_bytes != EXPECTED_AUDIO_BLOB_BYTES:
            raise ValueError(
                f"AudioMNIST WAV bytes mismatch: expected {EXPECTED_AUDIO_BLOB_BYTES}, "
                f"got {selected_bytes}"
            )
        temporary_root.rename(dataset_root)
    return dataset_root


def parse_recording(path: Path) -> AudioMnistRecording:
    pieces = path.stem.split("_")
    if len(pieces) != 3:
        raise ValueError(f"unexpected AudioMNIST filename: {path.name}")
    digit, speaker, index_text = pieces
    if digit not in "0123456789" or speaker not in EXPECTED_SPEAKERS:
        raise ValueError(f"unexpected AudioMNIST digit or speaker: {path.name}")
    if path.parent.name != speaker:
        raise ValueError(
            f"AudioMNIST filename speaker {speaker} disagrees with directory {path.parent.name}"
        )
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"unexpected AudioMNIST repetition index: {path.name}") from exc
    if index not in range(50):
        raise ValueError(f"AudioMNIST repetition index must be 0..49: {path.name}")
    return AudioMnistRecording(path, digit, speaker, index)


def scan_recordings(dataset_root: Path, validate_audio: bool = True) -> list[AudioMnistRecording]:
    rows = [
        parse_recording(path)
        for path in sorted((dataset_root / "data").glob("*/*.wav"))
    ]
    if len(rows) != EXPECTED_RECORDINGS:
        raise ValueError(f"found {len(rows)} AudioMNIST recordings, expected {EXPECTED_RECORDINGS}")
    if len({row.relative_name for row in rows}) != EXPECTED_RECORDINGS:
        raise ValueError("duplicate AudioMNIST relative filenames")
    for speaker in EXPECTED_SPEAKERS:
        for digit in "0123456789":
            group = [row for row in rows if row.speaker == speaker and row.digit == digit]
            if len(group) != 50 or {row.index for row in group} != set(range(50)):
                raise ValueError(
                    f"expected repetitions 0..49 for speaker={speaker}, digit={digit}"
                )
    if validate_audio:
        for position, row in enumerate(rows, start=1):
            info = sf.info(row.path)
            if (
                info.samplerate != EXPECTED_SAMPLE_RATE
                or info.channels != 1
                or info.subtype != "PCM_16"
                or info.frames <= 0
            ):
                raise ValueError(
                    f"audio contract failed for {row.relative_name}: sr={info.samplerate}, "
                    f"channels={info.channels}, subtype={info.subtype}, frames={info.frames}"
                )
            if position % 5_000 == 0:
                print(f"validated audio headers: {position}/{EXPECTED_RECORDINGS}", flush=True)
    return rows


def load_metadata(dataset_root: Path) -> dict[str, SpeakerMetadata]:
    raw = json.loads((dataset_root / "audioMNIST_meta.json").read_text(encoding="utf-8"))
    if set(raw) != set(EXPECTED_SPEAKERS):
        raise ValueError("AudioMNIST metadata must cover all 60 speakers")
    metadata = {}
    for speaker in EXPECTED_SPEAKERS:
        row = raw[speaker]
        native_text = str(row["native speaker"]).strip().lower()
        if native_text not in {"yes", "no"}:
            raise ValueError(f"unexpected native-speaker value for {speaker}: {native_text}")
        metadata[speaker] = SpeakerMetadata(
            speaker=speaker,
            accent=str(row["accent"]).strip(),
            age=int(row["age"]),
            gender=str(row["gender"]).strip().lower(),
            native_speaker=native_text == "yes",
            origin=str(row["origin"]).strip(),
            recording_date=str(row["recordingdate"]).strip(),
            recording_room=str(row["recordingroom"]).strip(),
        )
    return metadata


def build_manifest(
    recordings: list[AudioMnistRecording],
    metadata: dict[str, SpeakerMetadata],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "dataset": "AudioMNIST",
        "official_repository": "https://github.com/soerenab/AudioMNIST",
        "revision": AUDIOMNIST_REVISION,
        "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "MIT",
        "recording_count": len(recordings),
        "speaker_count": len(metadata),
        "sample_rate": EXPECTED_SAMPLE_RATE,
        "speakers": {key: asdict(value) for key, value in metadata.items()},
        "entries": [
            {
                "file": row.relative_name,
                "digit": row.digit,
                "speaker": row.speaker,
                "index": row.index,
                "bytes": row.path.stat().st_size,
                "frames": sf.info(row.path).frames,
            }
            for row in recordings
        ],
    }


def prepare_audiomnist(
    cache_dir: Path,
    validate_audio: bool = True,
) -> tuple[Path, list[AudioMnistRecording], dict[str, SpeakerMetadata]]:
    archive = download_archive(cache_dir)
    dataset_root = extract_dataset(archive, cache_dir)
    recordings = scan_recordings(dataset_root, validate_audio=validate_audio)
    metadata = load_metadata(dataset_root)
    return dataset_root, recordings, metadata


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root, recordings, metadata = prepare_audiomnist(root / ".local_data")
    manifest = build_manifest(recordings, metadata)
    output = root / "artifacts" / "audiomnist_external_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dataset_root:", dataset_root)
    print("archive_sha256:", ARCHIVE_SHA256)
    print("recordings:", len(recordings), "speakers:", len(metadata))
    print("manifest:", output)


if __name__ == "__main__":
    main()
