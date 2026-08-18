from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = ROOT / "data" / "audio_diagnosis_lab"
ANSWER_PATH = LAB_DIR / "answer_key.json"


ALIASES = {
    "normal": "clean",
    "no_fault": "clean",
    "正常": "clean",
    "无明显故障": "clean",
    "正常立体声": "clean_stereo",
    "quiet": "low_level",
    "low_gain": "low_level",
    "声音小": "low_level",
    "幅值低": "low_level",
    "clip": "hard_clipping",
    "clipping": "hard_clipping",
    "削波": "hard_clipping",
    "asymmetric_clip": "asymmetric_clipping",
    "不对称削波": "asymmetric_clipping",
    "dc": "dc_offset",
    "直流偏移": "dc_offset",
    "hum": "mains_hum_50hz",
    "工频嗡声": "mains_hum_50hz",
    "white_noise": "broadband_white_noise",
    "白噪声": "broadband_white_noise",
    "whistle": "narrowband_whistle_2800hz",
    "啸叫": "narrowband_whistle_2800hz",
    "clicks": "impulsive_clicks",
    "爆音": "impulsive_clicks",
    "dropout": "dropouts",
    "丢帧": "dropouts",
    "echo": "single_echo_120ms",
    "回声": "single_echo_120ms",
    "reverb": "reverberation",
    "混响": "reverberation",
    "lowpass": "lowpass_muffled",
    "低通": "lowpass_muffled",
    "闷": "lowpass_muffled",
    "highpass": "highpass_thin",
    "高通": "highpass_thin",
    "quantization": "low_bit_quantization",
    "量化": "low_bit_quantization",
    "pumping": "gain_pumping",
    "增益抽吸": "gain_pumping",
    "wrong_sample_rate": "wrong_sample_rate_metadata",
    "采样率错误": "wrong_sample_rate_metadata",
    "silent_channel": "right_channel_silent",
    "单边无声": "right_channel_silent",
    "polarity": "stereo_polarity_inversion",
    "反相": "stereo_polarity_inversion",
    "delay": "stereo_channel_delay_2ms",
    "通道延迟": "stereo_channel_delay_2ms",
    "truncated": "truncated_boundaries",
    "截断": "truncated_boundaries",
    "aliasing": "naive_resampling_aliasing",
    "混叠": "naive_resampling_aliasing",
    "compression": "heavy_nonlinear_compression",
    "重压缩": "heavy_nonlinear_compression",
}


def normalize_case(raw: str) -> str:
    match = re.fullmatch(r"(?:case[_-]?)?(\d{1,2})", raw.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("case must look like 01 or case_01")
    number = int(match.group(1))
    if not 1 <= number <= 24:
        raise ValueError("case number must be 01..24")
    return f"case_{number:02d}"


def normalize_guess(raw: str) -> str:
    value = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(value, value)


def load_cases() -> dict[str, dict]:
    payload = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in payload["cases"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check blind audio-diagnosis practice answers.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List case IDs by difficulty without revealing answers.")
    group.add_argument("--categories", action="store_true", help="List allowed primary diagnoses without revealing case mapping.")
    group.add_argument("--hint", metavar="CASE", help="Show the first diagnostic action, not the answer.")
    group.add_argument("--reveal", metavar="CASE", help="Reveal answer and evidence after making a written attempt.")
    group.add_argument("--check", nargs=2, metavar=("CASE", "GUESS"), help="Check one diagnosis, e.g. --check 03 clipping")
    args = parser.parse_args()
    cases = load_cases()

    if args.list:
        for level in ("beginner", "intermediate", "advanced"):
            selected=[case_id for case_id,item in cases.items() if item["difficulty"]==level]
            print(f"{level}: {' '.join(selected)}")
        return 0

    if args.categories:
        for issue in sorted({item["issue"] for item in cases.values()}):
            print(issue)
        return 0

    raw_case = args.hint or args.reveal or args.check[0]
    try:
        case_id = normalize_case(raw_case)
    except ValueError as exc:
        parser.error(str(exc))
    item = cases[case_id]

    if args.hint:
        print(f"{case_id} hint: first check -> {item['first_checks'][0]}")
        print("Write down metadata, waveform, spectrum/spectrogram and channel evidence before another hint.")
        return 0

    if args.reveal:
        print(f"{case_id}: {item['issue']} ({item['severity']})")
        print("Audible:", item["audible"])
        print("Visible:", item["visible"])
        print("Why:", item["explanation"])
        print("First checks:", " -> ".join(item["first_checks"]))
        return 0

    guess = normalize_guess(args.check[1])
    expected = item["issue"]
    clean_equivalents = {"clean", "clean_stereo"}
    correct = guess == expected or guess in clean_equivalents and expected in clean_equivalents
    if correct:
        print(f"CORRECT: {case_id} -> {expected}")
        print("Evidence:", item["visible"])
        print("Boundary:", item["explanation"])
        return 0
    print(f"NOT YET: {case_id} is not best classified as {guess!r}.")
    print("Next check:", item["first_checks"][0])
    print(f"Use --hint {case_id} or, after writing evidence, --reveal {case_id}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
