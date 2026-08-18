from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Callable, Sequence

from .tutor import QUESTIONS, load_progress, passed_ids


class ChallengeFailure(Exception):
    pass


@dataclass(frozen=True)
class Challenge:
    id: str
    title: str
    lessons: str
    hint: str
    validator: Callable[[ModuleType], None]


def expect_equal(actual: object, expected: object, case: str) -> None:
    if actual != expected:
        raise ChallengeFailure(f"{case}: 期望 {expected!r}，实际 {actual!r}")


def expect_close(actual: float, expected: float, case: str, tolerance: float = 1e-9) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise ChallengeFailure(f"{case}: 期望约 {expected}，实际 {actual}")


def validate_sampling(module: ModuleType) -> None:
    expect_equal(module.sample_count(8_000, 0.5), 4_000, "8kHz × 0.5s")
    expect_equal(module.sample_count(16_000, 1.0), 16_000, "16kHz × 1s")
    expect_equal(module.sample_count(16_000, 0.025), 400, "16kHz × 25ms")


def validate_framing(module: ModuleType) -> None:
    expect_equal(module.frame_count(4_000, 256, 80), 47, "N=4000,n_fft=256,hop=80")
    expect_equal(module.frame_count(256, 256, 80), 1, "刚好一窗")
    expect_equal(module.frame_count(416, 256, 80), 3, "刚好三窗")


def validate_ctc(module: ModuleType) -> None:
    expect_equal(module.ctc_collapse([8, 8, 0, 8]), [8, 8], "blank 分隔重复")
    expect_equal(module.ctc_collapse([1, 1, 2, 2, 0, 2]), [1, 2, 2], "重复与 blank")
    expect_equal(module.ctc_collapse([0, 0, 0]), [], "全 blank")


def validate_streaming_ctc(module: ModuleType) -> None:
    decoder = module.StreamingCTCCollapse(blank_id=0)
    expect_equal(decoder.accept([1, 1]), [1], "第一个 chunk")
    expect_equal(decoder.accept([1, 0]), [1], "边界重复不能新增 token")
    expect_equal(decoder.accept([1, 2, 2]), [1, 1, 2], "blank 后重复可新增")


def validate_bigram(module: ModuleType) -> None:
    expect_close(module.add_k_bigram_probability(10, 3, 5, 0.1), 3.1 / 10.5, "常见 bigram")
    expect_close(module.add_k_bigram_probability(0, 0, 4, 0.5), 0.25, "未见 context 的平滑")


def validate_rtf(module: ModuleType) -> None:
    expect_close(module.real_time_factor(0.1, 2.0), 0.05, "正常 RTF")
    try:
        module.real_time_factor(0.1, 0.0)
    except ValueError:
        pass
    else:
        raise ChallengeFailure("audio_seconds=0 时应抛出 ValueError")


CHALLENGES: tuple[Challenge, ...] = (
    Challenge("sampling", "采样点计算", "第1课", "采样点数等于每秒采样次数乘以秒数；注意返回整数。", validate_sampling),
    Challenge("framing", "完整帧数量", "第2课", "center=False：1 + floor((N-n_fft)/hop)。", validate_framing),
    Challenge("ctc", "CTC collapse", "第10、13课", "遍历类别时同时保存 previous；顺序是先判断 run，再移除 blank。", validate_ctc),
    Challenge("streaming-ctc", "跨 chunk CTC 状态", "第15～17课", "previous 不能是 accept() 内的局部变量；它属于会话。", validate_streaming_ctc),
    Challenge("bigram", "Add-k Bigram", "第19课", "分子给当前 bigram 加 k，分母给 V 个可能 token 各加 k。", validate_bigram),
    Challenge("rtf", "实时率", "第18课", "RTF=处理秒数/音频秒数；先检查分母。", validate_rtf),
)


def find_challenge(challenge_id: str) -> Challenge:
    try:
        return next(item for item in CHALLENGES if item.id == challenge_id)
    except StopIteration as exc:
        raise ValueError(f"未知编码关卡: {challenge_id}") from exc


def load_practice(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("asr_practice", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载练习文件: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_challenge(challenge: Challenge, module: ModuleType) -> tuple[bool, str]:
    try:
        challenge.validator(module)
    except NotImplementedError as exc:
        return False, str(exc)
    except (ChallengeFailure, AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, "全部隐藏样例通过"


def record_result(path: Path, challenge: Challenge, passed: bool, message: str) -> dict[str, object]:
    payload = load_progress(path)
    section = payload.setdefault("coding_challenges", {})
    if not isinstance(section, dict):
        raise ValueError("learning_progress.json 中 coding_challenges 必须是对象")
    attempts = section.setdefault("attempts", {})
    if not isinstance(attempts, dict):
        raise ValueError("coding_challenges.attempts 必须是对象")
    item = attempts.setdefault(challenge.id, {"count": 0, "passed": False})
    if not isinstance(item, dict):
        raise ValueError(f"coding_challenges.attempts.{challenge.id} 必须是对象")
    timestamp = datetime.now().isoformat(timespec="seconds")
    item["count"] = int(item.get("count", 0)) + 1
    item["last_passed"] = passed
    item["last_message"] = message
    item["updated_at"] = timestamp
    if passed:
        item["passed"] = True
        item["passed_at"] = timestamp
    section["passed"] = sum(
        bool(value.get("passed")) for value in attempts.values() if isinstance(value, dict)
    )
    section["total"] = len(CHALLENGES)
    section["updated_at"] = timestamp
    payload["updated_at"] = timestamp
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def coding_passed_ids(progress: dict[str, object]) -> set[str]:
    section = progress.get("coding_challenges", {})
    attempts = section.get("attempts", {}) if isinstance(section, dict) else {}
    if not isinstance(attempts, dict):
        return set()
    return {
        challenge_id
        for challenge_id, item in attempts.items()
        if isinstance(item, dict) and item.get("passed") is True
    }


def format_status(progress: dict[str, object]) -> str:
    knowledge = passed_ids(progress)
    coding = coding_passed_ids(progress)
    lines = [
        f"知识关卡：{len(knowledge)}/{len(QUESTIONS)}",
        f"编码关卡：{len(coding)}/{len(CHALLENGES)}",
    ]
    for challenge in CHALLENGES:
        lines.append(f"  {'✓' if challenge.id in coding else '·'} {challenge.id} — {challenge.title}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR 渐进编码关卡验证器")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list", action="store_true", help="列出编码关卡")
    actions.add_argument("--status", action="store_true", help="显示知识和编码进度")
    actions.add_argument("--hint", metavar="ID", help="显示一关提示")
    actions.add_argument("--check", metavar="ID", help="验证一关")
    actions.add_argument("--check-all", action="store_true", help="依次验证全部关卡")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    practice_path = root / "learning_workspace" / "asr_practice.py"
    progress_path = root / "learning_progress.json"
    progress = load_progress(progress_path)
    if args.list:
        for item in CHALLENGES:
            print(f"{item.id:14s} {item.title}（{item.lessons}）")
        return 0
    if args.status:
        print(format_status(progress))
        return 0
    if args.hint:
        item = find_challenge(args.hint)
        print(f"{item.title}｜{item.lessons}\n提示：{item.hint}")
        return 0

    module = load_practice(practice_path)
    selected = CHALLENGES if args.check_all else (find_challenge(args.check),)
    all_passed = True
    for item in selected:
        passed, message = check_challenge(item, module)
        record_result(progress_path, item, passed, message)
        print(f"{'通过' if passed else '未通过'} {item.id}: {message}")
        if not passed:
            print(f"提示：{item.hint}")
            all_passed = False
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
