from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Sequence

import torch


@dataclass(frozen=True)
class Question:
    id: str
    stage: str
    lessons: str
    prompt: str
    choices: tuple[str, ...]
    accepted: tuple[str, ...]
    hints: tuple[str, ...]
    explanation: str


QUESTIONS: tuple[Question, ...] = (
    Question(
        id="sampling-count",
        stage="声音与采样",
        lessons="第1课",
        prompt="8,000 Hz 的录音持续 0.5 秒，一共有多少个采样点？",
        choices=(),
        accepted=("4000", "4000个", "4000个采样点"),
        hints=("采样点数 = 每秒采样点数 × 秒数。", "计算 8000 × 0.5。"),
        explanation="8,000 Hz 表示每秒记录 8,000 次；0.5 秒记录 4,000 次。",
    ),
    Question(
        id="sampling-value",
        stage="声音与采样",
        lessons="第1课",
        prompt="每个 PCM 采样值主要表示什么？",
        choices=("A. 整段声音的频率", "B. 某一时刻的振幅", "C. 一个文字 token"),
        accepted=("b", "b.", "振幅", "某一时刻的振幅", "幅度", "amplitude"),
        hints=("采样是在不同时间点读取同一种物理量。", "波形图的纵轴是什么？"),
        explanation="一个采样值是某一时刻的信号振幅；频率要从一段随时间变化的采样中分析。",
    ),
    Question(
        id="logmel-frames",
        stage="Log-Mel",
        lessons="第2～6课",
        prompt="center=False 时，N=4000、n_fft=256、hop=80 会产生多少帧？",
        choices=(),
        accepted=("47", "47帧"),
        hints=("公式是 1 + floor((N - n_fft) / hop)。", "先算 (4000 - 256) / 80，再向下取整并加 1。"),
        explanation="1 + floor((4000-256)/80) = 47；边缘没有自动中心填充。",
    ),
    Question(
        id="tensor-time-axis",
        stage="张量与编码器",
        lessons="第7～9课",
        prompt="声学特征 [B,T,F] 中，T 表示什么？",
        choices=("A. batch 数量", "B. 时间帧", "C. Mel/feature 维"),
        accepted=("b", "b.", "时间", "时间帧", "time"),
        hints=("B 是 batch，F 是 feature。", "语音从左到右展开的轴是哪一个？"),
        explanation="T 是时间帧数；每个时间位置有 F 维声学特征。",
    ),
    Question(
        id="ctc-collapse",
        stage="CTC",
        lessons="第10～13课",
        prompt="blank=0，类别 8 对应数字 7；路径 [8,8,0,8] 经 CTC collapse 后是什么？",
        choices=(),
        accepted=("77",),
        hints=("先合并相邻重复，再删除 blank。", "blank 把前后的两个类别 8 分成了两个 run。"),
        explanation="[8,8,0,8] → [8,0,8] → 77；blank 允许输出连续相同 token。",
    ),
    Question(
        id="ctc-classes",
        stage="CTC",
        lessons="第10～14课",
        prompt="识别数字 0～9 的 CTC head 为什么输出 11 类？",
        choices=("A. 10个数字+1个blank", "B. 需要一个未知说话人类", "C. 时间轴固定为11帧"),
        accepted=("a", "a.", "10个数字+1个blank", "数字加blank", "blank"),
        hints=("CTC 有一个不直接显示为文字的特殊类别。", "10 + 1 等于多少？"),
        explanation="10 个数字 token 加 1 个 CTC blank，共 11 类；类别数与时间帧数无关。",
    ),
    Question(
        id="streaming-state",
        stage="流式ASR",
        lessons="第15～18、24课",
        prompt="两个 WebSocket 客户端可以共享同一份前端/model/decoder cache 吗？",
        choices=("A. 可以，能节省内存", "B. 不可以，每个会话必须隔离", "C. 只要采样率相同就可以"),
        accepted=("b", "b.", "不可以", "不能", "每个会话隔离", "会话隔离"),
        hints=("想象两个用户的音频 chunk 交错到达。", "cache 代表哪一位用户的过去？"),
        explanation="不能共享。STFT buffer、模型 cache、CTC prefix 都属于单个会话，否则音频历史会串线。",
    ),
    Question(
        id="lm-boundary",
        stage="语言模型",
        lessons="第19～23课",
        prompt="声学证据没有支持某个词时，应该让语言模型无条件把它补出来吗？",
        choices=("A. 应该，常见句子优先", "B. 不应该，LM只能重排声学候选", "C. 热词开启时总是应该"),
        accepted=("b", "b.", "不应该", "不能", "lm只能重排", "语言模型只能重排"),
        hints=("语言模型知道什么常见，却不知道用户实际说了什么。", "LM 权重过大会造成哪类幻觉式纠错？"),
        explanation="LM/热词是先验，只能在声学候选之间调分；压过声学证据会产生幻觉式替换。",
    ),
    Question(
        id="generalization-evidence",
        stage="评测边界",
        lessons="第14、18、30课",
        prompt="教学模型在22条训练样本上22/22，能否证明它能识别新说话人？",
        choices=("A. 能", "B. 不能，还需speaker-disjoint测试"),
        accepted=("b", "b.", "不能", "不可以", "需要speaker-disjoint测试", "需要独立测试"),
        hints=("这些样本参与了参数优化。", "训练集效果与独立测试集效果是不是同一证据？"),
        explanation="不能。训练集 22/22 只证明闭环能过拟合；泛化必须用未参与训练/调参的说话人独立测试。",
    ),
    Question(
        id="sample-rate-contract",
        stage="部署契约",
        lessons="第28～31课",
        prompt="模型要求8 kHz，客户端发送16 kHz但未声明重采样时，服务应该怎样处理？",
        choices=("A. 静默当作8 kHz", "B. 明确拒绝或走版本一致的重采样", "C. 丢掉一半随机采样点"),
        accepted=("b", "b.", "拒绝", "明确拒绝", "重采样", "明确拒绝或重采样"),
        hints=("同样的样本序列按不同采样率解释，时间和频率都会变化。", "接口不确定时，静默猜测还是明确失败更容易排错？"),
        explanation="应明确拒绝，或进入经过训练一致性验证的重采样路径；绝不能静默错读。",
    ),
    Question(
        id="rtf",
        stage="性能",
        lessons="第18、30课",
        prompt="处理2秒音频耗时0.1秒，RTF是多少？",
        choices=(),
        accepted=("0.05", ".05"),
        hints=("RTF = 处理时间 / 音频时长。", "计算 0.1 / 2。"),
        explanation="RTF=0.05，表示平均计算速度能跟上实时；它不等于最终文字延迟。",
    ),
)


def normalize_answer(answer: str) -> str:
    return "".join(answer.strip().lower().split()).replace("。", "").replace("，", "")


def is_correct(question: Question, answer: str) -> bool:
    normalized = normalize_answer(answer)
    return normalized in {normalize_answer(item) for item in question.accepted}


def load_progress(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"无法读取学习进度 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("learning_progress.json 顶层必须是对象")
    return payload


def record_attempt(path: Path, question: Question, answer: str, correct: bool) -> dict[str, object]:
    payload = load_progress(path)
    tutor = payload.setdefault("tutor", {})
    if not isinstance(tutor, dict):
        raise ValueError("learning_progress.json 中 tutor 字段必须是对象")
    attempts = tutor.setdefault("attempts", {})
    if not isinstance(attempts, dict):
        raise ValueError("tutor.attempts 必须是对象")
    item = attempts.setdefault(question.id, {"count": 0, "passed": False})
    if not isinstance(item, dict):
        raise ValueError(f"tutor.attempts.{question.id} 必须是对象")
    item["count"] = int(item.get("count", 0)) + 1
    item["last_answer"] = answer
    item["last_correct"] = correct
    item["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if correct:
        item["passed"] = True
        item["passed_at"] = item["updated_at"]
    tutor["passed"] = sum(
        bool(value.get("passed"))
        for value in attempts.values()
        if isinstance(value, dict)
    )
    tutor["total"] = len(QUESTIONS)
    tutor["updated_at"] = item["updated_at"]
    payload["updated_at"] = item["updated_at"]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def passed_ids(progress: dict[str, object]) -> set[str]:
    tutor = progress.get("tutor", {})
    attempts = tutor.get("attempts", {}) if isinstance(tutor, dict) else {}
    if not isinstance(attempts, dict):
        return set()
    return {
        question_id
        for question_id, value in attempts.items()
        if isinstance(value, dict) and value.get("passed") is True
    }


def current_question(progress: dict[str, object]) -> Question | None:
    completed = passed_ids(progress)
    return next((question for question in QUESTIONS if question.id not in completed), None)


def show_demo(question: Question, root: Path) -> None:
    if question.id == "sampling-count":
        print("实验：8000 samples/s × 0.5 s = ? samples")
    elif question.id == "logmel-frames":
        from .features import LogMelConfig, LogMelFrontend, load_mono_audio

        config = LogMelConfig(sample_rate=8_000, n_fft=256, win_length=200, hop_length=80, n_mels=24)
        waveform = load_mono_audio(root / "data" / "spoken_digits_parts" / "7_jackson_0.wav", 8_000)
        features = LogMelFrontend(config)(waveform)
        print(f"真实数字7：waveform={tuple(waveform.shape)} → Log-Mel={tuple(features.shape)}")
    elif question.id == "ctc-collapse":
        from .decoder import ctc_collapse

        print("实验路径：", [8, 8, 0, 8], "→", ctc_collapse([8, 8, 0, 8], tuple("0123456789"))[0])
    elif question.id == "streaming-state":
        from .streaming import StreamingAcousticEngine

        engine = StreamingAcousticEngine.load(root / "artifacts" / "streaming_digit_ctc.pt")
        audio = root / "data" / "spoken_digits_parts" / "7_jackson_0.wav"
        results = {size: engine.recognize_file(audio, chunk_samples=size).text for size in (400, 800, 1600)}
        print("同一音频，不同 chunk：", results)
    elif question.id == "lm-boundary":
        from .decoder import prefix_beam_search
        from .language_model import AddKBigramLanguageModel, ShallowFusionScorer

        logits = torch.tensor([[torch.log(torch.tensor(0.01)), torch.log(torch.tensor(0.51)), torch.log(torch.tensor(0.48))]])
        acoustic = prefix_beam_search(logits, ("0", "1"), beam_size=3)[0].text
        lm = AddKBigramLanguageModel.fit(["0"] + ["1"] * 10, ("0", "1"))
        fused = prefix_beam_search(
            logits,
            ("0", "1"),
            beam_size=3,
            extension_scorer=ShallowFusionScorer(language_model=lm, lm_weight=1.0),
        )[0].text
        print(f"可控歧义：纯声学={acoustic}；加入领域LM={fused}")


def format_status(progress: dict[str, object]) -> str:
    completed = passed_ids(progress)
    lines = [f"交互教练进度：{len(completed)}/{len(QUESTIONS)}"]
    for question in QUESTIONS:
        marker = "✓" if question.id in completed else "·"
        lines.append(f"  {marker} {question.stage} — {question.id}")
    return "\n".join(lines)


def find_question(question_id: str) -> Question:
    try:
        return next(question for question in QUESTIONS if question.id == question_id)
    except StopIteration as exc:
        raise ValueError(f"未知题目: {question_id}") from exc


def run_question(question: Question, answer: str, progress_path: Path) -> bool:
    correct = is_correct(question, answer)
    payload = record_attempt(progress_path, question, answer, correct)
    attempts = payload["tutor"]["attempts"][question.id]["count"]
    if correct:
        print("正确。", question.explanation)
        print(f"已保存进度：{progress_path}")
        return True
    hint_index = min(int(attempts) - 1, len(question.hints) - 1)
    print("还不对。提示：", question.hints[hint_index])
    if int(attempts) >= len(question.hints) + 1:
        print("讲解：", question.explanation)
        print("这题仍未通关，请理解后再次作答。")
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="零基础 ASR 小步交互教练")
    parser.add_argument("--status", action="store_true", help="显示已保存进度")
    parser.add_argument("--question", help="指定题目 ID；默认继续第一道未通过题")
    parser.add_argument("--answer", help="非交互提交答案")
    parser.add_argument("--no-demo", action="store_true", help="不运行题目前的小实验")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    progress_path = root / "learning_progress.json"
    progress = load_progress(progress_path)
    if args.status:
        print(format_status(progress))
        return 0
    question = find_question(args.question) if args.question else current_question(progress)
    if question is None:
        print(format_status(progress))
        print("知识检查已通过；下一步仍需完成 README 中的闭卷编码、故障注入和迁移任务。")
        return 0

    print(f"\n[{question.stage}] 前置：{question.lessons}")
    if not args.no_demo:
        show_demo(question, root)
    print("\n问题：", question.prompt)
    for choice in question.choices:
        print(" ", choice)
    answer = args.answer if args.answer is not None else input("你的答案：")
    return 0 if run_question(question, answer, progress_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
