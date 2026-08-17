"""Build the semantic/post-processing intensive notebook.

Run with:
    uv run python scripts/build_semantic_lab.py
"""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_语义后处理实验室_时间戳ITN置信度与安全执行.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # 语义后处理综合实验室：从时间戳到安全执行

        声学模型给出的不是最终产品，而是一组带时间、分数和不确定性的假设。真正的系统还要回答：

        - 每个词什么时候说的，谁说的？
        - “二十六度”何时应写成 `26°C`，原始证据是否仍被保留？
        - 模型说自己有 90% 把握，长期看真的约有 90% 正确吗？
        - N-best 什么时候能救回同音错误，什么时候会被语义“带偏”？
        - 意图和槽位通过后，系统是否就可以直接执行？
        - LLM 怎样帮助整理信息，又怎样被禁止篡改听觉证据？

        本实验贯通第 37～41 课。我们会使用真实开源多说话人 FSDD 音频、可控置信度仿真和中文业务文本小型黄金集。三种证据来源会明确标注，不混为一谈。
        """
    ),
    md(
        """
        ## 最重要的分层：事实、变换、解释、行动

        | 层 | 例子 | 能否覆盖上一层 |
        |---|---|---|
        | 原始证据 | 音频、模型版本、N-best、分数、时间戳 | 不可变 |
        | 确定性变换 | 标点、ITN、说话人标签 | 另存结果，不能覆盖原文 |
        | 语义解释 | intent、slots、摘要 | 必须能追溯证据 |
        | 行动决定 | execute / clarify / reject | 还要经过权限、范围与风险策略 |

        核心原则：**“听起来合理”不等于“音频确实这样说”。**

        通过标准：所有自动断言通过，末尾 56 分测试至少 45 分，并能口述一条请求如何从音频证据走到安全行动。
        """
    ),
    code(
        """
        from pathlib import Path
        from collections import defaultdict
        from dataclasses import dataclass, asdict
        import json
        import math
        import re

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import soundfile as sf
        import librosa
        import ipywidgets as widgets
        from IPython.display import Audio, display

        installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
        for font_name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]:
            if font_name in installed_fonts:
                plt.rcParams["font.sans-serif"] = [font_name]
                break
        plt.rcParams["axes.unicode_minus"] = False

        candidates = [Path.cwd(), Path.cwd().parent]
        ROOT = next(p for p in candidates if (p / "pyproject.toml").exists())
        MULTI_DIR = ROOT / "data" / "fsdd_multispeaker"
        audio_files = sorted(MULTI_DIR.glob("*.wav"))
        print(f"项目根目录: {ROOT}")
        print(f"真实多说话人文件: {len(audio_files)} 个")
        assert len(audio_files) == 12
        """
    ),
    md(
        """
        ## 1. 时间戳不是一个数字，而是一条坐标变换链

        常见时间坐标包括：

        1. 原始输入 sample index；
        2. 特征 frame index，间隔由 hop 决定；
        3. encoder step，可能经过 4× 或 8× 下采样；
        4. CTC spike / RNN-T emission step；
        5. word/segment 边界。

        若特征 hop 为 10 ms、encoder 下采样 4×，相邻 encoder step 对应约 40 ms。`step × 40 ms` 只是坐标映射；CTC spike 是模型最愿意发 token 的位置，不保证等于人类听感上的音素起点。

        流式系统必须把裁剪、padding、右上下文和累计输入样本纳入映射。最稳妥的主时钟是**原始输入样本计数**。
        """
    ),
    code(
        """
        def encoder_step_to_time(step, feature_hop_ms=10, subsampling=4, stream_offset_samples=0, sr=16000):
            local_ms = step * feature_hop_ms * subsampling
            return stream_offset_samples / sr + local_ms / 1000

        token_steps = np.array([3, 7, 12, 18])
        tokens = ["打", "开", "客厅", "灯"]

        def show_time_mapping(subsampling=4, feature_hop_ms=10, right_context_ms=160):
            times = np.array([encoder_step_to_time(s, feature_hop_ms, subsampling) for s in token_steps])
            uncertainty = right_context_ms / 1000
            fig, ax = plt.subplots(figsize=(11, 2.8))
            ax.scatter(times, np.ones_like(times), s=70, zorder=3)
            for token, time_s in zip(tokens, times):
                ax.text(time_s, 1.06, token, ha="center", fontsize=12)
                ax.hlines(1, max(0, time_s-uncertainty), time_s+uncertainty, color="C0", alpha=.25, lw=8)
            ax.set(xlabel="映射到输入音频的时间 (s)", yticks=[], ylim=(.8, 1.25),
                   title="点是 emission step；粗线只是上下文影响范围示意，不是真实词边界")
            ax.grid(axis="x", alpha=.25); plt.tight_layout(); plt.show()
            print(dict(zip(tokens, np.round(times, 3))))

        time_ui = widgets.interactive(
            show_time_mapping,
            subsampling=widgets.IntSlider(min=1, max=8, step=1, value=4, description="下采样", continuous_update=False),
            feature_hop_ms=widgets.IntSlider(min=5, max=20, step=5, value=10, description="hop ms", continuous_update=False),
            right_context_ms=widgets.IntSlider(min=0, max=500, step=20, value=160, description="右上下文", continuous_update=False),
        )
        display(time_ui)

        assert abs(encoder_step_to_time(10, 10, 4) - .4) < 1e-12
        assert abs(encoder_step_to_time(10, 10, 4, 16000) - 1.4) < 1e-12
        """
    ),
    md(
        """
        ### 时间戳检查题（每题 2 分）

        1. hop 从 10 ms 改为 20 ms、其他不变，同一个 step 的时间如何变化？
        2. 第二个流式 chunk 从原音频第 32,000 个 sample 开始，为什么不能把它的 step 0 标成 0 秒？
        3. CTC spike 能否直接宣称是词的精确起点？若产品需要字幕高亮，应补什么？

        <details><summary>参考答案</summary>

        1. 映射时间加倍。2. 它的全局起点是 2 秒，必须加 stream offset。3. 不能；spike 是发射位置。可使用 forced alignment、边界模型或经过验证的扩展规则，并报告时间戳误差分布。

        </details>
        """
    ),
    md(
        """
        ## 2. 真实多说话人实验：speaker recognition 不等于 diarization

        文件命名是 `{digit}_{speaker}_{index}.wav`。我们用数字 0、1 建立每人的 enrollment prototype，用从未参与 enrollment 的数字 2 测试。这比拿同一条录音同时注册和测试更诚实。

        实验只做**已知边界的 segment-level speaker classification**。完整 diarization 还要从连续音频中发现语音边界、聚类未知说话人、处理重叠语音，并决定相邻段是否合并。
        """
    ),
    code(
        """
        def speaker_embedding(path):
            y, sample_rate = sf.read(path, dtype="float32")
            mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=20, n_fft=256, hop_length=80)
            embedding = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            return embedding / (np.linalg.norm(embedding) + 1e-9)

        recordings = defaultdict(dict)
        for path in audio_files:
            digit, speaker, _ = path.stem.split("_")
            recordings[speaker][digit] = path

        prototypes = {}
        for speaker, items in recordings.items():
            enrollment = np.mean([speaker_embedding(items[d]) for d in ["0", "1"]], axis=0)
            prototypes[speaker] = enrollment / (np.linalg.norm(enrollment) + 1e-9)

        speakers = sorted(prototypes)
        similarity = np.zeros((len(speakers), len(speakers)))
        predictions = {}
        margins = {}
        for row, true_speaker in enumerate(speakers):
            query = speaker_embedding(recordings[true_speaker]["2"])
            scores = {speaker: float(query @ prototypes[speaker]) for speaker in speakers}
            ranking = sorted(scores, key=scores.get, reverse=True)
            predictions[true_speaker] = ranking[0]
            margins[true_speaker] = scores[ranking[0]] - scores[ranking[1]]
            similarity[row] = [scores[speaker] for speaker in speakers]

        accuracy = np.mean([predictions[s] == s for s in speakers])
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(similarity, vmin=.95, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(speakers)), speakers, rotation=25)
        ax.set_yticks(range(len(speakers)), speakers)
        ax.set(xlabel="enrollment prototype", ylabel="数字 2 的真实说话人", title="余弦相似度：最高格不等于一定可信")
        for i in range(len(speakers)):
            for j in range(len(speakers)):
                ax.text(j, i, f"{similarity[i,j]:.3f}", ha="center", va="center", color="white")
        plt.colorbar(image, ax=ax, label="cosine similarity")
        plt.tight_layout(); plt.show()

        print("测试准确率:", accuracy)
        for speaker in speakers:
            print(f"{speaker:9s} -> {predictions[speaker]:9s}; top1-top2 margin={margins[speaker]:.6f}")
        print("注意：george 的 margin 极小，虽然 top-1 正确，证据仍很脆弱。")
        assert accuracy == 1.0
        """
    ),
    code(
        """
        # 将四个真实 test 文件拼成带已知边界的教学时间轴。
        timeline_audio = []
        reference_segments = []
        cursor = 0
        silence_sr = 8000
        silence = np.zeros(int(.18 * silence_sr), dtype=np.float32)
        for speaker in speakers:
            y, file_sr = sf.read(recordings[speaker]["2"], dtype="float32")
            assert file_sr == silence_sr
            start = cursor / file_sr
            timeline_audio.append(y); cursor += len(y)
            end = cursor / file_sr
            reference_segments.append({"start": start, "end": end, "speaker": speaker})
            timeline_audio.append(silence); cursor += len(silence)
        timeline_audio = np.concatenate(timeline_audio)
        display(Audio(timeline_audio, rate=silence_sr))

        colors = {s: f"C{i}" for i, s in enumerate(speakers)}
        fig, ax = plt.subplots(figsize=(12, 2.6))
        tt = np.arange(len(timeline_audio)) / silence_sr
        ax.plot(tt, timeline_audio, color=".55", lw=.6)
        for segment in reference_segments:
            ax.axvspan(segment["start"], segment["end"], color=colors[segment["speaker"]], alpha=.22)
            ax.text((segment["start"]+segment["end"])/2, .82, segment["speaker"], ha="center",
                    transform=ax.get_xaxis_transform(), color=colors[segment["speaker"]])
        ax.set(xlabel="时间 (s)", ylabel="幅值", title="真实 FSDD 拼接时间轴：这里的 segment 边界来自构造真值")
        plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        ### DER：错误时间占参考说话时间的比例

        常见定义：`DER = (Miss + False Alarm + Speaker Confusion) / Reference Speaker Time`。

        - Miss：参考有人说话，系统判成无人；
        - False Alarm：参考无人，系统判成有人；
        - Confusion：有人说话，但说话人标签错；
        - 分母通常是参考说话时间，而不是整段录音时长。

        实际工具还涉及 collar、重叠区是否计分、说话人标签最优映射等约定。比较论文或产品数据前必须统一 scoring protocol。
        """
    ),
    code(
        """
        def diarization_error_rate(reference, hypothesis):
            reference = np.asarray(reference, dtype=object)
            hypothesis = np.asarray(hypothesis, dtype=object)
            speech = reference != "silence"
            miss = np.sum(speech & (hypothesis == "silence"))
            false_alarm = np.sum(~speech & (hypothesis != "silence"))
            confusion = np.sum(speech & (hypothesis != "silence") & (hypothesis != reference))
            denominator = np.sum(speech)
            return {
                "miss": int(miss), "false_alarm": int(false_alarm), "confusion": int(confusion),
                "reference_speech": int(denominator),
                "DER": (miss + false_alarm + confusion) / denominator,
            }

        # 每个元素代表 10 ms。构造 100 帧：80 帧有说话，20 帧静音。
        ref = np.array(["A"]*40 + ["silence"]*10 + ["B"]*40 + ["silence"]*10, dtype=object)
        hyp = ref.copy()
        hyp[5:10] = "silence"       # 5 帧 miss
        hyp[40:45] = "A"            # 5 帧 false alarm
        hyp[60:65] = "A"            # 5 帧 speaker confusion
        der_parts = diarization_error_rate(ref, hyp)
        print(der_parts)

        fig, ax = plt.subplots(figsize=(10, 3))
        names = ["Miss", "False alarm", "Confusion"]
        values = [der_parts["miss"], der_parts["false_alarm"], der_parts["confusion"]]
        ax.bar(names, values, color=["C1", "C2", "C3"])
        ax.axhline(der_parts["reference_speech"], color="k", ls="--", label="参考说话帧数（分母）")
        ax.set(ylabel="10 ms 帧数", title=f"DER = 15 / 80 = {der_parts['DER']:.2%}")
        ax.legend(); plt.tight_layout(); plt.show()
        assert abs(der_parts["DER"] - 0.1875) < 1e-12
        """
    ),
    md(
        """
        ### 说话人检查题（每题 2 分）

        1. speaker verification、identification、diarization 分别回答什么？
        2. 为什么本实验 100% top-1 accuracy 仍不能证明可部署？
        3. 若整段 100 秒、参考说话仅 20 秒，5 秒 false alarm 的 DER 贡献为什么是 25% 而不是 5%？
        4. 两人重叠说话时，“每帧只能有一个标签”的模型哪里失效？

        <details><summary>参考答案</summary>

        verification 判断“是否为声称的人”；identification 从已知集合选身份；diarization 切时间轴并回答谁在何时说。数据只有 4 人、每人 3 条，环境单一，且 george margin 极小。DER 分母是参考说话时间 20 秒。重叠帧需要多标签、重叠检测或源分离。

        </details>
        """
    ),
    md(
        """
        ## 3. 标点与 ITN：展示文本不是原始证据

        ASR 常输出 spoken form：`温度调到二十六度`。ITN（inverse text normalization）把它变成 written form：`温度调到26度`。

        ITN 不是简单把每个中文数字替换成阿拉伯数字：

        - `一三八零零一三八零零零` 是逐位电话号码；
        - `一百零二` 是带单位的整数 102；
        - `三点` 可能是时间，也可能是小数语境；
        - 日期、货币、地址、编号各有自己的 semiotic class。

        标点同样会改变语义，例如 `不要关闭空调` 与 `不要，关闭空调`。因此原始 transcript、标点文本、ITN 文本必须分字段保存。
        """
    ),
    code(
        """
        DIGITS = {"零":0, "一":1, "二":2, "两":2, "三":3, "四":4, "五":5, "六":6, "七":7, "八":8, "九":9}
        UNITS = {"十":10, "百":100, "千":1000}

        def chinese_integer(text):
            if all(char in DIGITS for char in text):
                return int("".join(str(DIGITS[char]) for char in text))
            total, current = 0, 0
            for char in text:
                if char in DIGITS:
                    current = DIGITS[char]
                elif char in UNITS:
                    unit = UNITS[char]
                    total += (current if current else 1) * unit
                    current = 0
                else:
                    raise ValueError(f"不支持的数字字符: {char}")
            return total + current

        def itn(text):
            output = text
            # 教学版按明确上下文处理，范围故意有限。
            output = re.sub(r"号码是([零一二两三四五六七八九]+)",
                            lambda m: "号码是" + str(chinese_integer(m.group(1))), output)
            output = re.sub(r"(温度调到|调到)([零一二两三四五六七八九十百千]+)度",
                            lambda m: m.group(1) + str(chinese_integer(m.group(2))) + "度", output)
            output = re.sub(r"订单([零一二两三四五六七八九十百千]+)",
                            lambda m: "订单" + str(chinese_integer(m.group(1))), output)
            return output

        golden_itn = {
            "温度调到二十六度": "温度调到26度",
            "把温度调到二十二度": "把温度调到22度",
            "我的号码是一三八零零一三八零零零": "我的号码是13800138000",
            "查询订单一百零二": "查询订单102",
        }
        for spoken, expected in golden_itn.items():
            actual = itn(spoken)
            print(f"{spoken} -> {actual}")
            assert actual == expected
        print(f"黄金集 exact match: {sum(itn(k)==v for k,v in golden_itn.items())}/{len(golden_itn)}")
        """
    ),
    md(
        """
        这个 ITN 只通过 4 条教学黄金集，不能据此宣称“中文 ITN 已解决”。请主动找失败例：`百分之二十`、`二零二六年八月十八日`、`三点一四`、`一万零三`、`幺三八...`。生产系统通常按类别使用规则/FST、分类器或受约束模型，并用按类别的 sentence accuracy / token F1 验收。

        **小题：** 如果 ITN 不确定，应保留 spoken form、强制变换，还是编造一个最常见格式？为什么？
        """
    ),
    md(
        """
        ## 4. 置信度校准：90% 应当长期约有 90% 正确

        最大 softmax 只是模型分布中的最大值，并非天然校准的概率。校准关注：在置信度约 0.8 的样本中，真实正确率是否也约 0.8。

        本节生成一个可控三分类问题：标签按“真实温度”2.4 采样，但系统直接使用更尖锐的 logits，因此过度自信。我们只在 validation split 上选择 temperature，再到独立 test split 报告 ECE，避免在测试集调参。
        """
    ),
    code(
        """
        rng = np.random.default_rng(42)
        logits = rng.normal(0, 1.5, size=(6000, 3))

        def softmax(z):
            z = z - z.max(axis=1, keepdims=True)
            exp_z = np.exp(z)
            return exp_z / exp_z.sum(axis=1, keepdims=True)

        true_temperature = 2.4
        true_probs = softmax(logits / true_temperature)
        labels = np.array([rng.choice(3, p=p) for p in true_probs])
        val_logits, test_logits = logits[:3000], logits[3000:]
        val_labels, test_labels = labels[:3000], labels[3000:]

        def nll(z, y, temperature=1.0):
            probs = softmax(z / temperature)
            return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

        def calibration_stats(z, y, temperature=1.0, bins=10):
            probs = softmax(z / temperature)
            confidence = probs.max(axis=1)
            correct = probs.argmax(axis=1) == y
            edges = np.linspace(0, 1, bins+1)
            rows, ece = [], 0.0
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
                if mask.any():
                    mean_conf = float(confidence[mask].mean())
                    accuracy = float(correct[mask].mean())
                    weight = float(mask.mean())
                    ece += weight * abs(mean_conf - accuracy)
                    rows.append((mean_conf, accuracy, int(mask.sum())))
            return rows, float(ece)

        grid = np.linspace(.5, 4.0, 141)
        fitted_temperature = float(grid[np.argmin([nll(val_logits, val_labels, t) for t in grid])])
        raw_rows, raw_ece = calibration_stats(test_logits, test_labels, 1.0)
        cal_rows, calibrated_ece = calibration_stats(test_logits, test_labels, fitted_temperature)
        print(f"validation 拟合 temperature: {fitted_temperature:.3f}")
        print(f"独立 test ECE: {raw_ece:.4f} -> {calibrated_ece:.4f}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for rows, label in [(raw_rows, "未校准"), (cal_rows, "temperature scaling")]:
            ax[0].plot([r[0] for r in rows], [r[1] for r in rows], "o-", label=label)
        ax[0].plot([0,1], [0,1], "k--", label="理想校准")
        ax[0].set(xlabel="平均置信度", ylabel="真实准确率", title="Reliability diagram", xlim=(0,1), ylim=(0,1))
        ax[0].legend()
        ax[1].plot(grid, [nll(val_logits, val_labels, t) for t in grid])
        ax[1].axvline(fitted_temperature, color="r", ls="--")
        ax[1].set(xlabel="temperature", ylabel="validation NLL", title="只在 validation 上拟合 T")
        plt.tight_layout(); plt.show()

        assert 1.8 < fitted_temperature < 3.0
        assert calibrated_ece < raw_ece * .5
        """
    ),
    code(
        """
        def show_temperature(temperature=1.0):
            rows, ece = calibration_stats(test_logits, test_labels, temperature)
            plt.figure(figsize=(5, 4))
            plt.plot([0,1], [0,1], "k--")
            plt.plot([r[0] for r in rows], [r[1] for r in rows], "o-")
            plt.xlim(0,1); plt.ylim(0,1); plt.grid(alpha=.2)
            plt.xlabel("平均置信度"); plt.ylabel("真实准确率")
            plt.title(f"T={temperature:.2f}; ECE={ece:.3f}")
            plt.tight_layout(); plt.show()

        temperature_ui = widgets.interactive(
            show_temperature,
            temperature=widgets.FloatSlider(min=.5, max=4, step=.1, value=1, description="T", continuous_update=False)
        )
        display(temperature_ui)
        """
    ),
    md(
        """
        ### 校准检查题（每题 2 分）

        1. temperature scaling 通常改变 argmax 和准确率吗？主要改变什么？
        2. 为什么不能在测试集选择最佳 temperature 后仍把它叫“独立测试结果”？
        3. ECE 很低是否证明每个单独样本的概率都绝对正确？
        4. ASR 句级置信度为什么比单个分类 softmax 更复杂？

        <details><summary>参考答案</summary>

        正的 temperature 不改变 logits 排序，所以通常不改 argmax；它调整分布尖锐度。测试集调参会泄漏。ECE 是分箱总体统计，会受分箱方式影响，不能保证个体概率。ASR 还涉及长度、blank、路径合并、beam/lattice posterior、LM 分数和删插错误。

        </details>
        """
    ),
    md(
        """
        ## 5. N-best：给后端纠错机会，也给“语义带偏”留下风险

        1-best 只保留搜索最优候选；N-best 保留替代假设。如果正确文本还在候选中，领域语义可能救回同音错误。如果已被 beam 剪掉，后端不能在没有音频证据时假装恢复了它。

        下面将 acoustic/search score 与 semantic score 加权：`total = acoustic + alpha × semantic`。分数越大越好。拖动 alpha，观察决定何时从同音错误变正确，又何时被一个业务上常见、但声学证据更差的错误房间带偏。
        """
    ),
    code(
        """
        nbest = [
            {"text":"打开客厅等", "acoustic":-1.60, "semantic":-2.00, "note":"声学较强、语义不通"},
            {"text":"打开客厅灯", "acoustic":-2.10, "semantic":-0.40, "note":"本例参考答案"},
            {"text":"打开卧室灯", "acoustic":-3.00, "semantic":-0.10, "note":"语义很通、声学较弱"},
        ]

        def rerank(candidates, alpha):
            scored = []
            for item in candidates:
                row = dict(item)
                row["total"] = row["acoustic"] + alpha * row["semantic"]
                scored.append(row)
            return sorted(scored, key=lambda x: x["total"], reverse=True)

        def show_reranking(alpha=0.0):
            ranking = rerank(nbest, alpha)
            print("当前选择:", ranking[0]["text"])
            for row in ranking:
                print(f"{row['total']:6.2f} = {row['acoustic']:5.2f} + {alpha:.2f}×{row['semantic']:5.2f}  {row['text']}")
            xs = np.linspace(0, 5, 101)
            plt.figure(figsize=(9, 4))
            for item in nbest:
                plt.plot(xs, item["acoustic"] + xs*item["semantic"], label=item["text"])
            plt.axvline(alpha, color="k", ls="--")
            plt.xlabel("semantic weight alpha"); plt.ylabel("total score（越大越好）")
            plt.title("权重太小无法纠同音；太大可能无视声学证据")
            plt.legend(); plt.tight_layout(); plt.show()

        rerank_ui = widgets.interactive(
            show_reranking,
            alpha=widgets.FloatSlider(min=0, max=5, step=.1, value=.5, description="alpha", continuous_update=False)
        )
        display(rerank_ui)
        assert rerank(nbest, 0)[0]["text"] == "打开客厅等"
        assert rerank(nbest, .5)[0]["text"] == "打开客厅灯"
        assert rerank(nbest, 5)[0]["text"] == "打开卧室灯"
        """
    ),
    code(
        """
        def edit_distance(reference, hypothesis):
            ref, hyp = reference.split(), hypothesis.split()
            dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=int)
            dp[:,0] = np.arange(len(ref)+1); dp[0,:] = np.arange(len(hyp)+1)
            for i in range(1, len(ref)+1):
                for j in range(1, len(hyp)+1):
                    dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1,
                                  dp[i-1,j-1] + (ref[i-1] != hyp[j-1]))
            return int(dp[-1,-1]), len(ref)

        evaluation_nbest = [
            ("打开 客厅 灯", ["打开 客厅 等", "打开 客厅 灯", "打开 卧室 灯"]),
            ("温度 调到 二十六 度", ["温度 调到 二十五 度", "温度 调到 二十四 度", "温度 调到 二十六 度"]),
            ("播放 贝多芬", ["播放 贝多分", "播放 贝多峰", "播放 背多分"]),
        ]
        one_best_errors = one_best_words = oracle_errors = 0
        for reference, candidates_list in evaluation_nbest:
            first_error, words = edit_distance(reference, candidates_list[0])
            candidate_errors = [edit_distance(reference, h)[0] for h in candidates_list]
            one_best_errors += first_error; one_best_words += words
            oracle_errors += min(candidate_errors)
            print(reference, "| 1-best edits", first_error, "| oracle edits", min(candidate_errors))
        one_best_wer = one_best_errors / one_best_words
        oracle_wer = oracle_errors / one_best_words
        print(f"1-best WER={one_best_wer:.2%}; N-best oracle WER={oracle_wer:.2%}")
        print("第 3 句正确答案不在候选中，所以 oracle WER 仍不为 0。")
        assert oracle_wer < one_best_wer
        assert oracle_wer > 0
        """
    ),
    md(
        """
        ## 6. NLU：把文本解释为结构，但不负责授权

        NLU 常输出：`intent + slots`。例如：

        ```json
        {"intent":"set_temperature", "room":"客厅", "value":26, "unit":"C"}
        ```

        这只是解释，不是执行许可。业务层还要检查 schema、设备是否存在、数值范围、用户权限、动作是否可撤销，以及 ASR/NLU 是否足够可信。
        """
    ),
    code(
        """
        def parse_nlu(normalized_text):
            temp = re.search(r"(?:把)?(客厅|卧室)?(?:温度)?调到(\\d+)度", normalized_text)
            if temp:
                return {"intent":"set_temperature", "room":temp.group(1),
                        "value":int(temp.group(2)), "unit":"C"}
            switch = re.search(r"(打开|关闭)(客厅|卧室)?(灯|空调)", normalized_text)
            if switch:
                return {"intent":"switch_device", "action":switch.group(1),
                        "room":switch.group(2), "device":switch.group(3)}
            return {"intent":"unknown"}

        def validate_action(semantics):
            if semantics["intent"] == "set_temperature":
                if semantics.get("room") not in {"客厅", "卧室"}:
                    return False, "房间缺失或未知"
                if not 16 <= semantics.get("value", -999) <= 30:
                    return False, "温度越过安全范围 16～30°C"
                return True, "schema 与范围通过"
            if semantics["intent"] == "switch_device":
                if semantics.get("room") not in {"客厅", "卧室"}:
                    return False, "房间缺失或未知"
                return True, "schema 通过"
            return False, "未知意图"

        nlu_cases = [
            "把客厅温度调到26度",
            "把卧室温度调到80度",
            "关闭客厅灯",
            "把温度调到22度",
            "今天天气怎么样",
        ]
        for text in nlu_cases:
            semantics = parse_nlu(text)
            valid, reason = validate_action(semantics)
            print(text, "=>", semantics, "=>", valid, reason)

        assert validate_action(parse_nlu("把客厅温度调到26度"))[0]
        assert not validate_action(parse_nlu("把卧室温度调到80度"))[0]
        assert not validate_action(parse_nlu("把温度调到22度"))[0]
        """
    ),
    md(
        """
        ## 7. 联合决策：execute、clarify、reject 都是合法输出

        下面的门控只是可解释教学策略：

        - schema/范围不合法：`reject`；
        - ASR 或 NLU 低于阈值：`clarify`；
        - 高风险动作：即使分数高也要 `clarify` 或强认证；
        - 低风险且证据充分：`execute`。

        阈值必须在验证集上根据误执行成本选择，而不是凭感觉设成 0.8。
        """
    ),
    code(
        """
        def decide_action(semantics, asr_confidence, nlu_confidence, high_risk=False,
                          asr_threshold=.75, nlu_threshold=.80):
            valid, reason = validate_action(semantics)
            if not valid:
                return "reject", reason
            if high_risk:
                return "clarify", "高风险动作需要二次确认/认证"
            if asr_confidence < asr_threshold:
                return "clarify", "声学证据不足"
            if nlu_confidence < nlu_threshold:
                return "clarify", "语义解释不确定"
            return "execute", "证据与策略均通过"

        safe_semantics = parse_nlu("把客厅温度调到26度")
        decision_cases = [
            (.93, .94, False),
            (.55, .98, False),
            (.98, .55, False),
            (.99, .99, True),
        ]
        for asr_c, nlu_c, risk in decision_cases:
            print(asr_c, nlu_c, risk, "=>", decide_action(safe_semantics, asr_c, nlu_c, risk))
        assert decide_action(safe_semantics, .93, .94)[0] == "execute"
        assert decide_action(safe_semantics, .55, .98)[0] == "clarify"
        assert decide_action(safe_semantics, .99, .99, True)[0] == "clarify"
        """
    ),
    md(
        """
        ### NLU 与决策检查题（每题 2 分）

        1. ASR confidence 低、NLU confidence 高，为什么不能只信 NLU？
        2. `value=80` 被正确抽取，为什么仍不能执行？
        3. 对话状态为什么必须按 session 隔离？
        4. “请求复述”算系统失败吗？在高风险场景如何看？

        <details><summary>参考答案</summary>

        NLU 可能对错误文本做出很自信的合理解释；抽取正确不等于值在业务安全范围；共享对话状态会让不同用户的省略信息串话；澄清会增加一次交互，但能显著降低误执行，通常是正确的风险控制输出。

        </details>
        """
    ),
    md(
        """
        ## 8. LLM 的正确位置：生成解释，不改写证据

        给 LLM 的输入不应只有 1-best，至少应包含：

        - 原始 transcript 与音频/请求 ID；
        - N-best 或 lattice 中的关键候选及 acoustic/LM score；
        - token/word confidence、时间戳和 speaker segment；
        - 允许的 intent/slot schema、设备清单和安全策略；
        - 模型、prompt、规则和词表版本。

        LLM 输出应通过 JSON schema、值域、证据 grounding 和授权检查。它可以建议“可能是客厅灯”，但若所有候选和原始文本都没有“卧室”，就不能悄悄把房间改成卧室并执行。
        """
    ),
    code(
        """
        @dataclass(frozen=True)
        class EvidenceRecord:
            request_id: str
            raw_transcript: str
            nbest: tuple
            word_timestamps: tuple
            asr_confidence: float
            normalized_text: str
            interpretation: dict
            action_decision: str
            asr_model_version: str
            postprocess_version: str

        def grounded_in_evidence(interpretation, raw_transcript, nbest_texts):
            evidence = raw_transcript + " " + " ".join(nbest_texts)
            for key in ["room", "device", "action"]:
                value = interpretation.get(key)
                if value and str(value) not in evidence:
                    return False, f"{key}={value} 未出现在原始/N-best 证据中"
            return True, "字段可追溯"

        raw = "打开客厅等"
        nbest_texts = tuple(item["text"] for item in nbest)
        good_interpretation = {"intent":"switch_device", "action":"打开", "room":"客厅", "device":"灯"}
        # “卧室”确实存在于第三个 N-best 候选，不能称为无来源；“厨房”才完全不在证据包中。
        hallucinated_interpretation = {"intent":"switch_device", "action":"打开", "room":"厨房", "device":"灯"}
        print("grounded:", grounded_in_evidence(good_interpretation, raw, nbest_texts))
        print("hallucinated:", grounded_in_evidence(hallucinated_interpretation, raw, nbest_texts))

        decision = decide_action(good_interpretation, .82, .91)[0]
        record = EvidenceRecord(
            request_id="demo-001",
            raw_transcript=raw,
            nbest=tuple((item["text"], item["acoustic"], item["semantic"]) for item in nbest),
            word_timestamps=(("打开", .12, .42), ("客厅", .48, .82), ("等", .88, 1.12)),
            asr_confidence=.82,
            normalized_text="打开客厅灯",
            interpretation=good_interpretation,
            action_decision=decision,
            asr_model_version="tiny-demo-1",
            postprocess_version="rules-demo-1",
        )
        print(json.dumps(asdict(record), ensure_ascii=False, indent=2))
        assert grounded_in_evidence(good_interpretation, raw, nbest_texts)[0]
        assert not grounded_in_evidence(hallucinated_interpretation, raw, nbest_texts)[0]
        assert record.raw_transcript == raw
        """
    ),
    md(
        """
        上面的 grounding 检查只是字符串级教学护栏：它无法判断同义词、否定范围或跨句指代。真实系统可使用 span citation、候选约束解码、检索到的设备 ID、类型化 schema、policy engine 和人工复核。关键不是迷信某一个护栏，而是让每次解释和行动都有可审计证据。
        """
    ),
    md(
        """
        ## 9. 端到端验收矩阵

        | 模块 | 质量指标 | 必须分桶/压力测试 |
        |---|---|---|
        | 时间戳 | boundary MAE/P90/P99、字幕稳定延迟 | 语速、长词、chunk 边界、右上下文 |
        | Diarization | DER 及 miss/FA/confusion | 说话人数、重叠、短 turn、噪声、未知人 |
        | 标点/ITN | punctuation F1、分类别 ITN accuracy | 数字、日期、金额、地址、实体、歧义 |
        | Confidence | ECE、NLL、risk-coverage | 设备、口音、SNR、长度、领域漂移 |
        | N-best | oracle WER、重排后 WER/CER | 候选覆盖率、beam、LM 权重、长尾实体 |
        | NLU | intent accuracy、slot F1、exact match | 否定、省略、多轮、OOV、注入 |
        | 行动系统 | task success、误执行率、澄清率 | 权限、不可撤销动作、并发会话、审计/回滚 |

        不能只报总体均值。对于“开门、转账、医疗记录”等高风险任务，**误执行率和最坏分桶**通常比总体 intent accuracy 更重要。
        """
    ),
    md(
        """
        ## 10. 56 分综合测试（14 题，每题 4 分）

        先闭卷作答，再展开评分要点。

        1. 为什么 encoder step 不能在不知道 hop/subsampling/offset 时直接变成秒？
        2. CTC spike、forced alignment、VAD segment 分别提供哪种时间信息？
        3. verification、identification、diarization 有什么区别？
        4. 写出 DER 三类错误和分母；重叠语音会带来什么额外约定？
        5. 为什么 enrollment/test 使用同一录音是数据泄漏？
        6. ITN 为什么必须按数字、日期、金额等类别处理？为什么保留 raw transcript？
        7. “最大 softmax=0.9”为什么不自动意味着 90% 正确？
        8. reliability diagram 与 ECE 分别怎样看？temperature 在哪里拟合？
        9. 1-best WER、N-best oracle WER、重排后 WER 分别回答什么问题？
        10. 为什么 semantic weight 太大可能产生“合理但没说过”的结果？
        11. intent/slot 正确为什么仍不等于可以执行？
        12. ASR 低置信、NLU 高置信时，系统有哪些合法选择？
        13. 给 LLM 的证据包至少应有什么？为什么不能覆盖原 transcript？
        14. 设计一次“语音控制房门”的最小上线验收：列出至少 6 项证据。

        <details><summary>评分要点</summary>

        1. 需要完整坐标变换，流式还需原始 sample offset。
        2. spike 是发射峰；forced alignment 在已知文本条件下找对齐；VAD 给语音段边界。
        3. 一对一身份核验、一对多身份选择、未知时间轴分段聚类。
        4. Miss、false alarm、speaker confusion，除以参考说话时间；需说明 collar、overlap 计分等。
        5. 测试包含注册样本的录音/环境特征，不能代表新话语泛化。
        6. 相同数字串在不同语境的结构不同；raw 是不可变听觉证据，便于审计/回滚。
        7. softmax 可过度自信，必须在独立标注数据上校准验证。
        8. 图看置信度与实际正确率是否贴近对角线；ECE 汇总分箱差距；T 在 validation 上拟合。
        9. 当前首选错误率、候选集合理论上限、实际排序后的最终错误率。
        10. 语义先验压过声学证据，会选择常见但音频不支持的句子。
        11. 还要 schema、值域、设备、权限、风险和确认策略。
        12. 澄清、拒识、展示候选、转人工、仅做可撤销动作；不能被 NLU 自信掩盖。
        13. raw、N-best/分数、confidence、timestamps/speaker、schema/policy、版本；覆盖会丢失证据来源。
        14. 至少应覆盖独立口音/噪声 WER、置信度校准、N-best coverage、意图槽位、误执行/拒识、权限和二次确认、延迟、并发隔离、日志审计、故障回滚中的 6 项。

        </details>
        """
    ),
    md(
        """
        ## 11. 口述毕业题与下一步

        请用 5 分钟、不看笔记讲完这一条链：

        > “实时 ASR 在 chunk 中产生带 offset 的 token；怎样形成词时间戳和说话人段；为什么 raw、标点、ITN 要分开；怎样校准置信度并衡量 N-best coverage；如何抽取 intent/slots；最后怎样让 LLM 帮忙但不能越过证据和权限直接执行。”

        能说出每一步的**输入、输出、状态、指标、失败方式和回退策略**，才算掌握系统而不只是记住名词。

        后续建议进入完整 capstone：选择“实时字幕、会议转写或语音控制”之一，把前端、流式模型、解码、语义、量化部署和评测接成同一可运行系统。
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name":"Python (learn-asr)", "language":"python", "name":"python3"},
        "language_info": {"name":"python", "version":"3.13"},
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
