"""Build the interactive FSDD LOSO and nested-selection learning notebook."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_FSDD六折LOSO_嵌套选择与说话人统计.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        r"""
        # FSDD 六折 LOSO：嵌套选择、说话人方差与真正的实验纪律

        上一实验只留下 **1 位 test speaker**。这次轮流让 6 位说话人各做一次外层 test；每折再拿另一位做 dev，只允许 dev 在 `clean` 与 `gain_noise` 之间选模型。候选阶段在代码层面拿不到 test 录音，选定 checkpoint 后才构造外测特征。

        这节课不追求一个漂亮数字，而要回答四个更重要的问题：

        1. 单一说话人结果为什么可能误导？
        2. model selection 和 model evaluation 为什么必须分开？
        3. micro、speaker-macro、标准差、置信区间分别说什么？
        4. 做过 LOSO 后，为什么还需要新的 untouched test set？

        数据：固定 revision 的 Free Spoken Digit Dataset，CC BY-SA 4.0。原创代码 Apache-2.0；原创讲义 CC BY 4.0。
        """
    ),
    md(
        """
        ## 完成标准

        学完后你应当能够闭卷：

        - 画出任意一折的 train/dev/test speaker 集合并证明无交集；
        - 写出“先 dev 选候选，再解锁 test”的程序断言；
        - 手算两个 speaker 的 micro CER 和 macro CER，并解释它们为何不同；
        - 从 6 个 speaker 结果计算均值、样本标准差和均值区间；
        - 解释为什么 1,800 条 utterance 不等于 1,800 位独立说话人；
        - 看训练曲线、错误样例、长度分桶和 chunk 输出定位失败；
        - 设计一个真正新的外部盲测，而不把 LOSO 冒充“最终泛化证明”。

        建议分 3 次学习，每次 60～90 分钟。第一次到第 5 节，第二次到第 10 节，第三次闭卷做综合测试和答辩。
        """
    ),
    md(
        """
        ## 课前诊断：先写预测，不要先看结果

        1. 6 位 speaker，每折 1 test、1 dev，共需要几折？每折 train 有几位？
        2. 如果先比较两个候选的 test CER 再选更好的，test 实际变成了什么？
        3. speaker A 有 100 字符、10 错；speaker B 有 10 字符、5 错。micro CER 与 macro CER 各是多少？
        4. 500 条录音都来自同一人，独立的 speaker cluster 数是多少？
        5. dev CER 低是否保证 test CER 低？为什么？
        6. LOSO 的每一折 test 都没参与该折训练，是否就自动等于“全新盲测”？

        把答案写在学习日志；运行后只改正推理，不要抹掉原答案。
        """
    ),
    code(
        r"""
        from pathlib import Path
        import json, math, statistics, sys
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import Audio, display, Markdown

        HERE = Path.cwd().resolve()
        ROOT = HERE if (HERE / "pyproject.toml").exists() else HERE.parent
        assert (ROOT / "pyproject.toml").exists(), "请从仓库根目录或 notebooks 目录启动 Jupyter"
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from fsdd_generalization.data import EXPECTED_SPEAKERS, prepare_fsdd, scan_recordings
        from fsdd_generalization.loso import build_loso_folds, select_candidate, aggregate_outer_folds
        from fsdd_generalization.training import load_waveform, frontend_config
        from acoustic_engine.features import LogMelFrontend
        from acoustic_engine.streaming import StreamingAcousticEngine

        RESULTS_PATH = ROOT / "artifacts" / "fsdd_loso_results.json"
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        print("protocol:", results["protocol"])
        print("interpretation:", results["interpretation"])
        print("folds:", len(results["folds"]))
        """
    ),
    md(
        """
        ## 1. LOSO 到底在轮换什么？

        LOSO = leave one speaker out。外层每次留一位 speaker 做 test。因为还要选 epoch 和候选增强策略，本课程再从剩余 speaker 中固定一位做 dev，其余 4 位训练。

        这里的 dev 采用预先固定的循环：test 的下一位 speaker 做 dev。它不是唯一方案，也不是完整 nested cross-validation；它是计算量可控的 **one-inner-dev approximation**。工业研究若要调很多超参数，应在每个外层训练池内再做完整 inner CV。
        """
    ),
    code(
        r"""
        folds = build_loso_folds()
        role = {"train": 0, "dev": 1, "test": 2}
        matrix = np.zeros((len(folds), len(EXPECTED_SPEAKERS)), dtype=int)
        for row, fold in enumerate(folds):
            for col, speaker in enumerate(EXPECTED_SPEAKERS):
                matrix[row, col] = (
                    role["test"] if speaker == fold.test_speaker
                    else role["dev"] if speaker == fold.dev_speaker
                    else role["train"]
                )

        fig, ax = plt.subplots(figsize=(10, 4.5))
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(6), EXPECTED_SPEAKERS, rotation=25, ha="right")
        ax.set_yticks(range(6), [f"fold {i+1}" for i in range(6)])
        ax.set_title("Speaker role in every outer fold")
        labels = {0: "TRAIN", 1: "DEV", 2: "TEST"}
        for i in range(6):
            for j in range(6):
                ax.text(j, i, labels[matrix[i, j]], ha="center", va="center", color="white", fontsize=8)
        plt.tight_layout(); plt.show()

        assert all(len(fold.train_speakers) == 4 for fold in folds)
        assert {fold.test_speaker for fold in folds} == set(EXPECTED_SPEAKERS)
        """
    ),
    md(
        """
        ### 图应该怎么看？

        横轴是一位具体说话人，纵轴是一折实验。沿着任意一行看：4 个 TRAIN、1 个 DEV、1 个 TEST；沿着任意一列看：同一 speaker 会在不同折承担不同角色，且恰好做一次外层 TEST。

        **检查题：** 为什么不能把同一 speaker 的 index 0～4 放 test、5～49 放 train 来代替这张图？因为那只检验“见过的人说了新录音”，没有隔离 speaker identity、口音与声道条件。
        """
    ),
    code(
        r"""
        # 亲手检查集合不变量。删掉任一断言，解释会失去什么保护。
        for fold in folds:
            train = set(fold.train_speakers)
            dev = {fold.dev_speaker}
            test = {fold.test_speaker}
            assert not (train & dev or train & test or dev & test)
            assert train | dev | test == set(EXPECTED_SPEAKERS)
        print("6 folds × 3 pairwise leakage checks: PASS")
        """
    ),
    md(
        """
        ## 2. 两层决策：先选模型，再估计选择程序

        一折内的顺序必须是：

        `train 两个候选 → 看 dev → 冻结选择 → 构造 outer test → 只评选中者`

        外层 test 测的不是“某个预先固定权重”，而是完整选择程序：给定 train/dev，这套程序最后交付怎样的模型。若 clean 和 gain_noise 都看了外层 test，再挑更好的报告，就产生 multiple-comparisons selection bias。
        """
    ),
    code(
        r"""
        # 产物级泄漏审计：候选结果里不存在 test；外测只在 selected_candidate 之后出现。
        for fold in results["folds"]:
            expected = select_candidate(fold["candidate_results"])
            assert fold["selected_candidate"] == expected
            assert fold["outer_test_constructed_after_selection"] is True
            for candidate, candidate_result in fold["candidate_results"].items():
                assert candidate_result["test_evaluated"] is False
                assert "test" not in candidate_result
                assert candidate_result["test_sequences"] == 0
        print("dev-only selection + delayed outer-test construction: PASS")
        """
    ),
    md(
        """
        ### 代码边界为什么比“我保证没看”更好？

        候选阶段调用 `prepare_experiment_features(train, dev, [], ...)`，第三个参数是空列表。即使未来有人误写 test 指标打印，也没有 test 音频和标签可读。选择完成后才调用 `prepare_test_features`。

        仍要注意：数据路径隔离不能防止研究者早已从先前实验知道该数据集的结论。因此本实验被标记为 `exploratory_resampling_estimate_not_a_new_untouched_benchmark`。
        """
    ),
    md("## 3. 每一折怎样从训练曲线选择候选？"),
    code(
        r"""
        @widgets.interact(fold_number=widgets.IntSlider(value=1, min=1, max=6, step=1, description="fold"))
        def plot_candidate_curves(fold_number=1):
            fold = results["folds"][fold_number - 1]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            for name, candidate in fold["candidate_results"].items():
                history = candidate["history"]
                epochs = [row["epoch"] for row in history]
                axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label=name)
                axes[1].plot(epochs, [100*row["dev_cer"] for row in history], marker="o", label=name)
                best = candidate["best_epoch"]
                best_row = next(row for row in history if row["epoch"] == best)
                axes[1].scatter([best], [100*best_row["dev_cer"]], s=100, marker="*", zorder=3)
            axes[0].set(title="Training objective", xlabel="epoch", ylabel="CTC loss")
            axes[1].set(title=f"DEV only: {fold['dev_speaker']}", xlabel="epoch", ylabel="CER (%)")
            for ax in axes: ax.grid(alpha=.25); ax.legend()
            fig.suptitle(f"fold {fold_number}: test={fold['test_speaker']} | selected={fold['selected_candidate']}")
            plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        看右图的星号：它是每个候选各自最好的 dev epoch。先比较星号对应的 `(dev CER, -dev exact)`，完全相同才优先选更简单的 clean。不要拿最后一个 epoch 比，也不要拿最低 train loss 比。

        **预测题：** 某候选 train loss 持续下降，但 dev CER 从 45% 升到 60%，说明什么？答案不是“再训久一点”，而是泛化开始恶化，应该回到已保存的最佳 dev checkpoint。
        """
    ),
    code(
        r"""
        print(f"{'fold':<5}{'test':<11}{'dev':<11}{'selected':<13}{'dev CER':>9}{'test CER':>10}{'test exact':>12}")
        for fold in results["folds"]:
            chosen = fold["candidate_results"][fold["selected_candidate"]]["dev"]
            outer = fold["outer_test"]["sequence"]
            print(
                f"{fold['fold']:<5}{fold['test_speaker']:<11}{fold['dev_speaker']:<11}"
                f"{fold['selected_candidate']:<13}{chosen['cer']:>9.2%}{outer['cer']:>10.2%}{outer['exact_rate']:>12.2%}"
            )
        """
    ),
    md(
        """
        ## 4. 最重要的一张图：speaker variance

        若只报告所有 utterance 合起来的一个 CER，Theo 的灾难性失败会被平均掉。下面同时画每位外层 speaker 的序列 CER 和 exact rate；每根柱是一整个 speaker cluster，而不是一条录音。
        """
    ),
    code(
        r"""
        speaker_names = [f["test_speaker"] for f in results["folds"]]
        speaker_cer = np.array([f["outer_test"]["sequence"]["cer"] for f in results["folds"]])
        speaker_exact = np.array([f["outer_test"]["sequence"]["exact_rate"] for f in results["folds"]])
        x = np.arange(6)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].bar(x, speaker_cer * 100)
        axes[0].axhline(speaker_cer.mean()*100, color="black", linestyle="--", label="speaker macro mean")
        axes[0].set_ylabel("sequence CER (%)"); axes[0].legend(); axes[0].grid(axis="y", alpha=.25)
        axes[1].bar(x, speaker_exact * 100)
        axes[1].axhline(speaker_exact.mean()*100, color="black", linestyle="--")
        axes[1].set_ylabel("sequence exact (%)"); axes[1].set_xticks(x, speaker_names, rotation=25)
        axes[1].grid(axis="y", alpha=.25)
        fig.suptitle("Outer-test quality differs sharply by held-out speaker")
        plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        ### 图应该怎么看？

        - CER 越低越好；exact 越高越好。两图不是重复：一句有一个字符错，exact 就整句归零。
        - Theo 的序列 CER 约 90%，而 Yweweler 约 35%，差距不是小数点波动。
        - 因此上一课只留 Yweweler 得到的 40.16% CER 偏乐观，不能代表“任意新人”。
        - 这也说明 test set 的“人数”比同一个人多录多少条更关键。
        """
    ),
    md("## 5. dev 能否预测 test？"),
    code(
        r"""
        dev_cer = np.array([
            f["candidate_results"][f["selected_candidate"]]["dev"]["cer"]
            for f in results["folds"]
        ])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(dev_cer*100, speaker_cer*100, s=70)
        for name, dx, ty in zip(speaker_names, dev_cer*100, speaker_cer*100):
            ax.annotate(name, (dx, ty), xytext=(5, 4), textcoords="offset points")
        lo = min(dev_cer.min(), speaker_cer.min())*100 - 3
        hi = max(dev_cer.max(), speaker_cer.max())*100 + 3
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="dev = test")
        ax.set(xlabel="selected checkpoint DEV CER (%)", ylabel="OUTER TEST CER (%)", title="A good dev score is a decision signal, not a promise")
        ax.grid(alpha=.25); ax.legend(); plt.show()
        print("Pearson correlation (n=6, descriptive only):", np.corrcoef(dev_cer, speaker_cer)[0,1])
        """
    ),
    md(
        """
        点离对角线越远，dev 与外测难度差异越大。Theo 点尤其提醒我们：一位 dev speaker 无法覆盖所有口音和声学条件。这里的相关系数只有 6 个点，只能描述，不能做稳定因果结论。

        **检查题：** 是否应该看到 Theo 后把 patience 从 4 改成 10，再只重跑这一折？不应该；那是用 test 选择超参数。若提出新策略，应冻结后用新的外部数据验证，或清楚标为下一轮探索。
        """
    ),
    md("## 6. Micro 与 speaker-macro：权重到底给了谁？"),
    code(
        r"""
        rows = [f["outer_test"]["sequence"] for f in results["folds"]]
        micro = sum(r["errors"] for r in rows) / sum(r["reference_characters"] for r in rows)
        macro = statistics.fmean(r["cer"] for r in rows)
        print("micro CER = total edits / total reference chars =", f"{micro:.4%}")
        print("macro CER = mean(one CER per speaker)          =", f"{macro:.4%}")
        print("artifact cross-check:", results["aggregate"]["sequence"]["micro_cer"], results["aggregate"]["sequence"]["macro_speaker_cer"]["mean"])
        """
    ),
    md(
        """
        Micro 让每个参考字符权重相同，长 utterance 多的 speaker 权重更大；speaker-macro 先给每位 speaker 算一个 CER，再让每个人权重相同。本实验每折字符数接近，所以两者相近；这不是数学上必然相等。
        """
    ),
    code(
        r"""
        @widgets.interact(
            easy_chars=widgets.IntSlider(value=100, min=10, max=1000, step=10, description="easy chars"),
            hard_chars=widgets.IntSlider(value=100, min=10, max=1000, step=10, description="hard chars"),
        )
        def micro_macro_simulator(easy_chars=100, hard_chars=100):
            easy_cer, hard_cer = 0.10, 0.70
            micro_value = (easy_chars*easy_cer + hard_chars*hard_cer) / (easy_chars+hard_chars)
            macro_value = (easy_cer+hard_cer)/2
            print(f"micro={micro_value:.2%} | speaker-macro={macro_value:.2%}")
            fig, ax = plt.subplots(figsize=(7,2.8))
            ax.barh(["micro", "speaker macro"], [100*micro_value, 100*macro_value])
            ax.set(xlim=(0,100), xlabel="CER (%)", title="Change sample counts; speaker difficulty stays fixed")
            ax.grid(axis="x", alpha=.25); plt.show()
        """
    ),
    md("## 7. 区间：1,800 句为什么仍然只有 n=6？"),
    code(
        r"""
        summary = results["aggregate"]["sequence"]["macro_speaker_cer"]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        assert summary["speaker_count"] == 6

        means = []
        rng = np.random.default_rng(2026)
        for _ in range(20_000):
            means.append(rng.choice(speaker_cer, size=6, replace=True).mean())
        bootstrap = np.quantile(means, [0.025, 0.5, 0.975])
        print("speaker-cluster bootstrap [2.5%, 50%, 97.5%]:", [f"{x:.2%}" for x in bootstrap])
        """
    ),
    md(
        """
        产物报告的是 6 个 speaker CER 的均值、样本标准差，以及基于 df=5 的 t 区间。区间很宽，是诚实地告诉你：6 位说话人太少。

        上面 bootstrap 的抽样单位也是 speaker，而不是 utterance。如果把 1,800 句独立重采样，区间会虚假地变窄，因为它破坏了同一 speaker 内相关性。

        **术语提醒：** 这个区间依赖“6 个 speaker 可近似视作总体样本”等假设；FSDD 说话人并非随机代表所有语言、年龄、口音和设备，所以它不是全球 ASR 泛化区间。
        """
    ),
    code(
        r"""
        @widgets.interact(seed=widgets.IntSlider(value=1, min=1, max=100, description="seed"), draws=widgets.SelectionSlider(options=[100, 1000, 5000, 20000], value=5000, description="draws"))
        def bootstrap_stability(seed=1, draws=5000):
            rng = np.random.default_rng(seed)
            sampled = rng.choice(speaker_cer, size=(draws, 6), replace=True).mean(axis=1)
            q = np.quantile(sampled, [.025, .5, .975])
            plt.figure(figsize=(8,3.2)); plt.hist(sampled*100, bins=35)
            for value in q: plt.axvline(value*100, color="black", linestyle="--")
            plt.xlabel("resampled speaker-mean CER (%)"); plt.ylabel("count")
            plt.title(f"Cluster bootstrap, {draws} resamples | 95%=[{q[0]:.1%}, {q[2]:.1%}]")
            plt.show()
        """
    ),
    md("## 8. 听一听：困难 speaker 不是表格中的抽象数字"),
    code(
        r"""
        recordings_dir, _ = prepare_fsdd(ROOT / ".local_data")
        recordings = scan_recordings(recordings_dir, validate_audio=False)
        frontend = LogMelFrontend(frontend_config())

        @widgets.interact(
            speaker=widgets.Dropdown(options=list(EXPECTED_SPEAKERS), value="theo", description="speaker"),
            digit=widgets.Dropdown(options=list("0123456789"), value="7", description="digit"),
            index=widgets.IntSlider(value=0, min=0, max=49, description="index"),
        )
        def inspect_real_recording(speaker="theo", digit="7", index=0):
            item = next(r for r in recordings if r.speaker==speaker and r.digit==digit and r.index==index)
            wave = load_waveform(item)
            feat = frontend(wave)
            display(Audio(wave.numpy(), rate=8000))
            fig, axes = plt.subplots(2,1,figsize=(10,5))
            axes[0].plot(np.arange(len(wave))/8000, wave.numpy()); axes[0].set(ylabel="amplitude", title=item.name)
            axes[1].imshow(feat.T.numpy(), origin="lower", aspect="auto"); axes[1].set(xlabel="frame", ylabel="mel bin", title="log-Mel")
            plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        请在 Theo、Lucas、Yweweler 间切换同一个 digit，不要只听一条。记录语速、停顿、响度、音高、发音方式、录音噪声的差异。听音只能提出假设，不能代替系统指标；“我听着没问题”不是 500 条测试的分析。
        """
    ),
    md("## 9. 错误对齐：替换、删除、插入怎样定位？"),
    code(
        r"""
        def align_edits(reference, hypothesis):
            n, m = len(reference), len(hypothesis)
            dp = [[0]*(m+1) for _ in range(n+1)]
            back = [[None]*(m+1) for _ in range(n+1)]
            for i in range(1,n+1): dp[i][0], back[i][0] = i, "D"
            for j in range(1,m+1): dp[0][j], back[0][j] = j, "I"
            for i in range(1,n+1):
                for j in range(1,m+1):
                    choices = [
                        (dp[i-1][j]+1, "D"),
                        (dp[i][j-1]+1, "I"),
                        (dp[i-1][j-1]+(reference[i-1]!=hypothesis[j-1]), "M" if reference[i-1]==hypothesis[j-1] else "S"),
                    ]
                    dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])
            aligned=[]; i,j=n,m
            while i or j:
                op=back[i][j]
                if op in {"M","S"}: aligned.append((reference[i-1],hypothesis[j-1],op)); i-=1; j-=1
                elif op=="D": aligned.append((reference[i-1],"·",op)); i-=1
                else: aligned.append(("·",hypothesis[j-1],op)); j-=1
            return list(reversed(aligned))

        print(align_edits("8817", "8855"))
        """
    ),
    code(
        r"""
        @widgets.interact(
            speaker=widgets.Dropdown(options=speaker_names, value="theo", description="speaker"),
            length=widgets.IntSlider(value=3, min=1, max=4, description="length"),
            example=widgets.IntSlider(value=0, min=0, max=9, description="example"),
        )
        def inspect_errors(speaker="theo", length=3, example=0):
            fold = next(f for f in results["folds"] if f["test_speaker"]==speaker)
            rows = [p for p in fold["outer_test"]["sequence"]["predictions"] if p["reference_length"]==length and not p["correct"]]
            if not rows:
                print("这个分桶没有错误"); return
            row = rows[example % len(rows)]
            aligned = align_edits(row["reference"], row["hypothesis"])
            print("reference :", " ".join(a for a,_,_ in aligned))
            print("hypothesis:", " ".join(b for _,b,_ in aligned))
            print("operation :", " ".join(op for *_,op in aligned), "(M正确/S替换/D删除/I插入)")
        """
    ),
    md("## 10. 长度分桶：平均值会隐藏哪一类句子？"),
    code(
        r"""
        lengths = ["1","2","3","4"]
        cer_by_length = np.array([
            [fold["outer_test"]["sequence"]["per_length"][length]["cer"] for length in lengths]
            for fold in results["folds"]
        ])
        exact_by_length = np.array([
            [fold["outer_test"]["sequence"]["per_length"][length]["exact_rate"] for length in lengths]
            for fold in results["folds"]
        ])
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        for index,name in enumerate(speaker_names):
            axes[0].plot([1,2,3,4], cer_by_length[index]*100, marker="o", label=name)
            axes[1].plot([1,2,3,4], exact_by_length[index]*100, marker="o", label=name)
        axes[0].set(xlabel="reference digits", ylabel="CER (%)", title="CER by sequence length")
        axes[1].set(xlabel="reference digits", ylabel="exact (%)", title="Exact by sequence length")
        for ax in axes: ax.set_xticks([1,2,3,4]); ax.grid(alpha=.25)
        axes[1].legend(ncol=2); plt.tight_layout(); plt.show()
        """
    ),
    md(
        """
        CER 不一定随长度单调变差，因为分母也随字符数增长；sequence exact 通常更容易随长度下降，因为任意一个字符错都会让整句失败。若某 speaker 的长串主要发生 deletion，下一步应查 CTC blank、发音边界、模型感受野和 endpoint，而不是直接加语言模型掩盖声学错误。
        """
    ),
    md("## 11. 每折交付的是可流式运行的 checkpoint"),
    code(
        r"""
        @widgets.interact(
            speaker=widgets.Dropdown(options=speaker_names, value="george", description="outer test"),
            digit=widgets.Dropdown(options=list("0123456789"), value="7", description="digit"),
            chunk=widgets.SelectionSlider(options=[1, 80, 137, 400, 800, 1600], value=800, description="chunk samples"),
        )
        def stream_selected_checkpoint(speaker="george", digit="7", chunk=800):
            fold = next(f for f in results["folds"] if f["test_speaker"]==speaker)
            checkpoint = ROOT / "artifacts" / "fsdd_loso_checkpoints" / fold["selected_checkpoint"]
            engine = StreamingAcousticEngine.load(checkpoint)
            item = next(r for r in recordings if r.speaker==speaker and r.digit==digit and r.index==0)
            wave = load_waveform(item)
            update = engine.recognize_waveform(wave, chunk_samples=chunk)
            print(f"reference={digit} prediction={update.text!r} frames={update.total_frames} chunk={chunk} samples ({1000*chunk/8000:.1f} ms)")
        """
    ),
    code(
        r"""
        # Chunk invariance checks implementation consistency, not recognition correctness.
        fold = results["folds"][0]
        engine = StreamingAcousticEngine.load(ROOT / "artifacts" / "fsdd_loso_checkpoints" / fold["selected_checkpoint"])
        item = next(r for r in recordings if r.speaker==fold["test_speaker"] and r.digit=="7" and r.index==0)
        wave = load_waveform(item)
        chunk_outputs = {size: engine.recognize_waveform(wave, chunk_samples=size).text for size in [1,137,400,800,1600]}
        print(chunk_outputs)
        assert len(set(chunk_outputs.values())) == 1
        """
    ),
    md(
        """
        如果所有 chunk 输出一致但都错了，说明流式实现一致、声学模型质量差；如果离线对而某些 chunk 错，才优先查缓存、前端余帧、跨 chunk CTC 状态。实现正确性和模型准确率是两条独立证据。
        """
    ),
    md("## 12. 为什么 LOSO 仍不是新的最终盲测？"),
    md(
        """
        本课程在上一实验已经：

        - 用 George/Jackson/Lucas/Nicolas 训练；
        - 用 Theo 选 epoch；
        - 看过 Yweweler test；
        - 因此前结果决定继续研究 gain+noise 和 LOSO。

        LOSO 能更充分描述这 6 人之间的方差，也能评估当前选择程序，但研究方向已经受 FSDD 反馈影响。它属于 dataset-internal exploratory resampling，不是新 benchmark。

        真正下一步应在运行前冻结：模型结构、前端、增强、decoder、checkpoint 选择、文本规范、CER/WER 代码、流式 chunk、RTF 硬件和分桶；然后用从未参与设计的新 speaker / 新语料只评一次。结果不好也不能在同一 test 上反复改到好看。
        """
    ),
    code(
        r"""
        contact_ladder = [
            ("训练集", "可反复用于梯度；变化不会提供无偏泛化证据"),
            ("开发集", "可用于选择；反复使用会逐渐对 dev 过拟合"),
            ("LOSO 外折", "描述 FSDD 内 speaker 方差；本课程已查看，不能再称 untouched"),
            ("新外部测试", "协议冻结后一次性评测；当前尚未收集/运行"),
        ]
        for index,(name,meaning) in enumerate(contact_ladder,1):
            print(f"{index}. {name:<10} → {meaning}")
        """
    ),
    md("## 13. 编程迁移任务（必须亲手写）"),
    md(
        """
        1. **泄漏测试：** 故意把 test speaker 加入 train，确认 `SplitSpec.validate()` 抛错；再写一个仅检查文件名却漏掉 speaker 的错误版本，解释为何危险。
        2. **完整 inner CV：** 对某一个 outer test speaker，让剩余 5 人轮流做 inner dev，汇总两个候选的 5 个 dev CER 后再选；outer test 仍只能运行一次。
        3. **cluster bootstrap：** 从空白实现按 speaker 重采样，比较误把 utterance 当独立样本时区间会缩小多少。
        4. **错误操作统计：** 扩展 `align_edits`，汇总每位 speaker 的 S/D/I 次数以及不同长度的 deletion rate。
        5. **新增强预注册：** 在不查看任何现有 outer 结果的“改进版分数”前，写下 speed perturb 或 SpecAugment 的参数、预期改善的分桶和可能恶化的指标。
        6. **外部测试计划：** 选择一个许可兼容的新英语语音来源，先写数据卡、speaker split、文本规范和停止规则，再下载。
        """
    ),
    md("## 14. 96 分综合测试（24 题，每题 4 分）"),
    md(
        """
        1. LOSO 的外层评估对象是单一权重还是完整选择程序？
        2. 为什么每折还需要 dev？
        3. 本实验每折 train/dev/test 各有几位 speaker、多少原始录音？
        4. 写出三组 speaker 集合的泄漏断言。
        5. 为什么官方 index 0～4 split 不等于 speaker-disjoint？
        6. 候选阶段传入空 test 列表保护了什么？没有保护什么？
        7. `test_evaluated=False` 和没有 `test` 字段为什么都要检查？
        8. dev CER 相同时为什么再比较 exact，再相同为什么偏向 clean？
        9. 为什么不能根据 Theo 外测失败单独增加这一折 epochs？
        10. train loss 最低的 epoch 为什么不一定是 checkpoint？
        11. 手算：A 100 字符 20 错、B 20 字符 10 错的 micro CER。
        12. 同上两人的 speaker-macro CER。
        13. micro 和 macro 接近是否证明 speaker 方差小？
        14. 样本标准差 19.62 个百分点在这里表示什么？
        15. 为什么 t 区间使用 n=6，而不是 n=1,800？
        16. utterance bootstrap 会犯什么独立性错误？
        17. dev-test 散点远离对角线表示什么？
        18. CER 和 sequence exact 对长串错误的敏感性有何不同？
        19. CTC 的 substitution/deletion/insertion 分别怎样从对齐得到？
        20. chunk 输出一致能证明模型识别正确吗？
        21. 本轮六折全选 gain_noise 能否证明所有 ASR 都应该加同样噪声？
        22. 为什么本 LOSO 不能称为“全新盲测”？
        23. 完整 nested CV 比本课 one-inner-dev 多了哪层循环和汇总？
        24. 写出下一份 untouched test 在运行前必须冻结的至少 6 项内容。

        **通关线：** 77/96 且第 4、6、11、12、15、22、24 题不能为 0。低于通关线先重做图和代码，不要靠背答案进入下一阶段。
        """
    ),
    md(
        """
        <details><summary>评分要点（答完再展开）</summary>

        1. train/dev 输入到选择规则后交付 checkpoint 的完整程序。
        2. 选择 epoch/候选而不污染 outer test。
        3. 4/1/1 位，2,000/500/500 条。
        4. 三个 pairwise intersection 为空，union 覆盖全体。
        5. 同一 speaker 同时出现，只是 recording index 不同。
        6. 防止该次候选代码读取外测音频/标签；不能抹掉研究者过去的数据接触。
        7. 一个是执行状态，一个防止意外序列化结果；双重证据。
        8. CER 主指标、exact 次级；完全平手选低复杂度，规则预注册。
        9. 它把 test 反馈用于超参选择，破坏外测。
        10. 优化目标与未见数据泛化不同，按 dev 保存。
        11. `(20+10)/(100+20)=25%`。
        12. `(20%+50%)/2=35%`。
        13. 不能；样本量刚好相近也会令加权方式接近。
        14. 六位 held-out speaker CER 围绕均值的离散程度很大。
        15. 推断目标是新 speaker，独立 cluster 只有 6 个。
        16. 把同一人的相关录音当独立，区间虚假变窄。
        17. dev speaker 难度/分布不能稳定代表 test speaker。
        18. CER 按字符编辑归一；exact 任一错误整句归零，长度增长时更苛刻。
        19. 最小编辑距离回溯：错配、只消费参考、只消费假设。
        20. 不能，只证明当前实现不依赖 chunk 切法。
        21. 不能；候选集、语料、说话人和增强范围都有限，且研究已接触数据。
        22. 前一实验已使用/查看全部 speaker，方法选择受 FSDD 反馈影响。
        23. 每个 outer train pool 内再轮换多个 inner dev，先汇总 inner 结果再选候选。
        24. 模型、前端、增强、解码、选择规则、规范化、指标、分桶、流式参数、硬件/线程、停止规则等任 6 项且说明冻结时点。

        </details>
        """
    ),
    md("## 15. 10 分钟口头答辩"),
    md(
        """
        不看 Notebook，用白纸讲清：

        > “我为什么从单一 speaker split 走到六折 LOSO；一折中 test 怎样被代码锁住；dev 怎样选 clean/gain_noise；为什么六折都选增强仍不能夸大结论；speaker macro、micro、标准差和区间分别回答什么；Theo 失败揭示了什么；为什么 chunk 等价不代表准确；当前数据已经接触到什么程度；怎样设计下一份真正新的盲测。”

        录音复听并打 0～4 级：0 不会，1 看提示，2 能解释，3 能写代码排错，4 能迁移到新数据并守住边界。达到 3 才进入外部语料与完整 nested evaluation。
        """
    ),
    md(
        """
        ## 离场票

        今天只写三句话：

        1. 我过去把 ______ 当成独立样本，但真正的 cluster 是 ______。
        2. 本实验最能推翻我原判断的一张图是 ______，因为 ______。
        3. 下一次看到一个 ASR “测试集很好”的结论，我会先问 ______。

        复习安排：明天闭卷重画折叠矩阵；第 7 天手算 micro/macro/区间；第 30 天在新数据上从空白实现 split 与 test gate。
        """
    ),
]


for index, cell in enumerate(cells):
    cell["id"] = f"loso-{index:02d}"

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (learn-asr)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
        "course": {
            "stage": "speaker-generalization",
            "topic": "LOSO and nested model selection",
            "version": 1,
        },
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
