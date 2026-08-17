"""Build the preregistered AudioMNIST external-evaluation learning notebook."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    md(
        """
        # AudioMNIST 外部盲测：冻结协议、跨域失败与适配边界

        这是我们第一次把 FSDD 模型交给 **完全独立项目的 60 位新说话人**。协议先提交，模型再提交，最后才下载/评分；结果无论好坏都公开。你将学到的不是怎样把测试分数调漂亮，而是怎样让一个 ASR 泛化结论可信、可复现、可审计。

        外测数据：AudioMNIST，官方 MIT 许可，固定 revision `630d7dab…72accd`；30,000 条英语数字 0～9、60 位 speaker、48 kHz mono PCM_16。
        """
    ),
    md(
        """
        ## 完成标准

        学完应能闭卷完成：

        - 用 Git 时间线证明 protocol、model、score 的先后；
        - 区分 in-domain dev、dataset-internal LOSO 和 untouched external test；
        - 解释 48→8 kHz 重采样、Nyquist 与前端契约；
        - 同时解释 exact、CER、speaker macro、标准差和 t 区间；
        - 从混淆矩阵和 S/D/I 判断主要失败模式；
        - 解释 speaker/性别/accent 分桶为何只能描述，不能自动给因果结论；
        - 区分 chunk invariance、RTF 和识别质量；
        - 设计 AudioMNIST 适配实验，同时承认它已不再是 untouched test。

        建议学习 3 次，每次 75～100 分钟；第三次完成 120 分闭卷测试与答辩。
        """
    ),
    md(
        """
        ## 课前预测：先写后看

        1. FSDD 只有 6 位 speaker，换到 60 位独立 speaker，你预计 exact 上升还是下降？
        2. 训练是 8 kHz、外测是 48 kHz，能直接把 48 kHz samples 喂给模型吗？
        3. 若 60 位 speaker 的平均 exact 是 60%，能否说明每个人都接近 60%？
        4. chunk=1 和 chunk=1600 输出一致，能否证明识别正确？
        5. 女性组比男性组分数低，能否立即断言模型存在“性别因果偏差”？
        6. 外测后再用 AudioMNIST 调参，能否继续称同一结果为首次盲测？

        在学习日志保留原预测，最后写“哪条证据改变了我”。
        """
    ),
    code(
        r"""
        from pathlib import Path
        import sys, json, hashlib, statistics
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        import soundfile as sf
        import librosa
        import torch
        from IPython.display import Audio, display, Markdown

        HERE = Path.cwd().resolve()
        ROOT = HERE if (HERE / "pyproject.toml").exists() else HERE.parent
        assert (ROOT / "pyproject.toml").exists(), "请从仓库根目录或 notebooks 目录启动"
        if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

        from external_evaluation.protocol import load_protocol, protocol_sha256
        from external_evaluation.data import parse_recording
        from acoustic_engine.features import LogMelFrontend
        from fsdd_generalization.training import frontend_config

        protocol = load_protocol()
        result_path = ROOT / "artifacts" / "audiomnist_external_results.json"
        manifest_path = ROOT / "artifacts" / "audiomnist_external_manifest.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        AUDIO_ROOT = ROOT / ".local_data" / "audiomnist-630d7dab" / "data"
        assert AUDIO_ROOT.exists(), "先运行：uv run python -m external_evaluation.data"
        print(result["status"])
        print("protocol:", result["protocol_sha256"])
        print("checkpoint:", result["frozen_checkpoint_sha256"])
        """
    ),
    md("## 1. 最关键的图不是 CER，而是时间线"),
    code(
        r"""
        events = [
            (1, "0b30360", "protocol frozen\nbefore audio download"),
            (2, "cac8aa6", "model + checkpoint frozen\nbefore scoring"),
            (3, result["started_at"], "first AudioMNIST score"),
            (4, result["completed_at"], "result completed\nno rerun"),
        ]
        fig, ax = plt.subplots(figsize=(11, 2.8))
        ax.plot([e[0] for e in events], [0]*len(events), color="black")
        for x, stamp, label in events:
            ax.scatter(x, 0, s=90)
            ax.text(x, .12, stamp, ha="center", fontsize=9)
            ax.text(x, -.12, label, ha="center", va="top", fontsize=9)
        ax.set(xlim=(.6,4.4), ylim=(-.45,.35), title="Evidence existed before outcomes were known")
        ax.axis("off"); plt.show()

        assert result["protocol_sha256"] == protocol_sha256()
        assert result["preregistration_commit"].startswith("0b30360")
        assert result["frozen_model_commit"].startswith("cac8aa6")
        """
    ),
    md(
        """
        如果先看结果再写规则，人会无意识选择更有利的 epoch、speaker、metric 或分桶。Git 提交把“我本来就这么打算”变成可检查证据。预注册不保证模型好，只保证坏结果也不能被悄悄改写。

        **检查题：** 为什么 checkpoint hash 比文件名更可靠？文件名可以指向不同字节；SHA256 把结论绑定到确切模型内容。
        """
    ),
    md("## 2. 数据域真的变了什么？"),
    code(
        r"""
        comparison = [
            ("speakers", 6, 60),
            ("recordings", 3000, 30000),
            ("sample rate (Hz)", 8000, 48000),
            ("recordings / speaker", 500, 500),
        ]
        print(f"{'property':<24}{'FSDD train':>14}{'AudioMNIST test':>18}")
        for name,left,right in comparison:
            print(f"{name:<24}{left:>14}{right:>18}")

        external_durations = np.array([row["frames"]/48000 for row in manifest["entries"]])
        fsdd_manifest = json.loads((ROOT/"artifacts/fsdd_speaker_disjoint_manifest.json").read_text(encoding="utf-8"))
        fsdd_dir = ROOT/".local_data/fsdd-26eb9aaf/recordings"
        fsdd_durations = np.array([sf.info(fsdd_dir/row["file"]).duration for row in fsdd_manifest["entries"]])
        plt.figure(figsize=(9,3.5))
        plt.hist(fsdd_durations, bins=35, alpha=.65, density=True, label="FSDD")
        plt.hist(external_durations, bins=35, alpha=.65, density=True, label="AudioMNIST")
        plt.xlabel("utterance duration (s)"); plt.ylabel("density"); plt.title("Duration shift")
        plt.legend(); plt.show()
        """
    ),
    md(
        """
        “都是数字 0～9”只表示标签空间相同，不表示输入分布相同。speaker、麦克风、房间、采样率、响度、语速、静音和口音都可能改变 Log-Mel。外部测试正是要测训练时没覆盖的组合。
        """
    ),
    md("## 3. 48 kHz 为什么必须重采样？"),
    code(
        r"""
        def inspect_resampling(speaker="01", digit="7", repetition=0):
            path=AUDIO_ROOT/speaker/f"{digit}_{speaker}_{repetition}.wav"
            wave,sr=sf.read(path,dtype="float32")
            down=librosa.resample(wave,orig_sr=sr,target_sr=8000,res_type="soxr_hq")
            display(Audio(wave,rate=sr))
            fig,axes=plt.subplots(2,2,figsize=(11,5))
            axes[0,0].plot(np.arange(len(wave))/sr,wave); axes[0,0].set(title="48 kHz waveform",xlabel="seconds")
            axes[1,0].plot(np.arange(len(down))/8000,down); axes[1,0].set(title="8 kHz waveform",xlabel="seconds")
            for ax,x,rate,title in [(axes[0,1],wave,sr,"48 kHz spectrum"),(axes[1,1],down,8000,"8 kHz spectrum")]:
                freq=np.fft.rfftfreq(len(x),1/rate); mag=np.abs(np.fft.rfft(x))
                ax.plot(freq,mag); ax.set(xlabel="Hz",title=title,xlim=(0,min(12000,rate/2)))
            plt.tight_layout(); plt.show()
            print("samples:",len(wave),"→",len(down),"duration error:",abs(len(wave)/sr-len(down)/8000))

        inspect_resampling()
        widgets.interact_manual(
            inspect_resampling,
            speaker=widgets.Dropdown(options=[f"{i:02d}" for i in range(1,61)], value="01", description="speaker"),
            digit=widgets.Dropdown(options=list("0123456789"), value="7", description="digit"),
            repetition=widgets.IntSlider(value=0,min=0,max=49,description="repeat"),
        )
        """
    ),
    md(
        """
        8 kHz 的 Nyquist 频率是 4 kHz。高质量重采样先低通，避免 4 kHz 以上内容折叠成假低频，再每 6 个时间采样压到 1 个。若直接把 48 kHz samples 当 8 kHz，时间会被拉长 6 倍，frame/hop 的毫秒含义全部错误。

        本协议不做逐句 peak normalization 和 trimming，因为训练前端也没有；不能为了外测好看临时换预处理。
        """
    ),
    md("## 4. 先看主结果，再看为什么"),
    code(
        r"""
        for name,key in [("single digit","single_digit"),("multi digit","multi_digit")]:
            a=result[key]["aggregate"]
            print(f"{name:<14} exact={a['micro_exact_rate']:.2%} CER={a['micro_cer']:.2%} speakers={a['speaker_macro_cer']['speaker_count']}")
            print(f"  speaker CER mean={a['speaker_macro_cer']['mean']:.2%}, std={a['speaker_macro_cer']['sample_std']:.2%}, 95% CI=[{a['speaker_macro_cer']['ci95_low']:.2%}, {a['speaker_macro_cer']['ci95_high']:.2%}]")
        print("preregistered band:",result["preregistered_interpretation_band"])
        """
    ),
    md(
        """
        单数字 exact 59.30%，落在预注册的 40%～70% 区间，因此标签是 **weak transfer**。这不是生产可用，更不能用“比随机 10% 好很多”包装成成功；但也不是完全失效，它给下一轮适配提供可分析的声学基础。

        多数字 CER 与单数字接近但 exact 更低：一句含多个 token，只要任意位置错，整句 exact 就失败。
        """
    ),
    md("## 5. 平均 59.3% 并不存在于任何一个具体的人身上"),
    code(
        r"""
        per=result["single_digit"]["per_speaker"]
        speakers=sorted(per)
        exact=np.array([per[s]["exact_rate"] for s in speakers])
        cer=np.array([per[s]["cer"] for s in speakers])
        order=np.argsort(exact)
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        axes[0].bar(np.arange(60),exact[order]*100); axes[0].axhline(exact.mean()*100,color="black",ls="--")
        axes[0].set(ylabel="exact (%)",title="Every bar is one independent AudioMNIST speaker")
        axes[1].bar(np.arange(60),cer[order]*100); axes[1].axhline(cer.mean()*100,color="black",ls="--")
        axes[1].set(ylabel="CER (%)",xlabel="speakers sorted by exact",xticks=np.arange(60)[::3],xticklabels=np.array(speakers)[order][::3])
        for ax in axes: ax.grid(axis="y",alpha=.2)
        plt.tight_layout(); plt.show()
        print("best:",speakers[np.argmax(exact)],f"{exact.max():.2%}","worst:",speakers[np.argmin(exact)],f"{exact.min():.2%}")
        """
    ),
    md(
        """
        最佳 speaker 29 为 92.8%，最差 speaker 46 只有 14.0%。speaker 标准差约 19.55 个百分点。这比“30,000 条很大”更重要：泛化单位是人，样本量首先是 60 个 cluster。
        """
    ),
    code(
        r"""
        def speaker_dashboard(speaker="46"):
            s=per[speaker]
            meta=manifest["speakers"][speaker]
            print(f"speaker={speaker} exact={s['exact_rate']:.2%} CER={s['cer']:.2%}")
            print("metadata:",{k:meta[k] for k in ["age","gender","accent","native_speaker","recording_room"]})
            print("edits:",s["edit_operations"])
            digit_rates=[]
            for digit in "0123456789":
                rows=[p for p in s["predictions"] if p["reference"]==digit]
                digit_rates.append(sum(p["correct"] for p in rows)/len(rows))
            plt.figure(figsize=(8,3)); plt.bar(list("0123456789"),np.array(digit_rates)*100)
            plt.ylim(0,100); plt.ylabel("exact (%)"); plt.title(f"speaker {speaker}: per-digit accuracy"); plt.show()

        speaker_dashboard()
        widgets.interact_manual(
            speaker_dashboard,
            speaker=widgets.Dropdown(options=speakers,value="46",description="speaker"),
        )
        """
    ),
    md("## 6. 60 speaker 的区间比 6 speaker 告诉我们更多什么？"),
    code(
        r"""
        old=json.loads((ROOT/"artifacts/fsdd_loso_results.json").read_text(encoding="utf-8"))["aggregate"]["sequence"]["macro_speaker_cer"]
        new=result["single_digit"]["aggregate"]["speaker_macro_cer"]
        labels=["FSDD LOSO\nn=6","AudioMNIST external\nn=60"]
        means=[old["mean"],new["mean"]]
        lower=[means[0]-old["ci95_low"],means[1]-new["ci95_low"]]
        upper=[old["ci95_high"]-means[0],new["ci95_high"]-means[1]]
        plt.figure(figsize=(7,4)); plt.errorbar([0,1],np.array(means)*100,yerr=np.array([lower,upper]).T*100,fmt="o",capsize=8)
        plt.xticks([0,1],labels); plt.ylabel("speaker-mean CER (%)"); plt.title("More independent speakers narrow uncertainty")
        plt.grid(axis="y",alpha=.2); plt.show()
        """
    ),
    md(
        """
        区间变窄不代表 AudioMNIST 更容易；它表示我们对“这 60 位 speaker 的均值”估计更稳定。两个区间还对应不同数据集、模型拟合方式与 track，不能把区间重叠当作严格显著性检验。
        """
    ),
    md("## 7. S/D/I：模型主要怎么错？"),
    code(
        r"""
        tracks=[]
        for key,label in [("single_digit","single"),("multi_digit","multi")]:
            ops=result[key]["aggregate"]["edit_operations"]
            tracks.append((label,ops))
            total=ops["substitution"]+ops["deletion"]+ops["insertion"]
            print(label,{k:ops[k] for k in ["substitution","deletion","insertion"]},"error total",total)
        categories=["substitution","deletion","insertion"]
        x=np.arange(2); bottom=np.zeros(2)
        plt.figure(figsize=(7,4))
        for category in categories:
            values=np.array([row[1][category] for row in tracks])
            plt.bar(x,values,bottom=bottom,label=category); bottom+=values
        plt.xticks(x,[row[0] for row in tracks]); plt.ylabel("edit count"); plt.title("Deletions dominate external errors")
        plt.legend(); plt.show()
        """
    ),
    md(
        """
        单数字有 7,964 次 deletion、4,036 次 substitution、256 次 insertion。大量空输出或漏 token 往往说明 CTC 更偏 blank、声学峰值不够强或域差导致模型没触发，而不是语言模型“不够聪明”。先查声学/前端，再考虑 LM。
        """
    ),
    md("## 8. 混淆矩阵：哪些数字被替成什么？"),
    code(
        r"""
        labels=list("0123456789")+["empty","multi"]
        matrix=np.zeros((10,12),dtype=int)
        for speaker in speakers:
            for row in per[speaker]["predictions"]:
                hyp=row["hypothesis"]
                column=int(hyp) if len(hyp)==1 and hyp in "0123456789" else 10 if hyp=="" else 11
                matrix[int(row["reference"]),column]+=1
        normalized=matrix/matrix.sum(axis=1,keepdims=True)
        fig,ax=plt.subplots(figsize=(10,6)); image=ax.imshow(normalized,cmap="magma",vmin=0,vmax=1,aspect="auto")
        ax.set(xticks=range(12),xticklabels=labels,yticks=range(10),yticklabels=list("0123456789"),xlabel="hypothesis",ylabel="reference",title="Single-digit row-normalized confusion")
        plt.setp(ax.get_xticklabels(),rotation=35,ha="right")
        plt.colorbar(image,ax=ax,label="fraction of reference digit"); plt.tight_layout(); plt.show()
        """
    ),
    md("## 9. 听错误，不要只盯矩阵"),
    code(
        r"""
        def listen_prediction(speaker="46",kind="errors",example=0):
            rows=[p for p in per[speaker]["predictions"] if bool(p["correct"])==(kind=="correct")]
            if not rows: print("这个分组为空"); return
            row=rows[example%len(rows)]
            path=AUDIO_ROOT/row["utterance"]
            wave,sr=sf.read(path,dtype="float32")
            display(Audio(wave,rate=sr))
            print("file",row["utterance"],"reference",row["reference"],"hypothesis",repr(row["hypothesis"]),"edits",row["edit_operations"])
            down=librosa.resample(wave,orig_sr=sr,target_sr=8000,res_type="soxr_hq")
            feat=LogMelFrontend(frontend_config())(torch.from_numpy(down))
            fig,axes=plt.subplots(2,1,figsize=(10,4.5)); axes[0].plot(np.arange(len(wave))/sr,wave); axes[0].set(title="waveform",xlabel="seconds")
            axes[1].imshow(feat.T.numpy(),origin="lower",aspect="auto"); axes[1].set(title="model input: 8 kHz log-Mel",xlabel="frame",ylabel="mel bin")
            plt.tight_layout(); plt.show()

        listen_prediction()
        widgets.interact_manual(
            listen_prediction,
            speaker=widgets.Dropdown(options=speakers,value="46",description="speaker"),
            kind=widgets.ToggleButtons(options=["errors","correct"],value="errors",description="kind"),
            example=widgets.IntSlider(value=0,min=0,max=49,description="example"),
        )
        """
    ),
    md("## 10. 多数字长度分桶"),
    code(
        r"""
        multi=result["multi_digit"]["per_speaker"]
        by_length={length:[] for length in range(1,5)}
        for speaker in speakers:
            for row in multi[speaker]["predictions"]:
                by_length[len(row["reference"])].append(row)
        length_exact=[]; length_cer=[]
        for length,rows in by_length.items():
            exact_rate=sum(r["correct"] for r in rows)/len(rows)
            errors=sum(sum(r["edit_operations"][k] for k in ["substitution","deletion","insertion"]) for r in rows)
            chars=sum(len(r["reference"]) for r in rows)
            length_exact.append(exact_rate); length_cer.append(errors/chars)
            print(length,"digits",len(rows),f"exact={exact_rate:.2%}",f"CER={errors/chars:.2%}")
        fig,axes=plt.subplots(1,2,figsize=(10,3.5)); axes[0].plot(range(1,5),np.array(length_exact)*100,marker="o"); axes[1].plot(range(1,5),np.array(length_cer)*100,marker="o")
        axes[0].set(title="Sequence exact",xlabel="digits",ylabel="exact (%)"); axes[1].set(title="CER",xlabel="digits",ylabel="CER (%)")
        for ax in axes: ax.set_xticks(range(1,5)); ax.grid(alpha=.2)
        plt.tight_layout(); plt.show()
        """
    ),
    md("## 11. 元数据分桶：描述差异不等于解释原因"),
    code(
        r"""
        gender=result["single_digit"]["metadata_slices"]["gender"]
        names=list(gender); values=[gender[n]["speaker_macro_exact_rate"] for n in names]; counts=[gender[n]["speaker_count"] for n in names]
        plt.figure(figsize=(6,3.5)); bars=plt.bar(names,np.array(values)*100)
        for bar,count in zip(bars,counts): plt.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"n={count} speakers",ha="center")
        plt.ylim(0,100); plt.ylabel("speaker-macro exact (%)"); plt.title("Descriptive slice — not a causal gender effect")
        plt.show(); print(gender)
        """
    ),
    md(
        """
        女性 12 人、男性 48 人，并且 gender 与具体 speaker、accent、origin、录音日期/房间可能混杂。这个图能提出“需要更平衡数据和受控研究”的问题，不能证明生理性别导致错误，更不能据此评价所有人群公平性。native speaker 只有 3 人，同样不能作稳定总体推断。

        **排错题：** 若一个 group 只有 1 位 speaker、500 条录音，group 的有效 speaker cluster 数仍是 1，不是 500。
        """
    ),
    md("## 12. 流式一致性、准确率和速度是三条证据"),
    code(
        r"""
        stream=result["streaming"]
        rows=stream["rows"]
        stream_correct=sum(next(iter(r["outputs"].values()))==r["reference"] for r in rows)
        print("chunk invariant:",stream["invariant_speakers"],"/",stream["checked_speakers"])
        print("representative recognition correct:",stream_correct,"/",len(rows))
        print("same implementation output across chunks does not imply correct text")

        perf=result["performance"]["rows"]
        chunks=[r["chunk_ms"] for r in perf]
        p50=[r["latency_ms_p50"] for r in perf]; p95=[r["latency_ms_p95"] for r in perf]
        plt.figure(figsize=(7,3.5)); plt.plot(chunks,p50,marker="o",label="P50"); plt.plot(chunks,p95,marker="o",label="P95")
        plt.xlabel("chunk duration (ms)"); plt.ylabel("utterance inference latency (ms)"); plt.title("Preloaded/resampled CPU benchmark")
        plt.legend(); plt.grid(alpha=.2); plt.show()
        for row in perf: print(row)
        """
    ),
    md(
        """
        60/60 chunk invariant 证明跨 chunk 前端、卷积 cache 和 CTC collapse 的最终行为一致。它不证明文本正确。RTF 远小于 1 说明算得过来，但基准排除了磁盘与重采样，也不等于端到端 partial latency；chunk 更大还会增加等待音频填满的算法延迟。
        """
    ),
    md("## 13. 外测之后，什么还能做，什么不能再声称？"),
    md(
        """
        现在可以做新的 **AudioMNIST adaptation study**：预先按 speaker 划 train/dev/test，比较 speed perturb、SpecAugment、域对抗、CMVN、blank bias、focal/CTC 变体或更强 encoder。这个研究可以回答“怎样适配 AudioMNIST”。

        不能做的是：在刚看过的全部 60 人上反复调到高分，然后把同一批分数继续写成 untouched external test。要确认适配方法对未知世界有效，必须再找一份新来源，或在本次接触前就保留完全不查看的外层 speaker 集（本协议选择的是一次性全量诊断，所以没有留下这种集）。

        这正是实验设计的代价：全量外测给了更精确诊断，却消耗了 AudioMNIST 的“未接触”身份。
        """
    ),
    code(
        r"""
        states=[
            ("FSDD train", "可用于梯度"),
            ("FSDD LOSO", "已接触；可描述内部方差"),
            ("AudioMNIST full", "本次首次外测后已接触"),
            ("future independent source", "尚可作为新盲测"),
        ]
        for i,(dataset,state) in enumerate(states,1): print(f"{i}. {dataset:<26} → {state}")
        """
    ),
    md("## 14. 编程迁移任务"),
    md(
        """
        1. 从空白实现 48→8 kHz 重采样前后的时长、Nyquist 和频谱检查。
        2. 只根据保存的 predictions 重建 exact、CER、S/D/I 和混淆矩阵，禁止重新推理。
        3. 实现 speaker-cluster bootstrap，并与错误的 utterance bootstrap 比较区间宽度。
        4. 找出 deletion 最多的 5 位 speaker，随机听每人 10 条，先提出假设再用时长/RMS/口音分桶验证。
        5. 设计 AudioMNIST speaker-disjoint adaptation split；明确它已是“适配研究”，不是首次盲测。
        6. 为下一份全新外测写 protocol JSON，并在下载音频前提交。
        7. 把 greedy 换成 prefix beam，但只能标为 post-hoc exploratory；不能替换本课预注册主分数。
        8. 测量含磁盘 I/O、重采样、网络与队列的真正端到端延迟，对比本课 inference-only RTF。
        """
    ),
    md("## 15. 120 分综合测试（30 题，每题 4 分）"),
    md(
        """
        1. protocol commit、model commit、score time 三者为何必须有序？
        2. checkpoint SHA256 保护什么？
        3. FSDD 与 AudioMNIST 标签相同为何仍是 domain shift？
        4. 48 kHz 的 Nyquist 是多少？8 kHz 呢？
        5. 直接把 48 kHz samples 当 8 kHz 会造成什么时间错误？
        6. 为什么重采样前需要抗混叠低通？
        7. 为什么不能外测时临时开启 peak normalization？
        8. 单数字 exact 59.30% 的预注册解释标签是什么？
        9. exact 与 CER 分别怎样计算？
        10. 为什么多数字 exact 比单数字低而 CER 接近？
        11. 60 位 speaker 的平均 exact 能代表每个人吗？
        12. speaker 29 与 46 的差异说明什么？
        13. speaker 标准差与均值置信区间回答的问题有何不同？
        14. 为什么 n=60，不是 n=30,000？
        15. t 区间更窄是否等于模型更准确？
        16. 单数字错误中哪种操作最多？可能对应什么模型行为？
        17. 空 hypothesis 在编辑距离里是什么操作？
        18. hypothesis 有两个 digit、reference 一个时至少包含什么操作？
        19. 混淆矩阵按行归一化回答什么？
        20. 听 5 条错误能否估计总体 deletion rate？
        21. 女性分桶较差为什么不能直接作性别因果结论？
        22. native speaker 只有 3 人有什么统计限制？
        23. 60/60 chunk invariant 证明什么、不证明什么？
        24. RTF<1 与首字延迟有什么区别？
        25. 为什么本课 RTF 不包含重采样与磁盘？这会怎样限制解释？
        26. 外测完成后 AudioMNIST 的数据身份发生了什么变化？
        27. 如何合法开展 AudioMNIST adaptation study？
        28. 为什么 adaptation 后还需另一个独立来源确认？
        29. post-hoc beam search 结果应怎样标注？
        30. 为下一份外测列出至少 8 个必须在下载前冻结的项目。

        **通关线：** 96/120，且第 1、5、9、14、16、21、23、26、30 题不得为 0。
        """
    ),
    md(
        """
        <details><summary>评分要点（作答后展开）</summary>

        1. 防止结果驱动规则与模型；2. 绑定确切字节；3. speaker/设备/房间/采样率等输入分布不同；4. 24 kHz 与 4 kHz；5. 时间拉长 6 倍；6. 防 aliasing；7. 与冻结训练前端不一致且受 test 反馈；8. weak transfer；9. 全句正确比例与编辑数/参考字符；10. 任一 token 错使 exact 归零；11. 不能，14%～92.8%；12. 强 speaker heterogeneity；13. 个体离散程度与均值估计不确定性；14. 独立 cluster 是 speaker；15. 不等于；16. deletion，blank/弱 spike/域差；17. 删除；18. 插入；19. 给定 reference 的输出分布；20. 不能；21. 样本不平衡且多重混杂；22. 极不稳定、不能推广总体；23. 实现 chunk 一致，不证明文本正确；24. 总计算吞吐与等待上下文/endpoint 不同；25. 它是 inference-only 下界；26. 已接触测试/诊断数据；27. 重新预注册 speaker-disjoint train/dev/test；28. 防适配到本数据集；29. exploratory secondary analysis；30. 数据版本、模型、checkpoint、前端、重采样、decoder、指标、分桶、线程/硬件、停止/披露等。

        </details>
        """
    ),
    md("## 16. 口头答辩与离场票"),
    md(
        """
        用 12 分钟、不看答案讲清：

        > “为什么这次外测比上一轮 LOSO 更强；Git 怎样证明规则和模型早于结果；48→8 kHz 怎样保持时间并防混叠；59.3% exact 为什么只算弱迁移；60 位 speaker 的方差与区间说明什么；deletion 为何是首要线索；为什么元数据切片不能直接归因；chunk、RTF、accuracy 为什么独立；AudioMNIST 现在还能怎样用于适配、不能再怎样宣传。”

        离场票：写下一个被数据推翻的预测、一个最容易造成测试泄漏的动作、一个你会在下一份外测前冻结的新项目。达到证据等级 3 后再开始 AudioMNIST 适配。
        """
    ),
]


for index, cell in enumerate(cells):
    cell["id"] = f"external-{index:02d}"

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python (learn-asr)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "course": {"stage": "external-generalization", "topic": "preregistered AudioMNIST evaluation", "version": 1},
    },
)
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT)
print(f"Wrote {OUT} with {len(cells)} cells")
