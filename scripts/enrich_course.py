from pathlib import Path
import re
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
NB_DIR=ROOT/"notebooks"
TAG="course-upgrade-v2"

# concepts, prediction/debug scenario, coding task, cross-lesson connection
TOPICS={
1:(["采样率","幅值与归一化","dBFS与参考值"],"把 8 kHz 文件误标为 16 kHz 播放","生成 440 Hz 正弦波并验证周期、峰值、RMS 和 dBFS","解释本课量怎样进入第 2 课分帧"),
2:(["frame/window","hop","窗函数与频谱泄漏"],"把 hop 从 10 ms 改成 25 ms","实现不依赖 librosa 的分帧并验证首尾索引","说明分帧参数怎样决定 STFT 时间分辨率"),
3:(["DFT","FFT","频率 bin"],"观察长度不是整周期的正弦波 FFT","用显式 DFT 与 np.fft.rfft 对照误差","说明 FFT 为什么没有改变 DFT 的数学结果"),
4:(["STFT","声谱图","时间—频率分辨率"],"窗长增大但 hop 不变","手写 STFT 并重建时间/频率坐标","连接第 3 课单帧频谱与整段声谱图"),
5:(["Mel 标度","滤波器组","Log-Mel"],"把 n_mels 从 40 改为 8 或 128","画出 Mel 三角滤波器并检查覆盖范围","解释 dB、log power 和模型输入的关系"),
6:(["声学前端封装","数值一致性","参数契约"],"手写前端与 librosa 出现一帧差异","为前端写 shape、范围和误差断言","把前端输出契约交给第 7 课 batch"),
7:(["padding","length","mask"],"把补零区域错误地计入均值","实现 lengths_to_mask 并测试边界","说明 length 如何贯穿 CTC loss"),
8:(["tensor shape","线性层","最小编码器"],"交换 T 与 F 后仍能运行某些操作","为每层打印和断言 shape","说明 encoder 输出为何需要 CTC head"),
9:(["Conv1d","stride 下采样","感受野"],"连续两层 stride=2 后忘记更新 length","计算任意卷积栈的输出长度和感受野","连接下采样比例与 CTC 最短路径"),
10:(["未知对齐","blank","CTC collapse"],"目标含连续重复 token","枚举小词表路径并按折叠文本聚合概率","说明 CTC 为什么适合逐帧流式输出"),
11:(["扩展标签","前向动态规划","LogSumExp"],"长语音中直接连乘概率","用枚举验算 forward algorithm","连接动态规划总概率与 CTCLoss"),
12:(["CTCLoss shape","input/target lengths","zero_infinity"],"下采样后 input_lengths 仍使用原长度","写 batch audit 检查重复 token 和合法最短 T","说明 loss 异常怎样定位到数据管线"),
13:(["greedy path","prefix beam","p_blank/p_nonblank"],"beam size=1 且存在多路径累加翻盘","从空白实现 prefix beam 并与穷举对照","为第 20 课接入语言模型保留扩展点"),
14:(["真实音频训练","CTC spike","过拟合实验"],"训练集全对但测试说话人失败","加入独立 speaker split 并计算 CER","说明双向 GRU 为什么阻碍严格流式"),
15:(["音频 chunk","跨块缓存","在线分帧"],"chunk 边界落在一帧中间","用不规则 chunk 验证离线/在线帧完全一致","把前端 cache 与 encoder cache 区分"),
16:(["因果卷积","encoder cache","chunk attention"],"普通 same padding 偷看未来","比较多种 chunk 下离线/在线最大误差","把右上下文换算为算法延迟"),
17:(["PGS apd","PGS rpl/rg","片段状态"],"同一 sn 重复或 rpl 包乱序到达","实现幂等 PGSBuffer 并加入非法 rg 测试","连接 decoder partial 与客户端显示"),
18:(["RTF","first/final latency","P50/P99"],"平均 RTF 很低但排队很长","写 warm-up、重复、分位数 benchmark","把 chunk、右上下文和 endpoint 纳入延迟预算"),
19:(["N-gram","平滑与回退","困惑度"],"测试句含未见 bigram","实现 add-k bigram 并计算句子 log probability","说明 LM 分数怎样进入 beam/WFST"),
20:(["浅融合","LM scale/插入项","hotword"],"热词 bonus 过大","扫描 alpha/beta/bonus 并画准确率—偏置曲线","连接 Prefix Beam 与 G 图"),
21:(["FSA/FST","负对数权重","最短路径"],"把概率和 cost 的方向写反","实现小型 weighted shortest path","为 L、G composition 建立标签接口"),
22:(["L 图","G 图","HCLG 与 CTC TLG"],"相邻 FST symbol table 不一致","画出每张图输入/输出标签并检查 composition","解释经典 HCLG 哪些部分不能直接套 CTC"),
23:(["active state","lattice","stable prefix"],"chunk 结束时清空 decoder state","验证分块与整段 beam 最佳结果一致","连接 lattice rescoring、PGS 与热词"),
24:(["系统 contract","端到端状态","验收矩阵"],"单元模块都正确但时间戳整体漂移","为整条流式链路定义输入输出和回归测试","把前 23 课组织成一次可复现实验"),
25:(["ONNX 导出","显式 cache","运行时一致性"],"第一块一致但连续 cache 逐块漂移","导出后用多组多 chunk 比较 PyTorch/ORT","为 INT8 量化建立 FP32 基线"),
26:(["scale/zero-point","PTQ/QAT","per-tensor/per-channel"],"离群值扩大 activation 范围","实现对称量化并比较 bit 数和量化误差","连接量化误差与 CTC 排名变化"),
27:(["动态 INT8","精度回归","目标硬件 benchmark"],"INT8 文件更小但运行更慢","比较大小、logit、argmax、P50/P99 和 RTF","为服务选择 FP32/INT8 artifact"),
28:(["HTTP contract","health/readiness","无状态 cache"],"客户端传回错误 shape/cache","为 /infer 添加输入失败测试和模型版本","对比 HTTP 与下一课 WebSocket 状态"),
29:(["WebSocket session","backpressure","PGS event"],"客户端发送速度高于推理速度","测试两连接状态隔离、EOF 与断线","连接模型 cache、decoder state 和协议 state"),
30:(["容器复现","并发线程","监控灰度回滚"],"ORT 线程池与服务 worker 同时开满","用 wall-clock 修正并发吞吐 benchmark","设计量化模型的灰度和回滚门槛"),
31:(["PCM contract","DC/clipping","重采样与通道"],"16-bit PCM 被按错误端序读取","写输入审计器输出 peak/RMS/DC/clipping","把可靠 PCM 交给增强与 VAD"),
32:(["SNR","降噪","AGC与增强"],"AGC 在静音区放大背景噪声","构造不同 SNR 并比较增强前后 ASR 特征","连接 NS/AGC 与 VAD 顺序"),
33:(["VAD","endpoint 状态机","hangover"],"词内短停顿触发错误 final","实现起点/终点触发并扫描延迟—误切","把 endpoint 等待加入第 18 课延迟"),
34:(["AEC reference","NLMS","double-talk"],"近端说话时滤波器继续快速更新","实现冻结更新的 double-talk 基线","说明 AEC 为什么必须在模型前端早期"),
35:(["到达时间差","delay-and-sum","空间混叠"],"两个通道未对齐就平均","估计 delay、对齐并比较 SNR","把阵列输出接入单通道 NS/VAD"),
36:(["前端顺序","模块状态","时间戳映射"],"会话 reset 遗漏导致状态串话","为每个模块列出 cache/flush/旁路测试","把麦克风前端接到第 15 课流式特征"),
37:(["token 时间戳","speaker embedding","DER"],"enrollment 和 test 使用同一录音","做 speaker-disjoint 验证并报告混淆","连接说话人片段与最终文本结构"),
38:(["标点恢复","ITN","文本规范化"],"逐字数字规则把“一百零二”变错","为号码、数量、日期设计分类规则和反例","把 spoken/verbatim/normalized 三层分开"),
39:(["置信度校准","N-best","语义重排"],"NLU 高置信掩盖低声学置信","画 reliability diagram 并计算 ECE","连接 lattice 候选与拒识/澄清策略"),
40:(["intent","slot","dialogue state"],"两个用户共享状态导致槽位串话","实现 schema 校验和需要确认的策略","把 ASR 置信度传入业务决策"),
41:(["证据转写","受约束 LLM","任务安全"],"LLM 把不确定数字改成常见数字","校验结构化输出并禁止无证据实体","画出从 PCM 到 action 的版本化审计链")
}

PHASES={
range(1,7):"声音与声学特征",range(7,10):"张量与编码器",range(10,15):"CTC 核心",
range(15,19):"流式 ASR",range(19,24):"语言模型与 WFST",range(24,25):"系统综合",
range(25,31):"量化与部署",range(31,37):"音频信号前端",range(37,42):"后处理与语义"
}
def phase(n):
    return next(v for r,v in PHASES.items() if n in r)
def hours(n):
    if n<=9:return "2～4 小时"
    if 10<=n<=14:return "4～6 小时"
    return "3～5 小时"

def cell(kind,text):
    c=nbf.v4.new_markdown_cell(text) if kind=="md" else nbf.v4.new_code_cell(text)
    c.metadata["tags"]=[TAG]
    return c

def navigation(n,title,concepts):
    prereq="无；只需要会运行 Notebook" if n==1 else f"完成第 {n-1} 课；如果前测低于 2/3，先回看上一课小结"
    return f'''<!-- {TAG} -->
## 学习导航

| 项目 | 内容 |
|---|---|
| 所属阶段 | {phase(n)} |
| 建议投入 | {hours(n)}，可分 2～3 次完成 |
| 前置要求 | {prereq} |
| 本课核心 | {'、'.join(concepts)} |
| 完成标准 | 能口头解释核心概念；独立完成强化题；从空白重写核心函数 |

高效顺序：**先回答前测 → 预测代码结果 → 再运行 → 修改一个变量 → 关闭答案复现 → 次日回忆。**
'''

def pretest(concepts):
    return f'''<!-- {TAG} -->
## 课前诊断（先不要运行代码）

1. 分别用一句话解释：{concepts[0]}、{concepts[1]}、{concepts[2]}。
2. 画出这三个概念之间的输入—输出关系。
3. 写下你最不确定的一点，并给出一个暂时猜测。

自评：答对 0～1 题先复习前置课；答对 2 题可以正常学习；3 题都能讲清楚则直接挑战代码和迁移题。
'''

def exercises(n,concepts,scenario,challenge,connection):
    return f'''<!-- {TAG} -->
## 强化练习：第 {n} 课专属题库

请先把答案写进新的 Markdown/Code cell，再展开自评标准。

### A. 基础回忆

1. 不看上文，分别定义 `{concepts[0]}`、`{concepts[1]}`、`{concepts[2]}`。
2. 哪一个量/状态是本课最容易在模块边界丢失的？它的单位和 shape 是什么？
3. 本课至少写出两个“看起来能运行，但结果其实错误”的例子。

### B. 预测与推理

4. 场景：**{scenario}**。先预测现象，再说明原因，最后给出一项可以验证猜测的指标。
5. 改变本课最关键参数的 0.5×、1×、2×，分别预测准确率、延迟、内存或数值误差怎样变化。
6. 画一张最小数据流图，在每条边标出 dtype、shape、时间单位或概率/代价方向。

### C. 编程与排错

7. 编程任务：**{challenge}**。至少加入正常、边界、错误输入三类测试。
8. 故意制造一个 off-by-one、shape、状态未 reset 或数值稳定性错误；记录错误现象和定位过程。
9. 不看本课实现，从空白 cell 重写最核心函数，并用原实现作数值对照。

### D. 迁移与表达

10. 跨课任务：**{connection}**。
11. 用 90 秒向没有学过 ASR 的人解释本课；禁止只念术语，必须举一个数字或生活例子。
12. 写出一个生产系统中会监控的指标，以及它异常时优先检查的三处位置。

<details><summary>展开自评标准</summary>

- 每题 0～2 分：0=无法回答；1=方向正确但缺少单位、边界或验证；2=解释完整且能用代码/数字验证。
- 24 分满分：达到 19 分再进入下一课；15～18 分次日重做错题；低于 15 分回看本课图和核心代码。
- 第 4 题必须包含“预测—原因—指标”，第 7～9 题必须真正运行测试，第 10 题必须明确上下游 contract。
- 核心答案至少应正确使用：{concepts[0]}、{concepts[1]}、{concepts[2]}。

</details>
'''

def review(n,concepts):
    return f'''<!-- {TAG} -->
## 间隔复习与离场票

### 离场票（现在完成）

- [ ] 我能不用笔记解释 {'、'.join(concepts)}。
- [ ] 我能说出本课最常见的错误及其观测现象。
- [ ] 我能从空白重写一个核心函数，并通过至少 3 个测试。
- [ ] 我能说明本课对上一层和下一层接口的影响。

### 复习时间表

- **明天（5 分钟）**：闭卷写出三个核心概念和一个公式/shape。
- **7 天后（15 分钟）**：重做第 4、7、10 题，不运行原答案。
- **30 天后（20 分钟）**：从真实音频或随机张量重新构造一个最小实验。

把错题记录到根目录 `LEARNING_LOG.md`。不要只写“不会”，要写：原判断、证据、正确规则、下次检查动作。
'''

for path in sorted(NB_DIR.glob("[0-9][0-9]_*.ipynb")):
    if path.stem.endswith("_已运行"):continue
    m=re.match(r"(\d\d)_",path.name)
    if not m:continue
    n=int(m.group(1)); concepts,scenario,challenge,connection=TOPICS[n]
    nb=nbf.read(path,as_version=4)
    nb.cells=[c for c in nb.cells if TAG not in c.metadata.get("tags",[]) and TAG not in "".join(c.get("source",[]))]
    title=("".join(nb.cells[0].source).splitlines() or [path.stem])[0].lstrip("# ")
    nb.cells[1:1]=[cell("md",navigation(n,title,concepts)),cell("md",pretest(concepts))]
    nb.cells.extend([cell("md",exercises(n,concepts,scenario,challenge,connection)),cell("md",review(n,concepts))])
    nb.metadata.setdefault("course",{}).update({"lesson":n,"phase":phase(n),"upgrade":TAG})
    _,nb=nbf.validator.normalize(nb,strip_invalid_metadata=True)
    nbf.write(nb,path)
    print(f"upgraded {path.name}: {len(nb.cells)} cells")
