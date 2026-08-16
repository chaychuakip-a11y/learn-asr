from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def M(s): return ("markdown", textwrap.dedent(s).strip())
def C(s): return ("code", textwrap.dedent(s).strip())


SETUP = r'''
from pathlib import Path
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

def find_root():
    here=Path.cwd().resolve()
    for p in [here,*here.parents]:
        if (p/"pyproject.toml").exists(): return p
    raise FileNotFoundError("请从 learn_asr 或 notebooks 目录启动 Jupyter")

ROOT=find_root(); ARTIFACTS=ROOT/"artifacts";ARTIFACTS.mkdir(exist_ok=True)
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
plt.rcParams["figure.figsize"]=(11,4)
print("项目根目录:",ROOT)
'''


def quiz(text):
    return M(f'''<details><summary>展开参考答案</summary>

{text}

</details>''')


lessons={}

lessons[25] = ("模型导出_ONNX与运行时一致性", [
M('''# 第 25 课：模型导出——ONNX、固定接口与运行时一致性

训练代码不是部署接口。本课把一个带显式 cache 的因果声学模型导出为 ONNX，并逐项验证 PyTorch 与 ONNX Runtime 输出。'''),
C(SETUP + r'''
import onnx
import onnxruntime as ort
import torch
from deployment.model import (build_model,export_onnx,compare_torch_onnx,
    FEATURE_DIM,NUM_CLASSES,CHUNK_FRAMES,CACHE_FRAMES)
'''),
M('''## 1. 先固定模型 contract

```text
frames:    float32 [1, 8, 24]
cache:     float32 [1, 2, 24]
logits:    float32 [1, 8, 11]
new_cache: float32 [1, 2, 24]
```

固定 chunk shape 通常更容易优化和部署；代价是客户端必须缓冲并处理最后不足一块的数据。'''),
C(r'''
model=build_model()
x=torch.randn(1,CHUNK_FRAMES,FEATURE_DIM);cache=torch.zeros(1,CACHE_FRAMES,FEATURE_DIM)
with torch.inference_mode(): logits,new_cache=model(x,cache)
print("frames",x.shape,"cache",cache.shape,"logits",logits.shape,"new_cache",new_cache.shape)
'''),
M('''## 2. 使用新的 `torch.onnx.export(..., dynamo=True)` 导出'''),
C(r'''
onnx_path=export_onnx(ARTIFACTS/"streaming_ctc_demo.onnx")
print("saved:",onnx_path,"size KB:",onnx_path.stat().st_size/1024)
model_proto=onnx.load(onnx_path);onnx.checker.check_model(model_proto)
print("opset:",[(x.domain,x.version) for x in model_proto.opset_import])
print("nodes:",[n.op_type for n in model_proto.graph.node])
'''),
M('''## 3. 一致性不是“能加载就算成功”

至少比较多组输入的最大绝对误差、相对误差、余弦相似度和最终解码结果。下面先做单组逐张量比较。'''),
C(r'''
print(compare_torch_onnx(onnx_path))
session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"])
print("inputs:",[(i.name,i.shape,i.type) for i in session.get_inputs()])
print("outputs:",[(o.name,o.shape,o.type) for o in session.get_outputs()])
'''),
M('''## 4. 多 chunk cache 一致性'''),
C(r'''
rng=np.random.default_rng(8); chunks=rng.normal(size=(6,1,CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32)
torch_cache=torch.zeros(1,CACHE_FRAMES,FEATURE_DIM);ort_cache=np.zeros((1,CACHE_FRAMES,FEATURE_DIM),np.float32)
errors=[]
for chunk in chunks:
    with torch.inference_mode(): tlogits,torch_cache=model(torch.from_numpy(chunk),torch_cache)
    ologits,ort_cache=session.run(None,{"frames":chunk,"cache":ort_cache})
    errors.append(np.max(np.abs(tlogits.numpy()-ologits)))
print("per-chunk max errors:",errors)
'''),
M('''## 5. 导出常见坑

- 训练态没有切换 `eval()`；
- 把 Python 控制流或无法导出的算子带进图；
- cache shape/顺序不一致；
- 忘记 opset、输入名称、dtype；
- 只验一组随机输入；
- 导出成功但目标 runtime 不支持某个算子。'''),
M('''## 本课测试

1. ONNX 是训练框架还是模型交换表示？
2. 为什么 cache 必须成为显式输入输出？
3. 固定 chunk shape 的优缺点是什么？
4. 模型能被 ORT 加载是否等于结果正确？
5. 为什么要做连续多 chunk 一致性测试？'''),
quiz('''1. 模型交换/计算图表示。2. 服务跨请求保存状态，runtime 不知道 Python 对象内部状态。3. 易优化但客户端需严格分块和补齐。4. 不等于。5. cache 错误可能第一块看不出来，之后逐步累积。''')
])

lessons[26] = ("量化基础_INT8_PTQ_QAT与误差", [
M('''# 第 26 课：量化基础——INT8、PTQ、QAT 与误差

量化不是简单把 `float32` 强制转成整数，而是用 scale 和 zero-point 建立浮点值与有限整数网格的映射。'''),
C(SETUP + r'''
from ipywidgets import interact, IntSlider, FloatSlider
'''),
M(r'''## 1. Affine quantization

$$q=\operatorname{clamp}(\operatorname{round}(x/s)+z,q_{min},q_{max})$$
$$\hat{x}=s(q-z)$$

`q` 是整数，$\hat{x}$ 是反量化后的近似值。量化误差来自舍入和截断。'''),
C(r'''
def symmetric_quantize(x,bits=8):
    qmax=2**(bits-1)-1;scale=max(np.max(np.abs(x))/qmax,1e-12)
    q=np.clip(np.round(x/scale),-qmax,qmax).astype(np.int32)
    return q,q.astype(np.float32)*scale,scale

x=np.linspace(-2,2,401,dtype=np.float32)
for bits in [8,4,2]:
    q,xhat,s=symmetric_quantize(x,bits)
    print(bits,"bits scale",s,"max error",np.max(np.abs(x-xhat)),"levels",len(np.unique(q)))
'''),
M('''## 2. 交互观察 bit 数与离群值'''),
C(r'''
@interact(bits=IntSlider(min=2,max=8,value=4),outlier=FloatSlider(min=1,max=20,value=2,step=1))
def show(bits=4,outlier=2):
    values=np.concatenate([np.linspace(-1,1,200),[outlier]]).astype(np.float32)
    q,xhat,scale=symmetric_quantize(values,bits)
    plt.scatter(values,xhat,s=12);plt.plot([values.min(),values.max()],[values.min(),values.max()],color="C1")
    plt.xlabel("FP32 value");plt.ylabel("Dequantized value");plt.title(f"{bits}-bit quantization, scale={scale:.4f}");plt.show()
    print("mean abs error",np.mean(np.abs(values-xhat)))
'''),
M('''离群值会扩大 scale，使大量普通值挤在更粗的网格上。校准集、percentile clipping 和 per-channel quantization因此非常重要。'''),
M('''## 3. Per-tensor 与 per-channel'''),
C(r'''
rng=np.random.default_rng(2)
W=np.vstack([rng.normal(0,.05,100),rng.normal(0,.5,100),rng.normal(0,3,100)]).astype(np.float32)
_,Wt,_=symmetric_quantize(W,8)
Wc=np.empty_like(W)
for i,row in enumerate(W): _,Wc[i],_=symmetric_quantize(row,8)
print("per-tensor MAE",np.mean(np.abs(W-Wt)))
print("per-channel MAE",np.mean(np.abs(W-Wc)))
'''),
M('''## 4. PTQ、动态量化、静态量化、QAT

- Dynamic PTQ：权重预先量化，activation 参数运行时计算；容易上手，常用于 RNN/Transformer。
- Static PTQ：用代表性 calibration data 提前统计 activation 范围；运行开销更低，但更依赖校准质量。
- QAT：训练时模拟量化误差，让模型适应；成本更高，通常在 PTQ 精度不达标时使用。
- Weight-only：只量化权重，节省模型大小/带宽，但 activation 仍较高精度。

当前 PyTorch 量化能力正在集中到 `torchao`；后续代码避免把即将迁移的旧 eager API 当成唯一方案。'''),
M('''## 5. ASR 量化必须检查什么

- Logit/CTC posterior 差异；
- Greedy/beam 输出变化；
- CER/WER，而不只是 MSE；
- 长音频 cache 漂移；
- 热词和语言模型融合后的排序变化；
- 模型大小、峰值内存、RTF、P99；
- 目标 CPU/GPU/移动硬件是否有对应低精度 kernel。'''),
M('''## 本课测试

1. scale 越小是否永远越好？
2. 离群值怎样影响 per-tensor 量化？
3. static PTQ 为什么需要 calibration data？
4. QAT 是否应当永远作为第一选择？
5. 模型缩小 4 倍是否保证推理快 4 倍？'''),
quiz('''1. 不是，过小会截断大值。2. 扩大范围和 scale，使普通值误差变粗。3. 提前估计 activation 范围。4. 不，应先尝试成本低的 PTQ 并验证。5. 不保证，取决于低精度 kernel、内存、算子覆盖和量化/反量化开销。''')
])

lessons[27] = ("ONNXRuntime_INT8量化与Benchmark", [
M('''# 第 27 课：ONNX Runtime INT8——量化、模型大小与 Benchmark

本课对第 25 课模型进行动态 INT8 量化，并比较输出误差、文件大小和 CPU 延迟。'''),
C(SETUP + r'''
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic,QuantType
from deployment.model import FEATURE_DIM,CHUNK_FRAMES,CACHE_FRAMES,export_onnx
'''),
M('''## 1. 生成 FP32 与 INT8 模型'''),
C(r'''
fp32=ARTIFACTS/"streaming_ctc_demo.onnx"
if not fp32.exists(): export_onnx(fp32)
int8=ARTIFACTS/"streaming_ctc_demo.int8.onnx"
quantize_dynamic(str(fp32),str(int8),weight_type=QuantType.QInt8)
print("FP32 KB",fp32.stat().st_size/1024,"INT8 KB",int8.stat().st_size/1024,
      "ratio",int8.stat().st_size/fp32.stat().st_size)
'''),
M('''## 2. 不能只看文件大小：先比较输出'''),
C(r'''
sess32=ort.InferenceSession(str(fp32),providers=["CPUExecutionProvider"])
sess8=ort.InferenceSession(str(int8),providers=["CPUExecutionProvider"])
rng=np.random.default_rng(10);errors=[];agreements=[]
for _ in range(100):
    x=rng.normal(size=(1,CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32)
    c=rng.normal(size=(1,CACHE_FRAMES,FEATURE_DIM)).astype(np.float32)
    y32=sess32.run(None,{"frames":x,"cache":c})[0];y8=sess8.run(None,{"frames":x,"cache":c})[0]
    errors.append(np.max(np.abs(y32-y8)));agreements.append(np.mean(y32.argmax(-1)==y8.argmax(-1)))
print("max abs error: median/max",np.median(errors),np.max(errors))
print("frame argmax agreement",np.mean(agreements))
'''),
C(r'''
plt.hist(errors,bins=25);plt.xlabel("Max absolute logit error per sample");plt.ylabel("Samples")
plt.title("FP32 vs INT8 output differences");plt.show()
'''),
M('''## 3. 正确 benchmark：warm-up、重复、分位数'''),
C(r'''
x=rng.normal(size=(1,CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32);c=np.zeros((1,CACHE_FRAMES,FEATURE_DIM),np.float32)
def bench(session,n=1000):
    for _ in range(50):session.run(None,{"frames":x,"cache":c})
    times=[]
    for _ in range(n):
        t=time.perf_counter();session.run(None,{"frames":x,"cache":c});times.append((time.perf_counter()-t)*1000)
    return np.asarray(times)
t32=bench(sess32);t8=bench(sess8)
for name,times in [("FP32",t32),("INT8",t8)]:print(name,"P50/P90/P99 ms",*[np.percentile(times,p) for p in [50,90,99]])
print("median speedup",np.median(t32)/np.median(t8))
'''),
M('''小模型可能量化后没有明显加速，甚至变慢，因为量化开销、算子规模和硬件 kernel 会主导结果。这不是实验失败，而是重要结论：**必须在目标硬件上测量。**'''),
M('''## 4. ASR 精度验收层次

1. 张量误差；
2. frame argmax agreement；
3. Greedy 文本 agreement；
4. Beam + LM 文本 agreement；
5. 全测试集 CER/WER；
6. 噪声、长音频、重复字符、热词等分桶回归。'''),
M('''## 本课测试

1. dynamic quantization 的 activation scale 何时计算？
2. INT8 文件更小是否保证 RTF 更低？
3. 为什么 frame argmax agreement 比 logit MSE更接近 CTC 行为？
4. 量化模型为什么仍需独立 CER/WER 测试？
5. benchmark 为什么要固定线程、硬件和输入 shape？'''),
quiz('''1. 运行时。2. 不保证。3. CTC 解码依赖类别排序，但最终仍需文本评估。4. 小 logit 误差可能改变 beam 排名和文本。5. 否则结果不可复现和公平比较。''')
])

lessons[28] = ("FastAPI_HTTP推理服务与模型契约", [
M('''# 第 28 课：FastAPI HTTP 推理服务——模型契约、健康检查与错误边界

HTTP 适合单块、离线短请求或控制接口；持续音频流更适合下一课的 WebSocket。'''),
C(SETUP + r'''
from fastapi.testclient import TestClient
from deployment.app import app,MODEL_PATH
from deployment.model import FEATURE_DIM,CHUNK_FRAMES,CACHE_FRAMES
client=TestClient(app)
print("loaded model:",MODEL_PATH)
'''),
M('''## 1. 健康检查不是准确率检查

`/health` 证明进程能够加载模型并响应；readiness 还应确认必要资源已就绪。模型质量需要单独的回归集。'''),
C(r'''
r=client.get("/health");print(r.status_code,r.json())
'''),
M('''## 2. 一次显式 cache 的推理请求'''),
C(r'''
rng=np.random.default_rng(1);frames=rng.normal(size=(CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32)
r=client.post("/infer",json={"frames":frames.tolist()})
body=r.json();print("status",r.status_code,"logits shape",np.asarray(body["logits"]).shape,"cache shape",np.asarray(body["cache"]).shape)
'''),
M('''## 3. 客户端负责把 cache 传回去'''),
C(r'''
cache=body["cache"]
frames2=rng.normal(size=(CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32)
r2=client.post("/infer",json={"frames":frames2.tolist(),"cache":cache})
print(r2.status_code,np.asarray(r2.json()["logits"]).shape)
'''),
M('''HTTP 显式 cache 的优点是服务端无会话状态，容易横向扩容；缺点是 cache 反复传输、客户端复杂、容易篡改。WebSocket 可以把 cache 留在连接内。'''),
M('''## 4. 启动真实服务

```powershell
uv run uvicorn deployment.app:app --host 127.0.0.1 --port 8000
```

生产环境还需要请求大小限制、超时、认证、TLS、日志脱敏、进程管理和滚动升级。'''),
M('''## 本课测试

1. `/health` 返回 200 是否代表模型准确？
2. 无状态 HTTP 为什么容易扩容？
3. 显式传输 cache 有什么风险？
4. 为什么不能把任意长度 JSON 一次读入内存？
5. 模型版本应在哪里暴露？'''),
quiz('''1. 不代表。2. 任意实例都能处理下一请求。3. 带宽、客户端状态错误和篡改。4. 可能造成内存/拒绝服务风险。5. health/metadata、日志和每个结果事件中均可包含可追踪版本。''')
])

lessons[29] = ("WebSocket流式服务_会话状态与PGS", [
M('''# 第 29 课：WebSocket 流式服务——会话状态、cache 与 PGS 事件

每个 WebSocket 连接拥有独立的 feature buffer、encoder cache、CTC previous token 和 transcript。'''),
C(SETUP + r'''
from fastapi.testclient import TestClient
from deployment.app import app
from deployment.model import FEATURE_DIM,CHUNK_FRAMES
client=TestClient(app);rng=np.random.default_rng(6)
'''),
M('''## 1. 半块不会立即推理

服务端模型 contract 是 8 帧。先发送 4 帧，服务端缓存；再发送 4 帧才产生一个结果事件。'''),
C(r'''
with client.websocket_connect("/stream") as ws:
    ws.send_json({"features":rng.normal(size=(4,FEATURE_DIM)).tolist()})
    ws.send_json({"features":rng.normal(size=(4,FEATURE_DIM)).tolist()})
    first=ws.receive_json();print("first:",first)
    ws.send_json({"features":rng.normal(size=(CHUNK_FRAMES,FEATURE_DIM)).tolist()})
    second=ws.receive_json();print("second:",second)
    ws.send_json({"eof":True})
    final=ws.receive_json();print("final:",final)
'''),
M('''## 2. 四层状态不要混在一起

```text
前端状态：未成帧的 PCM/feature 尾部
编码器状态：卷积/注意力/RNN cache
解码器状态：CTC prefix、WFST active state、LM history
协议状态：sn、PGS slices、stable/final 标记
```

连接断开时要明确哪些状态销毁，哪些能通过 session id 恢复。'''),
M('''## 3. Backpressure

如果客户端发送速度超过服务端处理速度，队列会无限增长。常见策略：限制每连接缓冲、暂停读取、拒绝新连接、降低 beam、分配独立 worker；不能悄悄丢掉中间音频。'''),
M('''## 4. 当前教学服务为什么接收 feature 而不是 PCM

这是为了单独验证模型 serving、cache 和 PGS。生产系统通常接收二进制 PCM/Opus，并在服务端执行有状态重采样、分帧、Log-Mel 和 CMVN；也可以把前端部署到设备侧，但必须固定特征规范。'''),
M('''## 本课测试

1. 每条 WebSocket 连接应共享还是隔离 decoder state？
2. 为什么不能在 chunk 结束时清空 previous token？
3. EOF 到达时应做哪些 flush？
4. backpressure 为什么影响稳定性？
5. PGS `rpl` 事件在客户端应如何处理？'''),
quiz('''1. 隔离。2. 跨 chunk 重复折叠依赖它。3. 处理尾部帧策略、完成 decoder/endpoint、发送 final 并释放状态。4. 队列增长会增加延迟和内存，最终雪崩。5. 按 rg 替换历史 slice，而不是简单追加。''')
])

lessons[30] = ("生产部署_容器并发监控与验收", [
M('''# 第 30 课：生产部署——容器、并发、监控、灰度与验收

模型能够运行只是部署的起点。本课建立上线前的完整检查清单。'''),
C(SETUP + r'''
from concurrent.futures import ThreadPoolExecutor
import onnxruntime as ort
from deployment.model import FEATURE_DIM,CHUNK_FRAMES,CACHE_FRAMES
model_path=ARTIFACTS/"streaming_ctc_demo.int8.onnx"
session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
'''),
M('''## 1. 并发不是简单增加线程

ORT 自身有 intra-op/inter-op 线程池，服务框架也有 worker/线程。层层都开满会发生 oversubscription，造成 P99 抖动。应在固定 CPU 配额下联合调参。'''),
C(r'''
rng=np.random.default_rng(4);x=rng.normal(size=(1,CHUNK_FRAMES,FEATURE_DIM)).astype(np.float32);c=np.zeros((1,CACHE_FRAMES,FEATURE_DIM),np.float32)
def one(_):
    t=time.perf_counter();session.run(None,{"frames":x,"cache":c});return (time.perf_counter()-t)*1000
for workers in [1,2,4,8]:
    with ThreadPoolExecutor(max_workers=workers) as pool: times=np.array(list(pool.map(one,range(300))))
    print("workers",workers,"throughput req/s",len(times)/(times.sum()/1000),"P50/P99 ms",np.percentile(times,[50,99]))
'''),
M('''上面的 throughput 算法是教学近似：并发请求有重叠，严谨吞吐应使用整批 wall-clock 时间；延迟仍按每请求统计。请在练习中修正它。'''),
M('''## 2. Docker 构建

项目已经准备 `deployment/Dockerfile`：

```powershell
docker build -f deployment/Dockerfile -t learn-asr:0.1 .
docker run --rm -p 8000:8000 learn-asr:0.1
```

镜像应固定 lockfile、模型 checksum、运行时版本和非 root 用户。教学 Dockerfile 保持简洁，生产版还需要最小权限和镜像扫描。'''),
M('''## 3. 监控四类信号

- Traffic：连接数、音频秒数、请求率、并发；
- Errors：协议错误、模型异常、断线、OOM；
- Latency：first partial/stable/final、P50/P90/P99、队列等待；
- Saturation：CPU/GPU、内存、线程池、队列、网络。

ASR 还必须监控 RTF、空结果率、平均输出长度、CER/WER 抽样和热词误触发。'''),
M('''## 4. 灰度与回滚

新模型不能只比较总体 WER：需要固定回归集、领域分桶、量化版对照、shadow traffic、少量灰度、版本化指标和一键回滚。服务事件应携带 model/config/LM/graph 版本，才能追查问题。'''),
M('''## 5. 安全与隐私

语音可能含敏感信息。必须考虑 TLS、认证授权、日志脱敏、音频保留策略、地域与合规、依赖供应链，以及对超长输入、连接洪泛和恶意 ONNX 模型的限制。'''),
M('''## 最终部署验收题

1. 量化部署必须比较哪三类结果？
2. RTF、首字延迟和 P99 为什么必须同时报告？
3. WebSocket worker 重启后 session state 怎么办？
4. 为什么模型、LM、WFST graph 必须分别版本化？
5. 如何证明新模型可以安全回滚？
6. 生产镜像为什么要使用 `uv.lock`？'''),
quiz('''1. 精度/文本质量、性能延迟、资源与模型大小。2. 它们分别描述计算效率、交互体验和尾部稳定性。3. 明确终止并让客户端重连，或把可恢复状态放入有版本的外部状态层。4. 任一组件变化都可能改变结果，必须可追踪。5. 保留旧镜像/配置、兼容协议、灰度指标和自动回滚阈值。6. 固定可复现的依赖解析。'''),
M('''## 完整课程终点

至此课程覆盖：声音与特征 → 编码器 → CTC → 流式 → PGS/RTF → LM/WFST → ONNX → INT8 → HTTP/WebSocket → 容器与生产验收。下一阶段应选择真实可泛化模型和数据集，完成一次端到端工程项目。''')
])


def write(number,slug,cells):
    nb=nbf.v4.new_notebook()
    nb.metadata={"kernelspec":{"display_name":"Python (learn-asr)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}}
    nb.cells=[nbf.v4.new_markdown_cell(x) if k=="markdown" else nbf.v4.new_code_cell(x) for k,x in cells]
    path=NB_DIR/f"{number:02d}_{slug}.ipynb";nbf.write(nb,path);print(path)


for number,(slug,cells) in lessons.items():write(number,slug,cells)
