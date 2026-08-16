from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def M(s): return ("markdown", textwrap.dedent(s).strip())
def C(s): return ("code", textwrap.dedent(s).strip())


SETUP = r'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def find_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists(): return p
    raise FileNotFoundError("请从 learn_asr 或 notebooks 目录启动 Jupyter")

ROOT = find_root()
BLANK = "∅"
plt.rcParams["figure.figsize"] = (11, 4)
print("项目根目录:", ROOT)
'''


def quiz(answer):
    return M(f'''<details><summary>展开参考答案</summary>

{answer}

</details>''')


lessons = {}

lessons[11] = ("CTC前向算法_动态规划与LogSumExp", [
M('''# 第 11 课：CTC 前向算法——不用枚举指数级路径

本课目标：从暴力枚举出发，亲手推导 CTC forward algorithm，并理解为什么真实实现必须使用 log-space。'''),
C(SETUP + r'''
from itertools import product
from ipywidgets import interact, IntSlider

def collapse(path):
    merged=[]; previous=None
    for x in path:
        if x != previous: merged.append(x)
        previous=x
    return "".join(x for x in merged if x != BLANK)
'''),
M('''## 1. 暴力枚举为什么不能用于真实训练

若类别数为 $C$、时间步数为 $T$，路径数是 $C^T$。下面的小例子只有 `{blank,A,B}`。'''),
C(r'''
for T in [4, 10, 20, 50, 100]:
    print(f"T={T:3d} 路径数=3^{T}={3**T:,}")
'''),
M('''## 2. 用枚举得到正确答案，作为动态规划的“验算器”'''),
C(r'''
symbols=[BLANK,"A","B"]
probs=np.array([
 [0.60,0.20,0.15,0.55],
 [0.30,0.65,0.20,0.10],
 [0.10,0.15,0.65,0.35],
])
def path_prob(path):
    ids={s:i for i,s in enumerate(symbols)}
    return np.prod([probs[ids[s],t] for t,s in enumerate(path)])
valid=[p for p in product(symbols,repeat=4) if collapse(p)=="AB"]
brute=sum(path_prob(p) for p in valid)
print("合法路径数:",len(valid),"P(AB|X)=",brute)
for p in sorted(valid,key=path_prob,reverse=True)[:8]: print(f"{path_prob(p):.6f}",p)
'''),
M(r'''## 3. 扩展标签序列

将目标 `AB` 扩展为 `∅ A ∅ B ∅`。状态只能：

- 停在原状态；
- 前进一步；
- 当前符号不是 blank 且不等于前两个符号时，跨两步。

这个限制恰好保证 CTC 的重复合并规则。'''),
C(r'''
def extend_target(target):
    out=[BLANK]
    for token in target: out += [token,BLANK]
    return out

def ctc_forward(probabilities, labels, target):
    ext=extend_target(target); S=len(ext); T=probabilities.shape[1]
    idx={s:i for i,s in enumerate(labels)}
    alpha=np.zeros((T,S))
    alpha[0,0]=probabilities[idx[ext[0]],0]
    if S>1: alpha[0,1]=probabilities[idx[ext[1]],0]
    for t in range(1,T):
        for s,sym in enumerate(ext):
            total=alpha[t-1,s]
            if s-1>=0: total += alpha[t-1,s-1]
            if s-2>=0 and sym!=BLANK and sym!=ext[s-2]: total += alpha[t-1,s-2]
            alpha[t,s]=total*probabilities[idx[sym],t]
    total=alpha[-1,-1]+alpha[-1,-2]
    return total,alpha,ext

dp,alpha,ext=ctc_forward(probs,symbols,"AB")
print("扩展标签:",ext)
print("暴力枚举:",brute,"动态规划:",dp,"差值:",abs(brute-dp))
'''),
C(r'''
fig,ax=plt.subplots(figsize=(10,4))
im=ax.imshow(alpha.T,aspect="auto",cmap="magma")
for t in range(alpha.shape[0]):
    for s in range(alpha.shape[1]): ax.text(t,s,f"{alpha[t,s]:.3f}",ha="center",va="center",color="white")
ax.set(xticks=range(4),xticklabels=[f"t{i+1}" for i in range(4)],yticks=range(len(ext)),yticklabels=ext,
       xlabel="Time",ylabel="Extended-target state",title="CTC forward probabilities α(t,s)")
fig.colorbar(im,ax=ax,label="Forward probability"); plt.show()
'''),
M('''## 4. 交互：逐列揭示动态规划表'''),
C(r'''
@interact(step=IntSlider(min=1,max=4,value=1,description="已计算时间步"))
def reveal(step=1):
    shown=alpha[:step].T
    fig,ax=plt.subplots(figsize=(2+step*1.6,4))
    im=ax.imshow(shown,aspect="auto",cmap="magma",vmin=0,vmax=alpha.max())
    for s in range(shown.shape[0]):
        for t in range(shown.shape[1]): ax.text(t,s,f"{shown[s,t]:.3f}",ha="center",va="center",color="white")
    ax.set(xticks=range(step),xticklabels=[f"t{i+1}" for i in range(step)],yticks=range(len(ext)),yticklabels=ext)
    ax.set(title=f"Forward table through t={step}",xlabel="Time",ylabel="State"); plt.show()
'''),
M(r'''## 5. 为什么要用 LogSumExp

路径概率是许多小数连乘，长语音会下溢成 0。在 log-space 中：乘法变加法，加法用 `logsumexp`。'''),
C(r'''
tiny=np.float32(0.01)
print("float32 中 0.01^100 =",tiny**100)
def logsumexp(values):
    values=np.asarray(values,dtype=np.float64); m=values.max()
    return m+np.log(np.exp(values-m).sum())
a,b=np.log(1e-200),np.log(2e-200)
print("稳定 log(exp(a)+exp(b)) =",logsumexp([a,b]))
print("再 exp 回去 =",np.exp(logsumexp([a,b])))
'''),
M('''## 本课测试

1. 目标长度为 $U$，扩展序列长度是多少？
2. 为什么某些状态可以跨两步？
3. 最终概率为什么取最后两个状态之和？
4. 动态规划复杂度大约是多少？
5. `logsumexp` 解决什么问题？'''),
quiz('''1. $2U+1$。 2. 允许跳过 blank，但不能由此错误地产生相邻重复 token。 3. 路径可以结束在最后一个 token 或末尾 blank。 4. $O(TU)$，远小于枚举。 5. 防止许多小概率连乘产生数值下溢。'''),
M('''## 下一课

把手算结果交给 PyTorch：理解 `CTCLoss` 的每个 shape、length 与常见报错。''')
])

lessons[12] = ("PyTorch_CTCLoss_shape_length与排错", [
M('''# 第 12 课：PyTorch CTCLoss——shape、length 与排错

目标：能够独立构造 `torch.nn.CTCLoss` 输入，理解 reduction、blank、无穷 loss 和梯度。'''),
C(SETUP + r'''
import torch
import torch.nn.functional as F
torch.manual_seed(7)
'''),
M('''## 1. 四个输入及其形状

- `log_probs`: `[T, N, C]`
- `targets`: 拼接的一维标签，或 `[N,S]`
- `input_lengths`: 每个样本有效输出时间步
- `target_lengths`: 每个标签的有效长度'''),
C(r'''
T,N,C=6,2,4
logits=torch.randn(T,N,C,requires_grad=True)
log_probs=logits.log_softmax(dim=-1)
targets=torch.tensor([1,2, 2,3,1],dtype=torch.long)
input_lengths=torch.tensor([6,5],dtype=torch.long)
target_lengths=torch.tensor([2,3],dtype=torch.long)
print("log_probs",log_probs.shape,"targets",targets.shape)
loss_fn=torch.nn.CTCLoss(blank=0,reduction="none",zero_infinity=False)
losses=loss_fn(log_probs,targets,input_lengths,target_lengths)
print("per-sample loss:",losses)
'''),
M('''## 2. Loss 就是负对数概率

`reduction="none"` 时，第一个样本的 `exp(-loss)` 就是目标文本所有合法路径的概率和。'''),
C(r'''
print("P(target_0|X_0) =",torch.exp(-losses[0]).item())
loss=losses.mean(); loss.backward()
print("梯度 shape:",logits.grad.shape,"梯度是否有限:",torch.isfinite(logits.grad).all().item())
'''),
M('''## 3. 最常见的长度错误：下采样后忘记更新 input_lengths'''),
C(r'''
def conv_len(L,k=3,s=2,p=1,d=1): return (L+2*p-d*(k-1)-1)//s+1
raw_lengths=torch.tensor([101,80,57])
after1=conv_len(raw_lengths); after2=conv_len(after1)
print("原帧数:",raw_lengths.tolist())
print("两层 stride=2 后:",after2.tolist())
'''),
M('''如果 logits 只有 26 步，却仍传入原始 101 步，CTCLoss 会报长度超过输入；如果时间轴被压得比标签合法最短路径还短，loss 会是 `inf`。'''),
C(r'''
# 目标 [1,1] 至少需要 3 步；这里故意只给 2 步
bad_lp=torch.randn(2,1,3).log_softmax(-1)
bad_target=torch.tensor([1,1])
for zero in [False,True]:
    fn=torch.nn.CTCLoss(blank=0,reduction="none",zero_infinity=zero)
    print("zero_infinity=",zero,"loss=",fn(bad_lp,bad_target,torch.tensor([2]),torch.tensor([2])).item())
'''),
M('''`zero_infinity=True` 能避免训练被 `inf` 污染，但它只是把不可能样本的 loss/gradient 置零；真正应该检查下采样比例、标签长度和数据。'''),
M('''## 4. 建立统一的 batch 检查器'''),
C(r'''
def min_ctc_steps(seq):
    return len(seq)+sum(a==b for a,b in zip(seq,seq[1:]))

def audit_ctc_batch(input_lengths,target_list,num_classes,blank=0):
    problems=[]
    for i,(T,target) in enumerate(zip(input_lengths.tolist(),target_list)):
        if any(x==blank for x in target): problems.append((i,"target 中含 blank"))
        if any(x<0 or x>=num_classes for x in target): problems.append((i,"token id 越界"))
        need=min_ctc_steps(target)
        if T<need: problems.append((i,f"T={T} < 最少需要 {need}"))
    return problems

print(audit_ctc_batch(torch.tensor([4,2]),[[1,2],[1,1]],4))
'''),
M('''## 本课测试

1. 为什么 `log_probs` 要先做 `log_softmax`？
2. `[T,N,C]` 三个维度分别是什么？
3. blank 能否出现在训练 target 中？
4. `zero_infinity=True` 是否真正修复了数据？
5. 两层 stride=2 后，长度大约缩短多少倍？'''),
quiz('''1. CTCLoss 接受对数概率。2. 时间、batch、类别。3. 不应出现。4. 没有，只是让不可能样本不破坏梯度。5. 约 4 倍，精确值应逐层套卷积长度公式。'''),
M('''## 下一课：Greedy 与 CTC Prefix Beam Search。''')
])

lessons[13] = ("CTC_Greedy与PrefixBeamSearch", [
M('''# 第 13 课：CTC 解码——从 Greedy 到 Prefix Beam Search

目标：理解为什么“每帧第一名”不等于“文本第一名”，并逐行实现 Prefix Beam Search。'''),
C(SETUP + r'''
from collections import defaultdict
from itertools import product
from ipywidgets import interact, IntSlider

symbols=[BLANK,"A","B"]
probs=np.array([[.40,.35,.40],[.35,.40,.10],[.25,.25,.50]])

def collapse(path):
    out=[];prev=None
    for x in path:
        if x!=prev and x!=BLANK: out.append(x)
        prev=x
    return "".join(out)
'''),
M('''## 1. Greedy decoding'''),
C(r'''
path=[symbols[i] for i in probs.argmax(0)]
print("greedy path:",path,"text:",collapse(path),"path probability:",np.prod(probs.max(0)))
'''),
M('''## 2. 用穷举计算每个文本的精确概率'''),
C(r'''
totals=defaultdict(float)
for path in product(symbols,repeat=probs.shape[1]):
    p=np.prod([probs[symbols.index(s),t] for t,s in enumerate(path)])
    totals[collapse(path)]+=p
for text,p in sorted(totals.items(),key=lambda x:x[1],reverse=True): print(repr(text),f"{p:.5f}")
'''),
M('''## 3. Prefix Beam Search 的两个概率

对每个前缀保存：

- `p_b(prefix)`：以 blank 结尾的路径概率和
- `p_nb(prefix)`：以非 blank 结尾的路径概率和

必须分开保存，才能正确处理 `A A` 与 `A blank A`。'''),
C(r'''
def prefix_beam_search(P, labels, beam_size=5):
    beam={"":(1.0,0.0)}
    history=[]
    for t in range(P.shape[1]):
        nxt=defaultdict(lambda:[0.0,0.0])
        for prefix,(pb,pnb) in beam.items():
            for i,c in enumerate(labels):
                p=P[i,t]
                if c==BLANK:
                    nxt[prefix][0]+=(pb+pnb)*p
                elif prefix and c==prefix[-1]:
                    nxt[prefix][1]+=pnb*p
                    nxt[prefix+c][1]+=pb*p
                else:
                    nxt[prefix+c][1]+=(pb+pnb)*p
        beam=dict(sorted(nxt.items(),key=lambda kv:sum(kv[1]),reverse=True)[:beam_size])
        history.append(beam)
    return beam,history

beam,history=prefix_beam_search(probs,symbols,beam_size=20)
for prefix,(pb,pnb) in sorted(beam.items(),key=lambda x:sum(x[1]),reverse=True)[:8]:
    print(repr(prefix),f"total={pb+pnb:.5f}",f"blank={pb:.5f}",f"nonblank={pnb:.5f}")
'''),
M('''## 4. 交互观察 beam 如何随时间变化'''),
C(r'''
@interact(t=IntSlider(min=1,max=probs.shape[1],value=1,description="已处理帧"))
def show_beam(t=1):
    items=sorted(history[t-1].items(),key=lambda x:sum(x[1]),reverse=True)
    names=[k or "<empty>" for k,_ in items]; vals=[sum(v) for _,v in items]
    plt.barh(names[::-1],vals[::-1]); plt.xlabel("Prefix probability");plt.title(f"Beam after frame {t}");plt.show()
'''),
M('''## 5. Beam size 是速度与准确率旋钮

beam 太小会过早剪掉后来可能翻盘的前缀；beam 太大增加 CPU、内存和延迟。工业系统还常用相对阈值 beam pruning。'''),
C(r'''
for k in [1,2,3,5,20]:
    b,_=prefix_beam_search(probs,symbols,k)
    best=max(b.items(),key=lambda x:sum(x[1]))
    print("beam",k,"->",repr(best[0]),sum(best[1]))
'''),
M('''## 本课测试

1. Greedy 搜索的是路径还是文本？
2. 为什么 prefix 要分 `p_b` 和 `p_nb`？
3. beam size 越大是否一定更适合实时系统？
4. blank 会不会出现在最终前缀中？
5. 后面接语言模型时，应在什么时机增加 token 的 LM 分数？'''),
quiz('''1. 单条逐帧路径。2. 为了正确处理重复 token。3. 不一定，会增加计算和延迟。4. 不会，blank 只改变概率状态。5. 只有当前 token 真正扩展了输出前缀时。''')
])

lessons[14] = ("真实音频上的最小CTC训练实验", [
M('''# 第 14 课：真实开源音频上的最小 CTC 训练实验

我们使用 Free Spoken Digit Dataset 的真实语音，构造一个很小的可过拟合实验。重点是看懂完整数据流，不追求泛化性能。'''),
C(SETUP + r'''
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
torch.manual_seed(3); np.random.seed(3)
'''),
M('''## 1. 构造多数字语音

把 0～9 的开源单数字录音随机拼接，中间插入短静音，同时保存字符标签。'''),
C(r'''
files={str(i):ROOT/"data"/"spoken_digits_parts"/f"{i}_jackson_0.wav" for i in range(10)}
waves={}; sr0=None
for d,p in files.items():
    y,sr=sf.read(p); y=y.astype(np.float32); sr0=sr
    waves[d]=y/(np.max(np.abs(y))+1e-8)

def make_example(text):
    silence=np.zeros(int(sr0*.06),np.float32)
    return np.concatenate([z for i,d in enumerate(text) for z in ([waves[d]] if i==len(text)-1 else [waves[d],silence])])

texts=[str(i) for i in range(10)] + ["12","21","34","43","56","65","78","87","90","09","11","22"]
print("样本数",len(texts),"示例",texts[:5],"采样率",sr0)
'''),
M('''## 2. Log-Mel 与 batch'''),
C(r'''
def feat(text):
    y=make_example(text)
    m=librosa.feature.melspectrogram(y=y,sr=sr0,n_fft=256,win_length=200,hop_length=80,n_mels=24,power=2,center=False)
    return torch.tensor(np.log(m+1e-6).T,dtype=torch.float32)
features=[feat(t) for t in texts]
mean=torch.cat(features).mean(0); std=torch.cat(features).std(0).clamp_min(1e-5)
features=[(x-mean)/std for x in features]
lengths=torch.tensor([len(x) for x in features])
padded=pad_sequence(features,batch_first=True)
targets=torch.tensor([int(c)+1 for t in texts for c in t])
target_lengths=torch.tensor([len(t) for t in texts])
print("features [N,T,F]",padded.shape,"targets",targets.shape)
'''),
M('''## 3. 小型双向 GRU + CTC head

双向 GRU 是为了让小实验容易收敛；它不是流式编码器。第 16 课会改为因果结构。'''),
C(r'''
class TinyCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(24,32,batch_first=True,bidirectional=True)
        self.head=nn.Linear(64,11)  # 0 blank, 1..10 digits
    def forward(self,x):
        return self.head(self.rnn(x)[0])

model=TinyCTC(); opt=torch.optim.Adam(model.parameters(),lr=8e-3)
ctc=nn.CTCLoss(blank=0,zero_infinity=True)
loss_curve=[]
for epoch in range(160):
    opt.zero_grad(); logits=model(padded)
    loss=ctc(logits.log_softmax(-1).transpose(0,1),targets,lengths,target_lengths)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
    loss_curve.append(loss.item())
print("初始/最终 loss:",loss_curve[0],loss_curve[-1])
plt.plot(loss_curve);plt.xlabel("Epoch");plt.ylabel("CTC loss");plt.title("Tiny CTC overfitting experiment");plt.show()
'''),
M('''## 4. Greedy 查看训练集结果'''),
C(r'''
def decode(ids):
    out=[];prev=None
    for x in ids:
        if x!=0 and x!=prev: out.append(str(x-1))
        prev=x
    return "".join(out)
with torch.no_grad(): pred=model(padded).argmax(-1)
for i,t in enumerate(texts): print(f"target={t:>2} pred={decode(pred[i,:lengths[i]].tolist()):>4}")
'''),
M('''## 5. 看 blank 如何占据大多数帧'''),
C(r'''
i=texts.index("12")
with torch.no_grad(): p=model(padded[i:i+1]).softmax(-1)[0,:lengths[i]].numpy().T
plt.imshow(p,aspect="auto",origin="lower",cmap="magma")
plt.yticks(range(11),[BLANK]+list("0123456789"));plt.xlabel("Frame");plt.ylabel("CTC class")
plt.title("CTC posterior for target '12'");plt.colorbar(label="Probability");plt.show()
'''),
M('''## 本课测试

1. 这里为什么可以说是“过拟合实验”？
2. 为什么 target 的数字 id 要从 1 开始？
3. 双向 GRU 为什么不能严格流式？
4. 训练后 blank 占很多帧是否必然表示模型失败？
5. 真正评估泛化还缺少什么？'''),
quiz('''1. 训练和展示使用同一小批样本。2. id=0 留给 blank。3. 当前输出依赖未来帧。4. 不一定，这是 CTC 常见的尖峰输出。5. 独立训练/验证/测试划分、更多说话人和 CER/WER。''')
])

lessons[15] = ("流式特征_Chunk与缓存", [
M('''# 第 15 课：流式特征提取——Chunk、帧边界与缓存

目标：把任意大小的音频块送入系统，同时产生与离线分帧一致的帧，不丢样本、不重复计算。'''),
C(SETUP + r'''
import soundfile as sf
from ipywidgets import interact, IntSlider
y,sr=sf.read(ROOT/"data"/"spoken_digits_parts"/"3_jackson_0.wav")
y=y.astype(np.float32); frame_length=round(.025*sr); hop=round(.010*sr)
print(sr,frame_length,hop,len(y))
'''),
M('''## 1. 离线分帧作为标准答案'''),
C(r'''
def offline_frames(x,L,H):
    if len(x)<L:return np.empty((0,L),x.dtype)
    n=1+(len(x)-L)//H
    return np.stack([x[i*H:i*H+L] for i in range(n)])
reference=offline_frames(y,frame_length,hop)
print("离线帧数",len(reference))
'''),
M('''## 2. 有状态 StreamingFramer'''),
C(r'''
class StreamingFramer:
    def __init__(self,L,H): self.L,self.H=L,H; self.buffer=np.empty(0,np.float32)
    def accept(self,chunk):
        self.buffer=np.concatenate([self.buffer,np.asarray(chunk,dtype=np.float32)])
        out=[]
        while len(self.buffer)>=self.L:
            out.append(self.buffer[:self.L].copy()); self.buffer=self.buffer[self.H:]
        return np.stack(out) if out else np.empty((0,self.L),np.float32)

for chunk_ms in [7,20,100,333]:
    size=max(1,round(sr*chunk_ms/1000)); fr=StreamingFramer(frame_length,hop);parts=[]
    for start in range(0,len(y),size): parts.append(fr.accept(y[start:start+size]))
    got=np.concatenate(parts) if parts else np.empty_like(reference)
    print(f"chunk={chunk_ms:3d} ms frames={len(got):3d} max_error={np.max(np.abs(got-reference)):.1e}")
'''),
M('''## 3. 为什么必须缓存

25 ms 窗、10 ms hop 意味着相邻帧重叠 15 ms。chunk 边界经常切在一帧中间；直接对每块单独做 STFT 会丢掉跨边界帧。'''),
C(r'''
@interact(chunk_ms=IntSlider(min=5,max=100,value=30,step=5,description="chunk ms"))
def draw_boundaries(chunk_ms=30):
    duration=len(y)/sr; starts=np.arange(0,duration,hop/sr); chunks=np.arange(0,duration,chunk_ms/1000)
    fig,ax=plt.subplots(figsize=(11,2.4))
    for s in starts[:35]: ax.plot([s,s+frame_length/sr],[.55,.55],color="C0",alpha=.35)
    for c in chunks: ax.axvline(c,color="C1",alpha=.8)
    ax.set(xlim=(0,min(duration,.4)),ylim=(0,1),yticks=[],xlabel="Time (s)",title="Blue frame spans; orange chunk boundaries")
    plt.show()
'''),
M('''## 4. 在线特征还要注意

- `center=False` 更容易定义真实时间边界；
- 全句均值方差归一化偷看未来，不严格在线；
- 在线 CMVN 需要累计统计量或固定训练集统计量；
- 最后不足一帧的尾部要规定丢弃还是补零；
- 音频采集时钟、重采样器也可能有状态。'''),
M('''## 本课测试

1. 25 ms window、10 ms hop 的重叠是多少？
2. chunk=20 ms 时为什么仍可能暂时产不出第一帧？
3. chunk 大小必须是 hop 的整数倍吗？
4. 全句 CMVN 为什么不流式？
5. 缓存长度是否永远固定为 `window-hop`？'''),
quiz('''1. 15 ms。2. 第一帧需要收满 25 ms。3. 不必，有状态缓冲器可以处理任意大小。4. 它需要未来整句统计量。5. 简单分帧器消费后通常保留未消费尾部，长度会变化但小于 window；其他前端模块的状态另算。''')
])

lessons[16] = ("流式编码器_因果卷积与ChunkAttention", [
M('''# 第 16 课：流式编码器——因果卷积、缓存与 Chunk Attention

CTC head 可以逐帧输出，但编码器若查看无限未来，系统仍不流式。本课用数值实验验证“离线与分块结果一致”。'''),
C(SETUP + r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipywidgets import interact, IntSlider
torch.manual_seed(1)
'''),
M('''## 1. 因果卷积只看当前和过去'''),
C(r'''
conv=nn.Conv1d(3,5,kernel_size=3,bias=False)
x=torch.randn(1,3,17)
offline=conv(F.pad(x,(2,0)))

def stream_conv(x,chunk_size):
    cache=torch.zeros(x.size(0),x.size(1),2); outputs=[]
    for s in range(0,x.size(-1),chunk_size):
        chunk=x[:,:,s:s+chunk_size]; joined=torch.cat([cache,chunk],dim=-1)
        outputs.append(conv(joined)); cache=joined[:,:,-2:]
    return torch.cat(outputs,dim=-1)

for cs in [1,2,4,7,20]: print(cs,(stream_conv(x,cs)-offline).abs().max().item())
'''),
M('''缓存的是输入历史，不是把整段过去重新计算。kernel=3、dilation=1 时需要 2 个历史位置；多层卷积要分别维护各层缓存。'''),
M('''## 2. Chunk Attention mask

严格 causal attention 只能看左侧；chunk attention 允许当前块内部互相看，并可保留有限或无限左上下文。'''),
C(r'''
def chunk_mask(T,chunk,left_chunks):
    mask=np.zeros((T,T),dtype=int)
    for q in range(T):
        current=q//chunk; first=max(0,current-left_chunks); lo=first*chunk; hi=min(T,(current+1)*chunk)
        mask[q,lo:hi]=1
    return mask

@interact(chunk=IntSlider(min=1,max=8,value=4),left_chunks=IntSlider(min=0,max=4,value=1))
def show_mask(chunk=4,left_chunks=1):
    m=chunk_mask(24,chunk,left_chunks)
    plt.imshow(m,origin="lower",cmap="Blues",vmin=0,vmax=1)
    plt.xlabel("Key frame");plt.ylabel("Query frame");plt.title("Allowed attention positions");plt.show()
'''),
M('''## 3. 右上下文与算法延迟

允许当前帧查看未来 $R$ 帧，帧移 10 ms，则至少增加约 $R×10$ ms 等待。chunk attention 常常还要先收满一个块。准确率、吞吐量和延迟要联合衡量。'''),
C(r'''
hop_ms=10
for right in [0,4,8,16,32]: print(f"right context={right:2d} frames -> about {right*hop_ms:3d} ms lookahead")
'''),
M('''## 本课测试

1. CTC + 双向 LSTM 是否严格流式？
2. causal conv 的左 padding 与普通 same padding有什么区别？
3. cache 的主要工程价值是什么？
4. chunk 越大通常对上下文和延迟各有什么影响？
5. 流式结果与离线结果必须完全相同吗？'''),
quiz('''1. 否。2. causal 只在左侧补，不引入未来。3. 复用历史计算。4. 上下文通常更充分，但等待和单块计算可能增加。5. 不一定；若离线模型使用更多未来信息，两者可能不同，应分别评估。''')
])

lessons[17] = ("PGS动态修正_apd_rpl_rg", [
M('''# 第 17 课：PGS 动态修正——apd、rpl、rg 与实时字幕

本课把流式 API 返回的碎片恢复成用户看到的文本。这里的 PGS 指动态修正结果协议，而不是声学模型。'''),
C(SETUP + r'''
from ipywidgets import interact, IntSlider
'''),
M('''## 1. 三个字段

- `pgs="apd"`：追加本片结果；
- `pgs="rpl"`：替换历史片段；
- `rg=[a,b]`：替换返回序号区间（边界语义必须以具体 API 文档为准）。

识别结果会修正，是因为新音频和语言上下文改变了之前的判断。'''),
C(r'''
events=[
 {"sn":1,"pgs":"apd","text":"今天"},
 {"sn":2,"pgs":"apd","text":"天气"},
 {"sn":3,"pgs":"apd","text":"真"},
 {"sn":4,"pgs":"apd","text":"热"},
 {"sn":5,"pgs":"rpl","rg":[3,4],"text":"真不错"},
 {"sn":6,"pgs":"apd","text":"。"},
]

def apply_pgs(events):
    slices={}
    snapshots=[]
    for e in events:
        if e["pgs"]=="rpl":
            a,b=e["rg"]
            for sn in range(a,b+1): slices.pop(sn,None)
        slices[e["sn"]]=e["text"]
        snapshots.append("".join(slices[k] for k in sorted(slices)))
    return snapshots

for e,text in zip(events,apply_pgs(events)): print(e,"=>",text)
'''),
M('''## 2. 交互播放服务端返回'''),
C(r'''
snapshots=apply_pgs(events)
@interact(step=IntSlider(min=1,max=len(events),value=1,description="返回序号"))
def replay(step=1):
    for i in range(step): print(f"event {i+1}:",events[i])
    print("\n用户界面显示:",snapshots[step-1])
'''),
M('''## 3. 工业客户端不能只做字符串追加

至少要处理：重复包、乱序包、断线重连、未知 `rg`、最终标记、标点片段以及 UI 光标位置。推荐保存结构化 slice，而不是只保存一个不断拼接的大字符串。'''),
C(r'''
class PGSBuffer:
    def __init__(self): self.parts={}; self.seen=set(); self.final=False
    def accept(self,event):
        sn=event["sn"]
        if sn in self.seen: return self.text()
        self.seen.add(sn)
        if event.get("pgs")=="rpl":
            a,b=event["rg"]
            for old in range(a,b+1): self.parts.pop(old,None)
        self.parts[sn]=event.get("text","")
        self.final=self.final or event.get("ls",False)
        return self.text()
    def text(self): return "".join(self.parts[k] for k in sorted(self.parts))

b=PGSBuffer()
for e in events: print(b.accept(e))
print("重复发送最后一包:",b.accept(events[-1]))
'''),
M('''## 4. PGS 与 CTC 的关系

CTC prefix beam search 产生随时间变化的候选；PGS 是把“追加/替换”变化传给客户端的一种协议。CTC 不强制使用 PGS，PGS 也不限定后端必须是 CTC。'''),
M('''## 本课测试

1. `rpl` 为什么不能当成追加？
2. `rg=[2,5]` 通常表示什么？
3. 客户端为什么要按 `sn` 保存片段？
4. PGS 是否等于 CTC 解码算法？
5. partial 文本是否应该立即写入不可修改的业务记录？'''),
quiz('''1. 它会令旧假设失效。2. 替换第 2～5 次返回结果，仍需服从具体接口定义。3. 便于替换、乱序和去重。4. 不是，它是结果更新协议。5. 通常不应，应等待 stable/final 或建立可修订记录。''')
])

lessons[18] = ("RTF实时率_延迟吞吐与基准测试", [
M('''# 第 18 课：RTF 实时率、延迟、吞吐与基准测试

本课目标：不再用一个 RTF 数字概括全部性能，能够设计可信的流式 ASR benchmark。'''),
C(SETUP + r'''
import time
from ipywidgets import interact, FloatSlider
'''),
M(r'''## 1. RTF 定义

$$RTF=\frac{\text{处理时间}}{\text{音频时长}}$$

- RTF=1：处理 1 秒音频需要 1 秒；
- RTF=0.2：处理 10 秒音频约需 2 秒；
- RTF>1：单路处理追不上音频输入。

注意：离线批处理 RTF、单路流式 RTF、服务器并发吞吐不是同一个指标。'''),
C(r'''
@interact(audio_s=FloatSlider(min=1,max=60,value=10,step=1),rtf=FloatSlider(min=.05,max=2,value=.3,step=.05))
def rtf_calc(audio_s=10,rtf=.3):
    compute=audio_s*rtf
    print(f"音频 {audio_s:.1f}s × RTF {rtf:.2f} = 计算 {compute:.2f}s")
    print("单路速度判断:","快于实时" if rtf<1 else "无法追上实时" if rtf>1 else "刚好实时")
'''),
M('''## 2. 实测代码必须 warm-up 和重复'''),
C(r'''
rng=np.random.default_rng(0); A=rng.normal(size=(256,256)).astype(np.float32);B=A.copy()
for _ in range(3): A@B
times=[]
for _ in range(20):
    t0=time.perf_counter(); A@B; times.append(time.perf_counter()-t0)
print("median ms",np.median(times)*1000,"p90 ms",np.percentile(times,90)*1000)
'''),
M('''## 3. RTF 低，不代表用户立刻看到文字

端到端延迟可能包括：采集 chunk、右上下文、排队、特征、模型、解码、网络、稳定策略和 endpoint。'''),
C(r'''
rng=np.random.default_rng(4)
components={"chunk wait":160,"right context":80,"network":rng.normal(35,12,1000),"compute":rng.normal(45,10,1000),"stabilize":rng.gamma(2,45,1000)}
total=components["chunk wait"]+components["right context"]+components["network"]+components["compute"]+components["stabilize"]
plt.hist(total,bins=35);plt.axvline(np.percentile(total,50),color="C1",label="P50");plt.axvline(np.percentile(total,99),color="C3",label="P99")
plt.xlabel("End-to-end latency (ms)");plt.ylabel("Requests");plt.title("Simulated latency distribution");plt.legend();plt.show()
for p in [50,90,95,99]: print(f"P{p}={np.percentile(total,p):.1f} ms")
'''),
M('''## 4. 推荐记录的指标

- 单路流式 RTF、并发总吞吐；
- first partial、first stable、final latency；
- P50/P90/P99，不只平均值；
- CER/WER 与延迟联合曲线；
- CPU/GPU、线程数、batch、beam、音频长度、是否含 I/O；
- 冷启动与热启动分别测试。'''),
M('''## 本课测试

1. 60 秒音频处理 12 秒，RTF 是多少？
2. RTF=0.1 是否保证首字延迟小于 100 ms？
3. 为什么要报告 P99？
4. batch=32 的离线 RTF 能否代表单路流式？
5. beam 变大通常会怎样影响准确率、RTF 和延迟？'''),
quiz('''1. 0.2。2. 不能，系统可能等待 chunk、未来上下文或 endpoint。3. 平均值会隐藏尾部慢请求。4. 不能。5. 搜索更充分可能提高准确率，但通常增加计算、RTF 和延迟。''')
])

lessons[19] = ("Ngram语言模型_概率困惑度与回退", [
M('''# 第 19 课：N-gram 语言模型——概率、平滑、困惑度与回退

声学模型回答“听起来像什么”，语言模型回答“哪串 token 更自然”。'''),
C(SETUP + r'''
from collections import Counter
from math import log,exp
from ipywidgets import interact, FloatSlider
corpus=["今天天气很好","今天天气不错","明天天气很好","今天心情很好","北京天气很好","北京天气不错"]
'''),
M('''## 1. Bigram：下一个字符只看前一个字符'''),
C(r'''
uni=Counter();bi=Counter();vocab=set()
for s in corpus:
    seq=["<s>"]+list(s)+["</s>"];vocab.update(seq[1:])
    for a,b in zip(seq,seq[1:]): uni[a]+=1;bi[a,b]+=1
V=len(vocab)
def p_bigram(b,a,k=.1): return (bi[a,b]+k)/(uni[a]+k*V)
for b in ["天","心","北","好"]: print(f"P({b}|今)={p_bigram(b,'今'):.4f}")
'''),
M('''## 2. 句子概率要在 log-space 相加'''),
C(r'''
def sentence_logp(s,k=.1):
    seq=["<s>"]+list(s)+["</s>"]
    return sum(log(p_bigram(b,a,k)) for a,b in zip(seq,seq[1:]))
for s in ["今天天气很好","今天心情不错","北京心情很好"]: print(s,sentence_logp(s))
'''),
M(r'''## 3. 困惑度 Perplexity

$$PPL=\exp\left(-\frac{1}{N}\sum_i\log P(w_i|history)\right)$$

越低表示模型对测试文本越不“意外”，但不同 tokenization、词表和测试集的 PPL 不宜直接比较。'''),
C(r'''
def ppl(s): return exp(-sentence_logp(s)/(len(s)+1))
for s in ["今天天气很好","今天心情不错","北京心情很好"]: print(s,ppl(s))
'''),
M('''## 4. 平滑为什么必要'''),
C(r'''
for k in [0,1e-3,.1,1.0]:
    try: print("k",k,"P(北|今)",p_bigram("北","今",k))
    except ZeroDivisionError: print("division error")
'''),
M('''真实系统常用 modified Kneser–Ney、backoff 与 ARPA 格式。本课的 add-k 只是看懂概率用的教学版本。'''),
M('''## 5. LM scale 与 acoustic score'''),
C(r'''
candidates={"今天天气很好":-9.8,"今天心情很好":-9.2}
@interact(alpha=FloatSlider(min=0,max=2,value=.5,step=.1,description="LM scale"))
def combine(alpha=.5):
    for s,am in candidates.items(): print(s,"AM",am,"LM",sentence_logp(s),"total",am+alpha*sentence_logp(s))
'''),
M('''## 本课测试

1. 语言模型是否直接听声音？
2. 未见过的 bigram 为什么需要平滑？
3. PPL 越低是否总能保证 WER 越低？
4. LM scale 太大有什么风险？
5. 字级 LM 和词级 LM 的词典需求有何不同？'''),
quiz('''1. 不，它对 token 序列打分。2. 避免概率为零。3. 不保证，解码组合、领域和声学候选都会影响 WER。4. 模型可能忽视声音而偏向常见句。5. 词级通常依赖分词与 OOV 处理；字级词表更简单但序列更长。''')
])

lessons[20] = ("CTC结合语言模型_Hotword与分数融合", [
M('''# 第 20 课：CTC + 语言模型——Prefix Beam、LM scale 与 Hotword

目标：在“真正扩展输出 token”时加入 LM 分数，并理解插入惩罚和热词偏置。'''),
C(SETUP + r'''
from collections import defaultdict
from ipywidgets import interact, FloatSlider, IntSlider
labels=[BLANK,"天","田","气"]
P=np.array([[.55,.10,.50,.10,.55],[.22,.42,.15,.10,.10],[.18,.38,.15,.10,.10],[.05,.10,.20,.70,.25]])
bigram={("天","气"):.75,("田","气"):.18,("<s>","天"):.55,("<s>","田"):.35}
def lm_prob(prefix,c): return bigram.get(((prefix[-1] if prefix else "<s>"),c),.05)
'''),
M('''## 1. 融合分数

常见形式：`acoustic + α × LM + β × token_count + hotword_bonus`。不同实现使用概率或负对数代价，符号方向必须统一。'''),
C(r'''
def decode(P,beam_size=8,alpha=.0,beta=.0,hotword="",hot_bonus=0.0):
    beam={"":(1.0,0.0,0.0)}  # pb,pnb,额外log分数
    for t in range(P.shape[1]):
        nxt=defaultdict(lambda:[0.0,0.0,-np.inf])
        for pref,(pb,pnb,extra) in beam.items():
            for i,c in enumerate(labels):
                p=P[i,t]
                if c==BLANK:
                    nxt[pref][0]+=(pb+pnb)*p; nxt[pref][2]=max(nxt[pref][2],extra)
                elif pref and c==pref[-1]:
                    nxt[pref][1]+=pnb*p; nxt[pref][2]=max(nxt[pref][2],extra)
                    new=pref+c; add=alpha*np.log(lm_prob(pref,c))+beta+(hot_bonus if hotword and new.endswith(hotword) else 0)
                    nxt[new][1]+=pb*p; nxt[new][2]=max(nxt[new][2],extra+add)
                else:
                    new=pref+c; add=alpha*np.log(lm_prob(pref,c))+beta+(hot_bonus if hotword and new.endswith(hotword) else 0)
                    nxt[new][1]+=(pb+pnb)*p; nxt[new][2]=max(nxt[new][2],extra+add)
        score=lambda item: np.log(sum(item[1][:2])+1e-30)+item[1][2]
        beam={k:tuple(v) for k,v in sorted(nxt.items(),key=score,reverse=True)[:beam_size]}
    score=lambda item: np.log(sum(item[1][:2])+1e-30)+item[1][2]
    return sorted(beam.items(),key=score,reverse=True)

for alpha in [0,.5,1.0]: print("alpha",alpha,"best",decode(P,alpha=alpha)[0][0])
'''),
M('''## 2. 交互调 LM scale、插入项和热词'''),
C(r'''
@interact(alpha=FloatSlider(min=0,max=2,value=.5,step=.1),beta=FloatSlider(min=-1,max=1,value=0,step=.1),hot_bonus=FloatSlider(min=0,max=3,value=0,step=.25))
def tune(alpha=.5,beta=0,hot_bonus=0):
    result=decode(P,alpha=alpha,beta=beta,hotword="天气",hot_bonus=hot_bonus)
    for pref,state in result[:5]: print(pref,state)
'''),
M('''## 3. Hotword 不是无条件替换

热词应该只在声学候选仍合理时加有限 bonus。过强会把不相关语音强行识别成热词。生产系统需要在领域召回率和误触发率之间调参。'''),
M('''## 本课测试

1. LM 分数为什么只在前缀真正扩展时加入？
2. LM scale=0 表示什么？
3. 正的 token insertion bonus 通常偏向长还是短输出？
4. 热词权重越大是否越好？
5. 为什么调参必须同时看 WER/CER 与延迟？'''),
quiz('''1. blank 和不产生新 token 的重复不应重复计算 LM。2. 不使用 LM。3. 偏向更长输出。4. 不是，过大会误触发。5. 更复杂的搜索可能提高准确率但增加计算和延迟。''')
])

lessons[21] = ("FSA与WFST_状态弧权重和最短路径", [
M('''# 第 21 课：FSA 与 WFST——状态、弧、权重和最短路径

先用几十行 Python 实现最小图搜索。理解结构后，再看 OpenFst/Kaldi 的高性能实现。'''),
C(SETUP + r'''
from dataclasses import dataclass
import heapq
'''),
M('''## 1. FSA 与 FST

- FSA/acceptor：一条弧读入一个符号，判断序列是否合法并累计权重。
- FST/transducer：一条弧同时有 input label 和 output label，可完成 token→词、音素→词等映射。
- `ε`：不消耗或不产生普通符号。'''),
C(r'''
@dataclass(frozen=True)
class Arc:
    src:int; dst:int; ilabel:str; olabel:str; cost:float

arcs=[Arc(0,1,"n","你",.2),Arc(1,2,"h","好",.3),Arc(0,3,"n","泥",.8),Arc(3,2,"h","好",.4)]
for a in arcs: print(a)
'''),
M(r'''## 2. 权重常用负对数代价

若路径概率相乘，令 `cost=-log(p)` 后，整条路径的 cost 相加；最佳路径就是最短路径。'''),
C(r'''
def shortest_path(arcs,start,final):
    out={}
    for a in arcs: out.setdefault(a.src,[]).append(a)
    q=[(0.0,start,[])];best={start:0.0}
    while q:
        cost,s,path=heapq.heappop(q)
        if s==final:return cost,path
        if cost!=best[s]:continue
        for a in out.get(s,[]):
            nc=cost+a.cost
            if nc<best.get(a.dst,float("inf")):
                best[a.dst]=nc;heapq.heappush(q,(nc,a.dst,path+[a]))
    return float("inf"),[]
cost,path=shortest_path(arcs,0,2)
print("best cost",cost,"output","".join(a.olabel for a in path))
'''),
M('''## 3. 画出 transducer'''),
C(r'''
positions={0:(0,0),1:(1,1),3:(1,-1),2:(2,0)}
fig,ax=plt.subplots(figsize=(9,4))
for a in arcs:
    x1,y1=positions[a.src];x2,y2=positions[a.dst]
    ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="->",color="0.4"))
    ax.text((x1+x2)/2,(y1+y2)/2+.1,f"{a.ilabel}:{a.olabel}/{a.cost}",ha="center")
for s,(x,y) in positions.items():
    ax.scatter(x,y,s=900,color="C0" if s!=2 else "C2");ax.text(x,y,str(s),ha="center",va="center")
ax.set(xlim=(-.3,2.3),ylim=(-1.6,1.6),title="A tiny weighted transducer");ax.axis("off");plt.show()
'''),
M('''## 4. 三个核心优化操作

- composition：把前一个 FST 的输出与后一个 FST 的输入连接；
- determinization：让同一状态对同一输入不再有多条竞争弧；
- minimization：在保持行为等价时减少状态和弧。

它们不是“画图技巧”，而是让大型解码图可运行的关键。'''),
M('''## 本课测试

1. FSA 与 FST 的主要差异是什么？
2. 为什么使用 `-log(probability)`？
3. epsilon 表示普通空格吗？
4. 最短路径对应概率最大还是最小？
5. determinization/minimization 的目标是什么？'''),
quiz('''1. FST 同时定义输入和输出映射。2. 把概率连乘变成代价相加。3. 不是，是不消费/不输出普通符号的空转移。4. 概率最大，因为负对数代价最小。5. 保持等价的同时减少歧义和图规模，提高搜索效率。''')
])

lessons[22] = ("WFST组合_L_G_HCLG与CTC解码图", [
M('''# 第 22 课：WFST 组合——L、G、经典 HCLG 与 CTC 解码图

目标：知道每张图负责什么，并避免把经典 HMM 解码图原样套到 CTC。'''),
C(SETUP),
M('''## 1. L：词典 transducer

`L` 把音素或模型 token 序列映射成词。例如：

```text
jin tian  → 今天
qi        → 气
tian qi   → 天气
```

一个词可能有多个发音，一个发音也可能产生歧义，因此它天然适合用 transducer 表示。'''),
M('''## 2. G：语言模型 acceptor

`G` 限制并给词序列打分，例如 `今天天气` 比 `今天田气` 代价低。N-gram backoff 通常也编码成特殊的 epsilon/回退弧。'''),
C(r'''
words=["今天","天气","很好"]
G={("<s>","今天"):.1,("今天","天气"):.2,("天气","很好"):.15,
   ("<s>","天气"):1.5,("天气","今天"):2.0}
def graph_cost(seq):
    prev="<s>";cost=0
    for w in seq: cost+=G.get((prev,w),3.0);prev=w
    return cost
for seq in [["今天","天气","很好"],["天气","今天"]]:print(seq,graph_cost(seq))
'''),
M('''## 3. 经典 Kaldi HCLG

```text
H：HMM 拓扑/transition-id → 上下文相关音素
C：上下文相关音素 → 普通音素
L：音素 → 词
G：词语言模型

H ∘ C ∘ L ∘ G
```

这是传统 HMM-DNN 系统的结构。`G` 是语言模型，`L` 是发音词典。'''),
M('''## 4. CTC 解码图不一定叫 HCLG

端到端 CTC 已经没有传统 HMM state 拓扑和 context-dependent phone 的同一含义。常见概念图可能是：

```text
T：CTC blank/repeat topology
L：token/phone → word lexicon（字级系统可非常简单）
G：word/character language model

T ∘ L ∘ G
```

不同工具包命名不同。核心问题不是背缩写，而是逐张确认输入符号、输出符号和权重。'''),
M('''## 5. Composition 的接口思维

只有当前一张图的 output label 能与后一张图的 input label 匹配，两张图才能组合。任何 WFST 排错都先检查 symbol table 和标签空间。'''),
C(r'''
layers=[("CTC posterior","token id"),("T","normalized token"),("L","word"),("G","word score")]
fig,ax=plt.subplots(figsize=(11,2.5))
for i,(name,out) in enumerate(layers):
    ax.scatter(i,0,s=1800,color=f"C{i}");ax.text(i,0,name,ha="center",va="center")
    if i<len(layers)-1: ax.annotate(out,(i+1-.18,0),(i+.18,0),arrowprops=dict(arrowstyle="->"),ha="center")
ax.set(xlim=(-.5,len(layers)-.5),ylim=(-.7,.7),title="Conceptual CTC decoding pipeline");ax.axis("off");plt.show()
'''),
M('''## 本课测试

1. `L` 和 `G` 分别负责什么？
2. HCLG 中的 H/C 是否能不加解释地套到 CTC？
3. 字级 CTC 是否一定需要复杂发音词典？
4. composition 失败首先检查什么？
5. 为什么要把大图 determinize/minimize？'''),
quiz('''1. L 做发音/token 到词映射，G 对词/token 序列建模。2. 不能，模型拓扑不同。3. 不一定。4. 相邻图的输入/输出标签空间和 symbol table。5. 减小图和搜索状态，提高解码速度。''')
])

lessons[23] = ("流式Beam_WFST状态_Lattice与热词", [
M('''# 第 23 课：流式 Beam/WFST 状态、Lattice 与热词

目标：理解 chunk 结束并不意味着句子结束；解码器必须保存搜索状态，并能输出候选 lattice/N-best。'''),
C(SETUP + r'''
from collections import defaultdict
symbols=[BLANK,"A","B"]
P=np.array([[.6,.15,.55,.15,.6,.2],[.3,.7,.25,.15,.2,.1],[.1,.15,.2,.7,.2,.7]])
'''),
M('''## 1. 跨 chunk 保存 prefix beam'''),
C(r'''
class StreamingPrefixBeam:
    def __init__(self,beam_size=5): self.beam={"":(1.,0.)};self.beam_size=beam_size
    def accept(self,Pchunk):
        for t in range(Pchunk.shape[1]):
            nxt=defaultdict(lambda:[0.,0.])
            for pref,(pb,pnb) in self.beam.items():
                for i,c in enumerate(symbols):
                    p=Pchunk[i,t]
                    if c==BLANK:nxt[pref][0]+=(pb+pnb)*p
                    elif pref and c==pref[-1]:
                        nxt[pref][1]+=pnb*p;nxt[pref+c][1]+=pb*p
                    else:nxt[pref+c][1]+=(pb+pnb)*p
            self.beam=dict(sorted(nxt.items(),key=lambda kv:sum(kv[1]),reverse=True)[:self.beam_size])
        return sorted(self.beam.items(),key=lambda kv:sum(kv[1]),reverse=True)

d=StreamingPrefixBeam()
for i in range(0,P.shape[1],2): print("chunk",i//2+1,"best",d.accept(P[:,i:i+2])[0])
'''),
M('''如果每个 chunk 都重新从空前缀开始，重复字符、LM history 和候选路径全部丢失。WFST decoder 同样要保存 active states/tokens 及其代价。'''),
M('''## 2. Lattice 不只是 N-best 列表

Lattice 是许多共享前后缀候选的紧凑图，可以用于：

- best path；
- N-best；
- 置信度；
- 更大语言模型 rescoring；
- 时间对齐。
'''),
C(r'''
candidates=[("AB",1.2,0.6),("AAB",1.4,0.4),("ABB",1.6,0.7),("BA",2.3,0.5)] # text, acoustic cost, lm cost
for scale in [0,.5,1,2]:
    ranked=sorted((a+scale*l,t) for t,a,l in candidates)
    print("LM scale",scale,"best",ranked[0])
'''),
M('''## 3. Stable prefix 与 PGS

可以比较当前 top-K 假设的最长公共前缀，将共同部分标为 stable，其余作为可修改 partial。阈值越保守，字幕越稳但延迟越大。'''),
C(r'''
def common_prefix(strings):
    if not strings:return ""
    out=[]
    for chars in zip(*strings):
        if len(set(chars))==1:out.append(chars[0])
        else:break
    return "".join(out)
for hyps in [["北京天气","北京天启","北京天"],["今天天气","今天田气","今天"]]: print(hyps,"stable=",common_prefix(hyps))
'''),
M('''## 4. Hotword 在 WFST 中的常见位置

可以动态组合小 grammar、调整指定路径权重，或在 beam search 中加 context score。关键是支持动态更新，同时限制误触发和图膨胀。'''),
M('''## 本课测试

1. chunk 结束后为何不能清空 beam？
2. lattice 与单个 best path 有何区别？
3. stable prefix 越长是否一定越好？
4. LM rescoring 为什么需要保留候选？
5. 动态热词为何不宜每次重编译整个巨大图？'''),
quiz('''1. 需要保存前缀、重复状态和 LM/WFST 状态。2. lattice 紧凑保存多条候选。3. 不一定，过早稳定可能锁死错误。4. 被剪掉只剩一条后就无法翻盘。5. 成本和延迟高，应使用动态组合或上下文图。''')
])

lessons[24] = ("流式CTC系统综合实验与验收", [
M('''# 第 24 课：流式 CTC 系统综合实验与验收

这一课把前端、encoder contract、CTC 解码、PGS、RTF、语言模型和 WFST 思维连接起来。重点是建立可验证的系统接口。'''),
C(SETUP + r'''
import time
from collections import defaultdict
from ipywidgets import interact, IntSlider
'''),
M('''## 1. 系统 contract

```text
accept_audio(samples)
  → 新特征帧
  → encoder logits + 新 cache
  → decoder active states
  → partial/stable/final event
```

每层都必须明确：输入时间单位、shape、有效长度、缓存所有权和 flush 行为。'''),
M('''## 2. 在真实音频时间轴上模拟声学后验

为了只检验流式系统逻辑，下面使用真实 0～9 拼接音频的时长，并构造可控 CTC 后验。它不是声学准确率实验；声学模型训练已在第 14 课单独完成。'''),
C(r'''
import soundfile as sf
y,sr=sf.read(ROOT/"data"/"spoken_digits_0_to_9_16k.wav")
duration=len(y)/sr;hop_ms=20;T=int(np.ceil(duration*1000/hop_ms));labels=[BLANK]+list("0123456789")
P=np.full((len(labels),T),.002);P[0,:]=.95
centers=np.linspace(10,T-10,10).astype(int)
for digit,c in zip("0123456789",centers):
    P[:,max(0,c-1):c+2]=.002;P[labels.index(digit),max(0,c-1):c+2]=.93
P/=P.sum(0,keepdims=True)
print(f"audio={duration:.2f}s frames={T} target=0123456789")
'''),
M('''## 3. 流式 greedy + PGS 风格事件'''),
C(r'''
class GreedyStream:
    def __init__(self): self.prev=0;self.text=""
    def accept(self,chunk):
        old=self.text
        for x in chunk.argmax(0):
            if x!=0 and x!=self.prev:self.text+=labels[x]
            self.prev=x
        return {"pgs":"apd","text":self.text[len(old):],"full":self.text}

def run(chunk_frames):
    d=GreedyStream();events=[];t0=time.perf_counter()
    for s in range(0,T,chunk_frames): events.append(d.accept(P[:,s:s+chunk_frames]))
    elapsed=time.perf_counter()-t0
    return events,elapsed

events,elapsed=run(8)
for i,e in enumerate(events):
    if e["text"]: print(i,e)
print("RTF of decoder-only teaching loop:",elapsed/duration)
'''),
M('''## 4. 交互比较 chunk 大小与更新频率'''),
C(r'''
@interact(chunk_frames=IntSlider(min=1,max=40,value=8,description="chunk frames"))
def inspect(chunk_frames=8):
    events,_=run(chunk_frames); changes=[(i+1,e["full"]) for i,e in enumerate(events) if e["text"]]
    print("chunk ms =",chunk_frames*hop_ms,"updates =",len(events),"text changes =",len(changes))
    print(changes)
'''),
M('''## 5. 系统验收清单

### 正确性

- 离线与流式前端帧对齐；
- 不同 chunk 切法结果一致或差异有解释；
- repeated token、空音频、尾块、超长音频测试；
- PGS 重复、乱序和替换测试；
- CTC length audit。

### 性能

- RTF、first partial、first stable、final latency；
- P50/P90/P99；
- CPU/GPU/内存/并发；
- beam、LM scale、chunk、右上下文的准确率—延迟曲线。

### 质量

- CER/WER；
- 热词 recall 与 false trigger；
- 按噪声、说话人、语速、长度分桶分析。'''),
M('''## 6. 最终综合题

1. CTC、LM、WFST、PGS、RTF 各自解决什么问题？
2. 为什么一个 RTF=0.1 的系统仍可能有 1 秒首字延迟？
3. 为什么 CTC head 前面使用全局 Self-Attention 会破坏严格流式？
4. 如何验证 chunk cache 没有重复或遗漏？
5. 若热词召回提高但普通词误识别增加，应怎样评估？
6. 什么时候输出 partial，什么时候 stable，什么时候 final？'''),
quiz('''1. CTC 处理未知对齐；LM 评价序列合理性；WFST 组织约束和加权搜索；PGS 表达增量修改；RTF 衡量计算相对音频时长。2. 可能等待大 chunk、右上下文、网络或稳定策略。3. 当前帧依赖尚未到达的未来。4. 与离线输出逐帧比较，并测试多种不规则 chunk。5. 同时报告热词 recall、false trigger、整体/分桶 CER-WER 和延迟。6. partial 可修订；stable 达到稳定条件；final 在 endpoint/结束且解码完成后提交。'''),
M('''## 学完后的下一阶段

你已经具备阅读和实现流式 CTC 解码器的概念地图。下一阶段不再增加零散名词，而是选择一个真实框架（如 WeNet/Kaldi 风格组件）做源码对应与完整工程实现。''')
])


def write_notebook(number, slug, cells):
    nb=nbf.v4.new_notebook()
    nb.metadata={"kernelspec":{"display_name":"Python (learn-asr)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}}
    nb.cells=[nbf.v4.new_markdown_cell(x) if kind=="markdown" else nbf.v4.new_code_cell(x) for kind,x in cells]
    path=NB_DIR/f"{number:02d}_{slug}.ipynb"
    nbf.write(nb,path)
    print(path)


NB_DIR.mkdir(exist_ok=True)
for number,(slug,cells) in lessons.items(): write_notebook(number,slug,cells)
