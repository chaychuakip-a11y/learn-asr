from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1];NB_DIR=ROOT/"notebooks"
def M(s):return("markdown",textwrap.dedent(s).strip())
def C(s):return("code",textwrap.dedent(s).strip())
SETUP=r'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def find_root():
    here=Path.cwd().resolve()
    for p in [here,*here.parents]:
        if (p/"pyproject.toml").exists():return p
    raise FileNotFoundError("请从 learn_asr 或 notebooks 目录启动 Jupyter")
ROOT=find_root();plt.rcParams["figure.figsize"]=(11,4)
print("项目根目录:",ROOT)
'''
def quiz(x):return M(f'''<details><summary>展开参考答案</summary>

{x}

</details>''')
L={}

L[31]=("音频输入质量_通道幅度DC与重采样",[
M('''# 第 31 课：音频输入质量——通道、幅度、DC、削波与重采样

模型前端的第一步不是 Mel，而是确认收到的 PCM 究竟是什么。格式错误会让后面所有算法失效。'''),
C(SETUP+r'''
from IPython.display import Audio,display
p=ROOT/"data"/"spoken_digits_parts"/"5_jackson_0.wav";y,sr=sf.read(p);y=y.astype(np.float32)
display(Audio(y,rate=sr));print("sr",sr,"shape",y.shape,"dtype",y.dtype,"peak",np.max(np.abs(y)))
'''),
M('''## 1. 四个幅度指标

- Peak：是否接近削波；
- RMS：平均能量；
- dBFS：以数字系统满刻度 1.0 为参考；
- Crest factor：peak/RMS，描述瞬态峰值余量。'''),
C(r'''
def level_stats(x):
    peak=np.max(np.abs(x));rms=np.sqrt(np.mean(x*x)+1e-20)
    return {"peak":peak,"rms":rms,"peak_dBFS":20*np.log10(peak+1e-20),"rms_dBFS":20*np.log10(rms),"crest_dB":20*np.log10((peak+1e-20)/rms)}
print(level_stats(y))
'''),
M('''## 2. DC offset 与 clipping'''),
C(r'''
y_dc=y+.18;y_clip=np.clip(3*y,-1,1)
fig,ax=plt.subplots(3,1,figsize=(11,6),sharex=True)
for a,z,title in zip(ax,[y,y_dc,y_clip],["original","DC offset +0.18","clipped after ×3"]):a.plot(z[:1000]);a.set_title(title)
plt.tight_layout();plt.show()
print("means",np.mean(y),np.mean(y_dc),"clipped sample ratio",np.mean(np.abs(y_clip)>=.999))
'''),
M('''DC 会污染低频和能量 VAD；削波是不可逆失真，之后调小音量不能恢复被削平的波形。'''),
M('''## 3. 采样率与通道契约'''),
C(r'''
y16=librosa.resample(y,orig_sr=sr,target_sr=16000)
print("8k samples",len(y),"16k samples",len(y16),"duration diff",len(y)/sr-len(y16)/16000)
stereo=np.stack([y,.5*y],axis=1);mono_mean=stereo.mean(1);mono_left=stereo[:,0]
print("stereo",stereo.shape,"mean RMS",level_stats(mono_mean)["rms"],"left RMS",level_stats(mono_left)["rms"])
'''),
M('''生产入口必须记录：codec、sample rate、sample format、endianness、channel layout。把 16-bit PCM 字节误读为 float 或大小端错误，波形仍有数字但没有意义。'''),
M('''## 本课测试

1. 0 dBFS 表示声压级 0 dB SPL 吗？
2. 削波后缩小幅度能恢复吗？
3. 立体声直接平均一定安全吗？
4. 重采样为什么不能只隔点删除？
5. DC offset 会影响哪些前端模块？'''),
quiz('''1. 不是，dBFS 参考数字满刻度。2. 不能。3. 不一定，反相通道可能抵消。4. 需要抗混叠滤波。5. RMS/能量、低频谱、VAD、AGC 等。''')])

L[32]=("降噪_SNR_AGC与数据增强",[
M('''# 第 32 课：降噪、SNR、AGC 与数据增强

前端增强的目标不是“声音更好听”，而是在不过度破坏语音线索的前提下提高识别鲁棒性。'''),
C(SETUP+r'''
from IPython.display import Audio,display
y,sr=sf.read(ROOT/"data"/"spoken_digits_parts"/"6_jackson_0.wav");y=y.astype(np.float32);rng=np.random.default_rng(2)
'''),
M(r'''## 1. 按目标 SNR 加噪

$$SNR_{dB}=10\log_{10}\frac{P_s}{P_n}$$'''),
C(r'''
def add_noise_at_snr(clean,snr_db):
    noise=rng.normal(size=len(clean)).astype(np.float32);ps=np.mean(clean**2);pn=np.mean(noise**2)
    noise*=np.sqrt(ps/(pn*10**(snr_db/10)));return clean+noise,noise
noisy,noise=add_noise_at_snr(y,5)
print("measured SNR",10*np.log10(np.mean(y*y)/np.mean(noise*noise)))
display(Audio(noisy,rate=sr))
'''),
M('''## 2. 教学版频谱门控'''),
C(r'''
def spectral_gate(x,sr):
    D=librosa.stft(x,n_fft=256,hop_length=64,win_length=200,center=False);mag=np.abs(D);phase=np.exp(1j*np.angle(D))
    noise_floor=np.percentile(mag,20,axis=1,keepdims=True);gain=np.clip(1-1.5*noise_floor/(mag+1e-8),.08,1)
    return librosa.istft(mag*gain*phase,hop_length=64,win_length=200,center=False,length=len(x))
enh=spectral_gate(noisy,sr)
fig,ax=plt.subplots(3,1,figsize=(11,6),sharex=True)
for a,z,t in zip(ax,[y,noisy,enh],["clean","5 dB noisy","spectral gate"]):a.plot(np.arange(len(z))/sr,z);a.set_title(t)
plt.tight_layout();plt.show();display(Audio(enh,rate=sr))
'''),
M('''简单 spectral gate 会产生 musical noise，并可能删除辅音等低能量语音。真实 NS 需要在目标噪声、说话人和设备上用 ASR 指标验证。'''),
M('''## 3. AGC：目标电平与限幅'''),
C(r'''
def simple_agc(x,target_dbfs=-20,max_gain_db=18):
    rms=np.sqrt(np.mean(x*x)+1e-12);gain=10**((target_dbfs-20*np.log10(rms))/20);gain=min(gain,10**(max_gain_db/20))
    return np.tanh(x*gain),gain
quiet=.08*y;agc,gain=simple_agc(quiet)
print("gain",gain,"before/after RMS dBFS",20*np.log10(np.sqrt(np.mean(quiet**2))),20*np.log10(np.sqrt(np.mean(agc**2))))
'''),
M('''AGC 也会把静音背景噪声放大，因此通常要与 VAD、噪声估计、attack/release 平滑和 limiter 配合。训练数据增强应覆盖增益、噪声、混响、codec，而不是只加白噪声。'''),
M('''## 本课测试

1. SNR 从 20 dB 降到 5 dB意味着什么？
2. 降噪后听感更好是否保证 WER 更低？
3. AGC 为什么需要最大增益？
4. 数据增强与线上降噪能否互相完全替代？
5. 为什么要保留干净原始数据？'''),
quiz('''1. 噪声相对语音强得多。2. 不保证。3. 防止静音/噪声被无限放大。4. 不能，训练鲁棒性和在线处理作用不同。5. 便于重做处理、对照和审计。''')])

L[33]=("VAD与Endpoint_起点终点和Hangover",[
M('''# 第 33 课：VAD 与 Endpoint——何时开始、何时结束

VAD 判断帧是否含语音；endpoint 把一串 VAD 决策变成“开始识别/结束一句”。二者不是同一个模块。'''),
C(SETUP+r'''
from ipywidgets import interact,FloatSlider,IntSlider
y,sr=sf.read(ROOT/"data"/"spoken_digits_0_to_9_8k.wav");y=y.astype(np.float32);L=int(.025*sr);H=int(.01*sr)
frames=librosa.util.frame(y,frame_length=L,hop_length=H).T;energy=10*np.log10(np.mean(frames**2,axis=1)+1e-12);times=np.arange(len(energy))*H/sr
'''),
M('''## 1. 能量 VAD 只是最小基线'''),
C(r'''
@interact(threshold=FloatSlider(min=-60,max=-15,value=-38,step=1),hangover=IntSlider(min=0,max=30,value=8))
def show_vad(threshold=-38,hangover=8):
    raw=energy>threshold;smooth=raw.copy();left=0
    for i,v in enumerate(raw):
        if v:left=hangover
        elif left>0:smooth[i]=True;left-=1
    fig,ax=plt.subplots(2,1,figsize=(11,5),sharex=True)
    ax[0].plot(np.arange(len(y))/sr,y);ax[0].set_ylabel("Amplitude")
    ax[1].plot(times,energy,label="frame energy");ax[1].axhline(threshold,color="C1");ax[1].fill_between(times,energy.min(),energy.max(),where=smooth,alpha=.2)
    ax[1].set(xlabel="Time (s)",ylabel="dB",title=f"speech frames={smooth.sum()}");plt.show()
'''),
M('''## 2. 状态机比逐帧阈值更重要

```text
IDLE --连续若干语音帧--> IN_SPEECH
IN_SPEECH --短静音--> 仍保持
IN_SPEECH --足够长静音--> END
```

起点需要触发帧数，终点需要 silence/hangover；否则键盘声会误触发，词内停顿会过早截断。'''),
C(r'''
def endpoints(mask,start_trigger=3,end_silence=20):
    state="IDLE";speech_run=silence_run=0;segments=[];start=None
    for i,v in enumerate(mask):
        if state=="IDLE":
            speech_run=speech_run+1 if v else 0
            if speech_run>=start_trigger:start=i-start_trigger+1;state="SPEECH";silence_run=0
        else:
            silence_run=0 if v else silence_run+1
            if silence_run>=end_silence:segments.append((start,i-end_silence+1));state="IDLE";speech_run=0
    if state=="SPEECH":segments.append((start,len(mask)))
    return segments
print(endpoints(energy>-38))
'''),
M('''## 3. Endpoint 与最终延迟

终止静音设为 800 ms，就算模型 RTF=0.05，用户通常也至少要等这段静音才得到 final。可以用标点、CTC blank、语义完整性辅助，但误切与延迟仍要权衡。'''),
M('''## 本课测试

1. VAD=false 是否一定代表绝对静音？
2. hangover 的作用是什么？
3. endpoint 为什么直接影响 final latency？
4. 阈值太低会怎样？
5. VAD 是否应该直接删除所有非语音帧再送 CTC？'''),
quiz('''1. 不是，只是判为非语音。2. 防止短暂停顿切断语句。3. 系统等待足够终止静音。4. 噪声误触发、句子难结束。5. 不一定，粗暴删除会破坏时间轴和上下文。''')])

L[34]=("AEC回声消除_NLMS与DoubleTalk",[
M('''# 第 34 课：AEC 回声消除——参考信号、NLMS 与 Double-talk

扬声器播放的远端声音会经房间和设备耦合进入麦克风。AEC 使用“扬声器参考信号”估计这条回声路径。'''),
C(SETUP+r'''
from scipy.signal import lfilter
rng=np.random.default_rng(3);sr=8000;t=np.arange(sr*2)/sr
far=(.5*np.sin(2*np.pi*260*t)+.25*np.sin(2*np.pi*420*t)).astype(np.float32)
echo_path=np.array([0,0,0,.7,.35,.18,.08],np.float32);echo=lfilter(echo_path,[1],far)
near=np.zeros_like(far);speech,_=sf.read(ROOT/"data"/"spoken_digits_parts"/"8_jackson_0.wav");speech=speech.astype(np.float32)
near[5000:5000+min(len(speech),len(near)-5000)]=speech[:len(near)-5000];mic=echo+near+.01*rng.normal(size=len(far))
'''),
M('''## 1. NLMS 自适应滤波器'''),
C(r'''
def nlms(reference,mic,taps=32,mu=.5,eps=1e-6):
    w=np.zeros(taps);out=np.zeros_like(mic);estimated=np.zeros_like(mic)
    padded=np.pad(reference,(taps-1,0))
    for n in range(len(mic)):
        x=padded[n:n+taps][::-1];yhat=np.dot(w,x);e=mic[n]-yhat
        w+=mu*e*x/(np.dot(x,x)+eps);estimated[n]=yhat;out[n]=e
    return out,estimated,w
cleaned,estimated,w=nlms(far,mic)
fig,ax=plt.subplots(3,1,figsize=(11,6),sharex=True)
for a,z,title in zip(ax,[mic,estimated,cleaned],["microphone","estimated echo","AEC output"]):a.plot(t,z);a.set_title(title)
plt.tight_layout();plt.show()
'''),
M('''## 2. ERLE 衡量回声衰减

只应在 near-end 不说话的区段评估 echo return loss enhancement。'''),
C(r'''
region=slice(1000,4500);erle=10*np.log10(np.mean(mic[region]**2)/np.mean(cleaned[region]**2));print("ERLE dB",erle)
'''),
M('''## 3. Double-talk 是难点

近端用户和远端扬声器同时说话时，NLMS 会把近端语音误当成回声误差并污染滤波器。真实 AEC 需要 double-talk detection、nonlinear processing、延迟估计和时钟漂移处理。'''),
M('''## 本课测试

1. AEC 为什么需要 far-end reference？
2. 固定降噪能否替代 AEC？
3. NLMS 的 filter taps 表示什么？
4. ERLE 应在哪类区段测？
5. double-talk 时为什么要降低/冻结更新？'''),
quiz('''1. 用于预测扬声器回声。2. 不能，回声与参考高度相关且随路径变化。3. 所建模回声路径长度。4. 无近端语音的回声区段。5. 防止把近端语音学进回声滤波器。''')])

L[35]=("多麦克风与DelayAndSum波束形成",[
M('''# 第 35 课：多麦克风与 Delay-and-Sum 波束形成

多个麦克风接收同一声源时存在到达时间差。先对齐目标方向再求和，目标相长、部分噪声相消。'''),
C(SETUP+r'''
rng=np.random.default_rng(9);speech,sr=sf.read(ROOT/"data"/"spoken_digits_parts"/"2_jackson_0.wav");speech=speech.astype(np.float32)
def shift(x,n):
    if n>=0:return np.pad(x,(n,0))[:len(x)]
    return np.pad(x[-n:],(0,-n))
mic1=speech+.12*rng.normal(size=len(speech));mic2=shift(speech,3)+.12*rng.normal(size=len(speech))
'''),
M('''## 1. 扫描延迟估计相关性'''),
C(r'''
lags=range(-12,13);scores=[]
for lag in lags:scores.append(np.dot(mic1,shift(mic2,lag)))
best=list(lags)[int(np.argmax(scores))]
plt.stem(list(lags),scores);plt.axvline(best,color="C1");plt.xlabel("Applied delay (samples)");plt.ylabel("Cross-correlation");plt.title("Delay search");plt.show()
print("best alignment delay",best)
'''),
M('''## 2. 对齐后求平均'''),
C(r'''
beam=(mic1+shift(mic2,best))/2
def snr(ref,test):return 10*np.log10(np.mean(ref**2)/(np.mean((test-ref)**2)+1e-12))
print("mic1 SNR",snr(speech,mic1),"beam approximate SNR",snr(speech,beam))
'''),
M('''## 3. 真实阵列更复杂

延迟由麦克风几何、声速和方向决定；分数采样延迟需要插值或频域相位旋转。混响、多声源、空间混叠会限制简单 delay-and-sum。MVDR 等方法还会估计空间协方差。'''),
M('''## 本课测试

1. 未对齐直接平均可能发生什么？
2. 麦克风间距越大是否永远越好？
3. 3.5 samples 延迟怎样处理？
4. beamforming 是否等于声源分离？
5. 线上阵列为什么需要校准？'''),
quiz('''1. 目标相消或频谱梳状失真。2. 不是，会产生空间混叠并受设备约束。3. 插值或频域相位。4. 不等于，但可增强特定方向。5. 通道增益、相位、位置和时钟误差会破坏对齐。''')])

L[36]=("流式音频前端总管线与状态",[
M('''# 第 36 课：流式音频前端总管线——顺序、状态与时间轴

把前五课组合起来，并为每个模块明确状态、延迟和失败模式。'''),
C(SETUP+r'''
y,sr=sf.read(ROOT/"data"/"spoken_digits_parts"/"4_jackson_0.wav");y=y.astype(np.float32)
'''),
M('''## 1. 推荐概念顺序

```text
decode PCM → channel/AEC/beamforming → DC/AGC/NS
→ streaming framing → VAD/endpoint → Log-Mel/CMVN → encoder
```

实际顺序会因设备结构而变，但 AEC 必须尽早拿到同步参考，重采样和通道对齐也不能随意放置。'''),
C(r'''
class FrontendState:
    def __init__(self,frame=200,hop=80):self.frame=frame;self.hop=hop;self.buffer=np.empty(0,np.float32);self.samples_seen=0
    def accept(self,chunk):
        chunk=np.asarray(chunk,np.float32);chunk=chunk-np.mean(chunk) # 教学版逐块去 DC
        self.buffer=np.concatenate([self.buffer,chunk]);out=[];starts=[]
        while len(self.buffer)>=self.frame:
            frame=self.buffer[:self.frame].copy();out.append(frame);starts.append(self.samples_seen)
            self.buffer=self.buffer[self.hop:];self.samples_seen+=self.hop
        return np.asarray(out),np.asarray(starts)
f=FrontendState();all_frames=[];all_starts=[]
for s in range(0,len(y),137):
    frames,starts=f.accept(y[s:s+137]);all_frames.extend(frames);all_starts.extend(starts)
print("frames",len(all_frames),"leftover samples",len(f.buffer),"last timestamp s",all_starts[-1]/sr)
'''),
M('''## 2. 每个模块都要登记

- 输入/输出采样率与 shape；
- 每连接状态；
- 算法 lookahead；
- 时间戳如何映射；
- flush/reset 行为；
- 可观测指标；
- 旁路策略。

否则线上只看到“ASR 变差”，无法判断是 PCM、AEC、VAD、特征还是模型。'''),
M('''## 3. 前端质量指标

除了听感和 SNR，还应记录 clipping ratio、DC、RMS/dBFS、VAD 占比、endpoint 时长、丢包、重采样漂移，以及最终 CER/WER。'''),
M('''## 本课测试

1. 为什么时间戳必须从输入采样计数推导？
2. 模块 reset 遗漏会造成什么？
3. 前端降噪是否应该只用 SNR 验收？
4. VAD 应在 AEC 前还是后？
5. 旁路开关有什么价值？'''),
quiz('''1. chunk 到达时间不等于音频内容时间。2. 上一会话状态污染下一会话。3. 不，应同时看 ASR 与分桶指标。4. 通常 AEC/增强后更可靠，但架构需结合参考同步设计。5. 快速定位模块影响并在故障时降级。''')])

L[37]=("时间戳_说话人分段与Diarization",[
M('''# 第 37 课：时间戳、说话人分段与 Diarization

ASR 回答“说了什么”，diarization 回答“谁在什么时候说”。教学实验用 FSDD 多说话人开源录音建立简单声纹原型。'''),
C(SETUP+r'''
from collections import defaultdict
files=sorted((ROOT/"data"/"fsdd_multispeaker").glob("*.wav"));print(len(files),files[:3])
'''),
M('''## 1. 教学版 speaker embedding'''),
C(r'''
def embedding(path):
    y,sr=sf.read(path);m=librosa.feature.mfcc(y=y.astype(np.float32),sr=sr,n_mfcc=13,n_fft=256,hop_length=80)
    e=np.concatenate([m.mean(1),m.std(1)]);return e/(np.linalg.norm(e)+1e-8)
groups=defaultdict(list)
for p in files:groups[p.stem.split("_")[1]].append((p,embedding(p)))
prototypes={s:np.mean([e for _,e in items],axis=0) for s,items in groups.items()}
for s in prototypes:prototypes[s]/=np.linalg.norm(prototypes[s])+1e-8
for s,items in groups.items():
    p,e=items[0];scores={name:float(e@proto) for name,proto in prototypes.items()};print(p.name,"->",max(scores,key=scores.get),scores)
'''),
M('''这不是生产声纹模型：数据太少，而且 enrollment 与测试复用。它只展示 embedding→相似度→聚类/匹配的接口。'''),
M('''## 2. 时间戳的来源

- 帧时间：由采样计数、window/hop 得到；
- CTC spike：可给 token 粗时间；
- forced alignment：在已知文本条件下寻找更精确对齐；
- segment 时间：由 VAD/diarization 边界得到。

经过卷积下采样、右上下文和重采样后，必须维护从 encoder step 回到原始样本的映射。'''),
M('''## 3. Overlap speech

普通 diarization 常假设一帧一个 speaker；两人重叠时这个假设失效，需要 overlap detection 或多输出源分离。DER 也通常拆成 miss、false alarm、speaker confusion。'''),
M('''## 本课测试

1. speaker recognition 与 diarization 有何不同？
2. CTC spike 时间是否等于真实音素边界？
3. 为什么不能用同一录音同时 enrollment 和测试并宣称准确？
4. 两人同时说话为何困难？
5. DER 包含哪些主要错误？'''),
quiz('''1. 前者识别身份，后者把时间轴按说话人切分/聚类。2. 不等于，只是模型输出峰。3. 会数据泄漏。4. 单标签帧假设失效。5. 漏检、误检和说话人混淆。''')])

L[38]=("标点恢复_ITN与文本规范化",[
M('''# 第 38 课：标点恢复、ITN 与文本规范化

ASR token 往往是“spoken form”。用户需要“written form”：数字、日期、金额、单位、大小写和标点。'''),
C(SETUP+r'''
import re
'''),
M('''## 1. ITN 不是普通字符串替换

“一二三”可能是号码 123，也可能是逐字读数；“一百零二”是 102。上下文、locale 和业务领域会改变规则。'''),
C(r'''
digit_map=dict(zip("零一二三四五六七八九","0123456789"))
def digit_sequence_itn(text):
    return re.sub(r"[零一二三四五六七八九]{2,}",lambda m:"".join(digit_map[c] for c in m.group()),text)
for s in ["电话一三八零零一二三四五六","编号一二三","一百零二元"]:print(s,"->",digit_sequence_itn(s))
'''),
M('''最后一个例子故意展示失败：简单逐字映射不理解“百”。成熟 ITN 常用分类器 + 规则/FST，把不同 semiotic class（数字、日期、货币等）分别处理。'''),
M('''## 2. 标点恢复会影响语义'''),
C(r'''
examples=[("如果下雨就不去",["如果下雨，就不去。","如果下雨就不去。"]),("他说你不行",["他说：你不行。","他说你不行。"])]
for raw,cands in examples:print("raw",raw,"candidates",cands)
'''),
M('''流式标点通常会修订最近若干词，因此也需要 partial/stable 机制。不能让标点模块无限回改已经提交的业务文本。'''),
M('''## 3. Normalization 与 ITN 方向相反

- Text normalization：`102元` → “一百零二元”，常用于 TTS/训练文本。
- Inverse text normalization：口语识别结果 → `102元`。

训练标签规范必须和 tokenizer、解码评估、线上展示一致。'''),
M('''## 本课测试

1. “一二三”和“一百二十三”能否用同一逐字规则？
2. 标点是否可能改变意图？
3. 流式标点为什么需要可修订区域？
4. WER 评估前为什么要统一 normalization？
5. ITN 出错时应优先保留原文还是编造规范形式？'''),
quiz('''1. 不能。2. 可以。3. 后续词会改变句法判断。4. 否则格式差异被误算成识别错误。5. 应可追溯地保留 spoken form，避免生成错误事实。''')])

L[39]=("置信度_NBest与语义重排序",[
M('''# 第 39 课：置信度、N-best 与语义重排序

系统不仅要给答案，还要知道何时不确定，并保留足够候选让语义模块纠错。'''),
C(SETUP+r'''
from scipy.special import softmax
from ipywidgets import interact,FloatSlider
'''),
M('''## 1. 最大 softmax 不是天然校准的置信度'''),
C(r'''
logits=np.array([[4,1,0],[2.2,2.0,0],[10,8,0]],float)
for z in logits:
    p=softmax(z);entropy=-np.sum(p*np.log(p+1e-12));print("p",p,"max",p.max(),"entropy",entropy)
'''),
M('''模型可能过度自信。需要在独立数据上做 reliability diagram、ECE，并可使用 temperature scaling；句级置信度还要处理长度、blank 和 beam/lattice posterior。'''),
M('''## 2. N-best 语义重排序'''),
C(r'''
cands=[{"text":"打开空调","asr":-2.2,"intent":.92},{"text":"打开空道","asr":-1.9,"intent":.08},{"text":"打卡空调","asr":-2.0,"intent":.12}]
@interact(weight=FloatSlider(min=0,max=3,value=1,step=.1,description="semantic weight"))
def rerank(weight=1):
    for c in cands:c["score"]=c["asr"]+weight*np.log(c["intent"]+1e-6)
    for c in sorted(cands,key=lambda x:x["score"],reverse=True):print(c)
'''),
M('''语义重排只能在声学/搜索仍保留正确候选时起作用。若正确文本已被 beam 剪掉，后端无法凭空恢复而不承担幻觉风险。'''),
M('''## 3. Reject/clarify 是合法输出

低置信度时可以请求复述、展示候选、转人工或只执行可撤销操作。高风险命令不能因为“语义看起来合理”就忽略声学不确定性。'''),
M('''## 本课测试

1. max softmax=0.99 是否保证 99% 正确？
2. N-best 为什么比 1-best 更适合后处理？
3. semantic weight 太大会怎样？
4. 正确候选被 beam 删除后还能可靠重排回来吗？
5. 低置信度时系统必须强行给一个答案吗？'''),
quiz('''1. 不保证，需要校准。2. 保留替代假设。3. 可能无视声音选择语义常见句。4. 不能。5. 不必，可以拒识或澄清。''')])

L[40]=("NLU意图识别_槽位抽取与对话状态",[
M('''# 第 40 课：NLU——意图识别、槽位抽取与对话状态

ASR 输出文本；NLU 把文本转换成业务可执行的结构，例如 `{intent:set_temperature, value:26, unit:C}`。'''),
C(SETUP+r'''
import torch
import torch.nn as nn
torch.manual_seed(2)
samples=[("打开空调","open_ac"),("开启空调","open_ac"),("把空调打开","open_ac"),("关闭空调","close_ac"),("关掉空调","close_ac"),("把空调关闭","close_ac"),("温度调到二十六度","set_temp"),("设置温度二十四度","set_temp"),("调到二十二度","set_temp")]
labels=sorted(set(y for _,y in samples));chars=sorted(set("".join(x for x,_ in samples)));ci={c:i for i,c in enumerate(chars)}
def bow(text):
    x=torch.zeros(len(chars))
    for c in text:
        if c in ci:x[ci[c]]+=1
    return x
X=torch.stack([bow(x) for x,_ in samples]);Y=torch.tensor([labels.index(y) for _,y in samples])
model=nn.Linear(len(chars),len(labels));opt=torch.optim.Adam(model.parameters(),lr=.08)
for _ in range(300):opt.zero_grad();loss=nn.functional.cross_entropy(model(X),Y);loss.backward();opt.step()
for text in ["请打开空调","空调关掉","温度调到二十三度"]:print(text,labels[model(bow(text)).argmax().item()])
'''),
M('''这个 bag-of-characters 分类器是教学基线：它没有词序、上下文和 OOV 泛化能力。生产 NLU 会使用预训练 encoder、规则、检索或 LLM，并对领域数据评估。'''),
M('''## 2. 槽位抽取'''),
C(r'''
cn={"二十":20,"二十一":21,"二十二":22,"二十三":23,"二十四":24,"二十五":25,"二十六":26,"二十七":27,"二十八":28}
def parse_command(text):
    intent=labels[model(bow(text)).argmax().item()];slots={}
    for spoken,value in cn.items():
        if spoken in text:slots["temperature"]=value
    return {"intent":intent,"slots":slots}
for s in ["温度调到二十三度","请打开空调"]:print(parse_command(s))
'''),
M('''## 3. 对话状态用于补全省略

用户：“把温度调到二十六度。” 下一句：“还是二十四吧。” 第二句需要上一轮 intent/device 才能补全。状态必须有过期、用户隔离和可审计更新。'''),
M('''## 4. 执行前验证

Schema 校验、权限、范围、设备存在性和确认策略属于业务层。NLU 高置信不代表允许执行危险操作。'''),
M('''## 本课测试

1. intent 与 slot 有何区别？
2. ASR 置信度低但 NLU 置信度高时应信谁？
3. 对话状态为什么必须按用户/会话隔离？
4. 槽位值为什么需要 schema 验证？
5. “删除全部记录”是否应直接执行？'''),
quiz('''1. intent 表示动作类别，slot 是动作参数。2. 综合判断，不能让语义自信掩盖声学不确定。3. 防止上下文串话。4. 防止越界、类型错误和注入。5. 不应，应鉴权并二次确认。''')])

L[41]=("LLM语义后处理与端到端语音系统",[
M('''# 第 41 课：受约束的 LLM 后处理与端到端语音系统

LLM 可以利用长上下文修正标点、抽取结构和总结，但最危险的问题是把“不确定”改成“听起来合理但原音频没说过”。'''),
C(SETUP),
M('''## 1. 给 LLM 的输入不应只有 1-best

推荐同时提供：

- stable transcript 与可修改 partial；
- N-best/lattice 关键候选及 acoustic/LM score；
- token/word confidence 和时间戳；
- 允许的词典、实体和 JSON schema；
- 明确规则：不得添加候选中没有证据的数字、人名和否定词。'''),
M('''## 2. 分离 transcript 与 interpretation

```json
{
  "verbatim_transcript": "把会议改到周四三点",
  "normalized_transcript": "把会议改到周四15:00",
  "intent": "reschedule_meeting",
  "slots": {"day": "周四", "time": "15:00"},
  "needs_confirmation": true
}
```

原始转写、规范化文本和语义解释必须分别保存，不能用语义结果覆盖证据。'''),
C(r'''
allowed_intents={"open_ac","close_ac","set_temp","reschedule_meeting","unknown"}
def validate_semantic(result):
    errors=[]
    if result.get("intent") not in allowed_intents:errors.append("unknown intent")
    if "temperature" in result.get("slots",{}) and not 16<=result["slots"]["temperature"]<=30:errors.append("temperature out of range")
    if not result.get("verbatim_transcript"):errors.append("missing evidence transcript")
    return errors
print(validate_semantic({"verbatim_transcript":"温度调到四十度","intent":"set_temp","slots":{"temperature":40}}))
'''),
M('''## 3. 完整系统数据流

```text
PCM
 ↓ AEC / Beamforming / NS / AGC
VAD + Streaming Log-Mel
 ↓ Causal/Chunk Encoder + CTC
Prefix Beam / LM / WFST / Hotword
 ↓ partial/stable/final + confidence + timestamps
Punctuation / ITN / Diarization
 ↓ N-best semantic reranking
Intent / Slots / Dialogue State / constrained LLM
 ↓ policy validation / confirmation / action
```

每层都应能旁路、版本化、记录指标并独立回归。'''),
M('''## 4. 端到端验收矩阵

- 前端：SNR、ERLE、VAD miss/FA、endpoint latency；
- ASR：CER/WER、热词、RTF、first/final latency；
- 后处理：标点 F1、ITN accuracy、DER、confidence calibration；
- NLU：intent accuracy、slot F1、task success；
- 安全：错误执行率、需确认召回率、幻觉数字/实体率；
- 系统：P99、并发、断线恢复、资源和成本。
'''),
M('''## 最终测试

1. 为什么不能让 LLM 直接覆盖原始 transcript？
2. 哪些词的错误具有特别高风险？
3. semantic module 能否挽救已被 beam 删除的正确候选？
4. 为什么每层要有版本号和旁路？
5. 系统最终指标为什么不只是 WER？'''),
quiz('''1. 会丢失证据并掩盖幻觉。2. 数字、人名、金额、时间、否定词和动作词。3. 不能可靠挽救。4. 为审计、定位、灰度和回滚。5. 用户关心任务成功、延迟、稳定性和错误执行风险。'''),
M('''## 课程完成

现在路线已经从麦克风前端一直延伸到语义执行与生产部署。下一步应选择一个具体目标场景（命令词、会议转写或实时字幕），用真实数据完成端到端项目，而不是继续堆叠名词。''')])

for n,(slug,cells) in L.items():
    nb=nbf.v4.new_notebook();nb.metadata={"kernelspec":{"display_name":"Python (learn-asr)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.13"}}
    nb.cells=[nbf.v4.new_markdown_cell(x) if k=="markdown" else nbf.v4.new_code_cell(x) for k,x in cells]
    path=NB_DIR/f"{n:02d}_{slug}.ipynb";nbf.write(nb,path);print(path)
