from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import nbformat
import numpy as np
import soundfile as sf

from notebook_layout import executed_path


ROOT=Path(__file__).resolve().parents[1]
NB_DIR=ROOT/"notebooks"
MIN_LAST_LESSON=46
MIN_LM_LAST_LESSON=9
UPGRADED_LESSONS=set(range(1,42))
FOUNDATION_STEMS=[
    "基础_01_Python最小语法与函数",
    "基础_02_Tensor创建索引与Shape",
    "基础_03_Tensor运算广播与维度变换",
    "基础_04_Autograd损失与优化器",
    "基础_05_nnModule与训练验证循环",
    "基础_06_Dataset_DataLoader与变长语音Batch",
]
AUDIO_FOUNDATION_STEMS=[
    "音频基础_01_振动波形与时间轴",
    "音频基础_02_周期频率相位与正弦波",
    "音频基础_03_振幅RMS功率与dB",
    "音频基础_04_采样量化PCM位深与通道",
    "音频基础_05_叠加谐波噪声与SNR",
    "音频基础_06_真实WAV读取试听与输入审计",
]
RESEARCH_THINKING_STEMS=[
    "思维训练_01_从现象到可证伪假设",
    "思维训练_02_错误分类切片与辛普森悖论",
    "思维训练_03_对照实验消融混杂与交互",
    "思维训练_04_不确定性Bootstrap与配对比较",
    "思维训练_05_因果排错不变量与反事实",
    "思维训练_06_Pareto风险与证据化决策",
]
COGNITIVE_TOOLKIT_STEMS=[
    "认知拓展_01_论证反例与表达",
    "认知拓展_02_基础率贝叶斯与自然频数",
    "认知拓展_03_因果图混杂中介与碰撞点",
    "认知拓展_04_存量流量反馈延迟与杠杆",
    "认知拓展_05_期望值后悔值与信息价值",
    "认知拓展_06_熵编码互信息与表示",
    "认知拓展_07_认知偏差预测校准与复盘",
    "认知拓展_08_提取间隔反馈与迁移",
]
INFORMATION_REASONING_STEMS=[
    "研究进阶_01_问题定义问题树与费米估算",
    "研究进阶_02_必要充分量词作用域与反例",
    "研究进阶_03_演绎归纳溯因与竞争解释",
    "研究进阶_04_论证图钢人化与反方测试",
    "研究进阶_05_分面检索布尔查询与停止规则",
    "研究进阶_06_横向阅读来源溯源与独立性",
    "研究进阶_07_证据矩阵冲突消解与综合",
    "研究进阶_08_系统综述开放科学与研究档案",
]
CAPSTONE_STEM="结课项目_实时数字CTC声学引擎_从WAV到流式文本"
BEGINNER_CODE_STEM="代码伴读_零基础逐行理解ASR"
SPECIAL_NOTEBOOKS=[
    "学习中枢_诊断与掌握度仪表盘.ipynb",
    "专题_CTC可视化实验室_从路径到流式解码.ipynb",
    "专题_流式ASR实验室_Chunk缓存PGS与实时率.ipynb",
    "专题_WFST实验室_从L与G到流式TokenPassing.ipynb",
    "专题_量化部署实验室_ONNX_INT8性能与服务验收.ipynb",
    "专题_音频前端实验室_质量VAD_AEC与波束形成.ipynb",
    "专题_语义后处理实验室_时间戳ITN置信度与安全执行.ipynb",
    "专题_FSDD说话人泛化实验_数据划分增强与盲测.ipynb",
    "专题_FSDD六折LOSO_嵌套选择与说话人统计.ipynb",
    "专题_AudioMNIST外部盲测_冻结协议跨域失败与适配边界.ipynb",
]
REQUIRED=[
    "README.md","LEARNING_PATH.md","pyproject.toml","uv.lock","DATA_SOURCES.md","COURSE_AUDIT.md","LEARNING_LOG.md",
    "AUDIO_SOFTWARE_GUIDE.md","AUDIO_DIAGNOSIS_PRACTICE.md","LEARNING_EXPANSION_RESEARCH.md","COGNITIVE_TOOLKIT.md","LOGIC_INFORMATION_RESEARCH.md",
    "data/audio_software_lab/manifest.json",
    "data/audio_diagnosis_lab/manifest.json","data/audio_diagnosis_lab/answer_key.json",
    "data/audio_diagnosis_lab/reference_clean_speech.wav",
    "data/audio_software_lab/01_calibration_440hz_peak_minus12dbfs.wav",
    "data/audio_software_lab/02_two_tones_440hz_1000hz.wav",
    "data/audio_software_lab/03_dc_offset_220hz.wav",
    "data/audio_software_lab/04_hard_clipped_440hz.wav",
    "data/audio_software_lab/05_noise_profile_then_speech_snr10db.wav",
    "data/audio_software_lab/06_stereo_right_delayed_1ms.wav",
    "data/audio_software_lab/07_real_speech_digit_zero_8khz.wav",
    "FRONTIER_ASR_2026.md","FRONTIER_ASR_LM_READING.md","ASR_LM_ENVIRONMENT.md",
    "ASR_LM_OPENFST_KENLM_CHEATSHEET.md","CAPSTONE_GUIDE.md","FSDD_GENERALIZATION_GUIDE.md",
    "FSDD_LOSO_GUIDE.md","AUDIOMNIST_EXTERNAL_GUIDE.md","AUDIOMNIST_EXTERNAL_PROTOCOL.md",
    "AUDIOMNIST_EXTERNAL_REPORT.md","LICENSE","LICENSE-CONTENT","LICENSE-SCOPE.md","NOTICE",
    "notebooks/README.md","notebooks/核心课程索引_第01到41课.md",
    "notebooks/PyTorch零基础课程索引.md","notebooks/音频零基础课程索引.md",
    "notebooks/语言模型零基础_课程索引.md","notebooks/研究与工程思维课程索引.md","notebooks/认知拓展课程索引.md","notebooks/逻辑与信息研究进阶课程索引.md",
    "notebooks/零基础预备课_Python与PyTorch.ipynb","scripts/notebook_layout.py",
    "scripts/build_beginner_code_companion.py",
    "scripts/build_pytorch_foundation_course.py","scripts/build_frontier_course.py",
    "scripts/build_audio_foundation_course.py","scripts/build_audio_software_lab_assets.py",
    "scripts/build_audio_diagnosis_practice.py","scripts/audio_diagnosis_quiz.py",
    "scripts/enrich_course_bridges.py","scripts/build_research_thinking_course.py","scripts/build_cognitive_toolkit_course.py","scripts/build_information_reasoning_course.py",
    "scripts/build_learning_hub.py","scripts/build_ctc_lab.py","scripts/build_streaming_lab.py",
    "scripts/build_wfst_lab.py","scripts/build_quant_deploy_lab.py","scripts/build_frontend_lab.py",
    "scripts/build_semantic_lab.py","scripts/build_capstone_lab.py","scripts/build_fsdd_generalization_lab.py",
    "scripts/build_fsdd_loso_lab.py","scripts/build_audiomnist_external_lab.py","scripts/execute_course_labs.py",
    "scripts/run_notebook_headless.py",
    "scripts/validate_lm_course.py",
    "acoustic_engine/README.md","acoustic_engine/features.py","acoustic_engine/model.py",
    "acoustic_engine/decoder.py","acoustic_engine/language_model.py","acoustic_engine/engine.py",
    "acoustic_engine/streaming.py","acoustic_engine/api.py","acoustic_engine/benchmark.py",
    "acoustic_engine/tutor.py","acoustic_engine/challenge.py","acoustic_engine/mastery.py",
    "learning_workspace/README.md","learning_workspace/asr_practice.py",
    "fsdd_generalization/data.py","fsdd_generalization/training.py","fsdd_generalization/loso.py",
    "external_evaluation/protocol.py","external_evaluation/data.py","external_evaluation/evaluate.py",
    "external_evaluation/final_fit.py","external_evaluation/audiomnist_protocol.json",
    "artifacts/tiny_digit_ctc.pt","artifacts/streaming_digit_ctc.pt",
    "artifacts/fsdd_speaker_disjoint_results.json","artifacts/fsdd_loso_results.json",
    "artifacts/audiomnist_external_results.json","tests/test_acoustic_engine.py",
    "tests/test_fsdd_generalization.py","tests/test_external_evaluation_protocol.py",
    "tests/test_research_thinking_course.py","tests/test_cognitive_toolkit_course.py","tests/test_information_reasoning_course.py","tests/test_research_workspace.py",
    "research_workspace/README.md","research_workspace/__init__.py","research_workspace/workbench.py","research_workspace/example_dossier.json",
]


def lesson_number(path:Path):
    m=re.match(r"(\d\d)_",path.name)
    return int(m.group(1)) if m else None


def language_model_number(path:Path):
    m=re.match(r"语言模型零基础_(\d\d)_",path.name)
    return int(m.group(1)) if m else None


def validate_notebook(path:Path,require_upgrade:bool):
    errors=[]
    try:
        nb=nbformat.read(path,as_version=4)
        nbformat.validate(nb)
    except Exception as exc:
        return [f"{path.name}: invalid notebook: {exc}"]
    if not nb.cells or nb.cells[0].cell_type!="markdown" or not nb.cells[0].source.lstrip().startswith("#"):
        errors.append(f"{path.name}: first cell must be a Markdown heading")
    if require_upgrade:
        tags=sum("course-upgrade-v2" in c.metadata.get("tags",[]) for c in nb.cells)
        if tags!=4:errors.append(f"{path.name}: expected 4 course-upgrade cells, got {tags}")
        if "course" not in nb.metadata:errors.append(f"{path.name}: missing course metadata")
    for i,c in enumerate(nb.cells):
        for output in c.get("outputs",[]):
            if output.get("output_type")=="error":
                errors.append(f"{path.name}: cell {i} stores error {output.get('ename')}: {output.get('evalue')}")
    return errors


def validate_foundation(path:Path,expected_lesson:int,executed:bool):
    errors=validate_notebook(path,False)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    metadata=nb.metadata.get("foundation_course",{})
    if metadata.get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: foundation lesson metadata mismatch")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["课前诊断","本课练习","离场票与间隔复习"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<5:errors.append(f"{path.name}: expected at least 5 code cells")
    if executed:
        missing=[i for i,cell in enumerate(code_cells) if cell.execution_count is None]
        if missing:errors.append(f"{path.name}: unexecuted code cells {missing}")
    else:
        dirty=[i for i,cell in enumerate(code_cells) if cell.execution_count is not None or cell.outputs]
        if dirty:errors.append(f"{path.name}: source notebook stores outputs in cells {dirty}")
    return errors


def validate_audio_foundation(path:Path,expected_lesson:int,executed:bool):
    errors=validate_notebook(path,False)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    metadata=nb.metadata.get("audio_foundation_course",{})
    if metadata.get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: audio foundation lesson metadata mismatch")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["课前回忆","固定观察框架","分层练习","最小掌握门禁"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<4:errors.append(f"{path.name}: expected at least 4 code cells")
    if executed:
        missing=[i for i,cell in enumerate(code_cells) if cell.execution_count is None]
        if missing:errors.append(f"{path.name}: unexecuted code cells {missing}")
    else:
        dirty=[i for i,cell in enumerate(code_cells) if cell.execution_count is not None or cell.outputs]
        if dirty:errors.append(f"{path.name}: source notebook stores outputs in cells {dirty}")
    return errors


def validate_research_thinking(path:Path,expected_lesson:int,executed:bool):
    errors=validate_source_or_executed(path,executed)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    metadata=nb.metadata.get("course",{})
    if metadata.get("track")!="research_thinking" or metadata.get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: research-thinking metadata mismatch")
    if metadata.get("evidence_model")!="claim-evidence-falsifier-boundary-next-test":
        errors.append(f"{path.name}: missing evidence model")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["课前预测","一手资料与课程取舍","闭卷挑战","最小掌握门禁"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<3:errors.append(f"{path.name}: expected at least 3 code cells")
    return errors


def validate_cognitive_toolkit(path:Path,expected_lesson:int,executed:bool):
    errors=validate_source_or_executed(path,executed)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    metadata=nb.metadata.get("course",{})
    if metadata.get("track")!="cognitive_toolkit" or metadata.get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: cognitive-toolkit metadata mismatch")
    if metadata.get("evidence_model")!="question-model-prediction-evidence-boundary-update":
        errors.append(f"{path.name}: missing cognitive evidence model")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["课前预测","一手资料与课程取舍","误用警报","适用边界","迁移练习","闭卷挑战","最小掌握门禁"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<3:errors.append(f"{path.name}: expected at least 3 code cells")
    return errors


def validate_information_reasoning(path:Path,expected_lesson:int,executed:bool):
    errors=validate_source_or_executed(path,executed)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    metadata=nb.metadata.get("course",{})
    if metadata.get("track")!="information_reasoning" or metadata.get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: information-reasoning metadata mismatch")
    if metadata.get("evidence_model")!="question-query-provenance-matrix-synthesis-update":
        errors.append(f"{path.name}: missing information-reasoning evidence model")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["课前预测","一手资料与课程取舍","信息搜集实作","误用警报","适用边界","迁移练习","闭卷挑战","最小掌握门禁","来源家族","停止规则"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<3:errors.append(f"{path.name}: expected at least 3 code cells")
    return errors


def validate_course_bridge(path:Path):
    nb=nbformat.read(path,as_version=4)
    tagged=[cell for cell in nb.cells if "course-bridge-v3" in cell.metadata.get("tags",[])]
    errors=[]
    if len(tagged)!=1:
        errors.append(f"{path.name}: expected exactly one course-bridge-v3 cell, got {len(tagged)}")
    elif "知识接力" not in tagged[0].source or "本课接口契约" not in tagged[0].source:
        errors.append(f"{path.name}: incomplete course bridge")
    return errors


def validate_audio_software_assets():
    errors=[]
    directory=ROOT/"data"/"audio_software_lab"
    manifest_path=directory/"manifest.json"
    try:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid audio software manifest: {exc}"]
    assets=manifest.get("assets",[])
    if len(assets)!=7:
        errors.append(f"audio software manifest: expected 7 assets, got {len(assets)}")
        return errors
    for item in assets:
        path=directory/item["file"]
        try:
            waveform,sample_rate=sf.read(path,dtype="float64",always_2d=True)
        except Exception as exc:
            errors.append(f"audio software asset {path.name}: cannot read: {exc}")
            continue
        peak=float(np.max(np.abs(waveform)))
        rms=float(np.sqrt(np.mean(waveform**2)))
        dc=float(np.mean(waveform))
        expected_channels=int(item["channels"])
        if sample_rate!=int(item["sample_rate_hz"]):
            errors.append(f"audio software asset {path.name}: sample rate mismatch")
        if waveform.shape[1]!=expected_channels:
            errors.append(f"audio software asset {path.name}: channel mismatch")
        for label,actual in (("peak",peak),("rms",rms),("dc_mean",dc)):
            if not np.isclose(actual,float(item[label]),atol=2e-6):
                errors.append(f"audio software asset {path.name}: {label} mismatch")
    by_name={item["file"]:item for item in assets}
    if not np.isclose(by_name["01_calibration_440hz_peak_minus12dbfs.wav"]["peak_dbfs"],-12.0,atol=1e-6):
        errors.append("audio software calibration tone must have -12 dBFS peak")
    if by_name["06_stereo_right_delayed_1ms.wav"].get("right_channel_delay_samples")!=16:
        errors.append("audio software stereo delay must be 16 samples")
    if not np.isclose(by_name["05_noise_profile_then_speech_snr10db.wav"].get("speech_region_measured_snr_db"),10.0,atol=1e-9):
        errors.append("audio software noisy speech must measure 10 dB SNR")
    return errors


def validate_audio_diagnosis_assets():
    errors=[]
    directory=ROOT/"data"/"audio_diagnosis_lab"
    try:
        public=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        answers=json.loads((directory/"answer_key.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid audio diagnosis manifest: {exc}"]
    public_cases=public.get("cases",[])
    answer_cases=answers.get("cases",[])
    if len(public_cases)!=24 or len(answer_cases)!=24:
        return [f"audio diagnosis lab: expected 24 public and answer cases, got {len(public_cases)} and {len(answer_cases)}"]
    public_ids={item["case_id"] for item in public_cases}
    answer_ids={item["case_id"] for item in answer_cases}
    expected_ids={f"case_{number:02d}" for number in range(1,25)}
    if public_ids!=expected_ids or answer_ids!=expected_ids:
        errors.append("audio diagnosis lab: case IDs must be exactly case_01..case_24")
    if any("issue" in item for item in public_cases):
        errors.append("audio diagnosis public manifest must not reveal issues")
    difficulties={level:sum(item.get("difficulty")==level for item in public_cases) for level in ("beginner","intermediate","advanced")}
    if difficulties!={"beginner":7,"intermediate":9,"advanced":8}:
        errors.append(f"audio diagnosis lab: unexpected difficulty split {difficulties}")
    issues={item.get("issue") for item in answer_cases}
    if len(issues)!=24 or not {"clean","clean_stereo","hard_clipping","wrong_sample_rate_metadata","stereo_polarity_inversion"}.issubset(issues):
        errors.append("audio diagnosis lab: issue coverage or clean controls are incomplete")
    by_issue={item["issue"]:item for item in answer_cases}
    for item in answer_cases:
        path=directory/item["file"]
        if not path.exists():
            errors.append(f"audio diagnosis lab: missing {path.name}")
            continue
        waveform,sample_rate=sf.read(path,dtype="float64",always_2d=True)
        expected=item["measurements"]
        actual_peak=float(np.max(np.abs(waveform)))
        actual_rms=float(np.sqrt(np.mean(waveform**2)))
        actual_dc=float(np.mean(waveform))
        if sample_rate!=int(expected["sample_rate_hz"]) or waveform.shape[1]!=int(expected["channels"]):
            errors.append(f"audio diagnosis lab {path.name}: metadata mismatch")
        for label,actual in (("peak",actual_peak),("rms",actual_rms),("dc_mean",actual_dc)):
            if not np.isclose(actual,float(expected[label]),atol=5e-5):
                errors.append(f"audio diagnosis lab {path.name}: {label} mismatch")
    if by_issue["wrong_sample_rate_metadata"]["measurements"]["sample_rate_hz"]!=8000:
        errors.append("audio diagnosis lab: wrong-sample-rate case must be labelled 8 kHz")
    polarity_path=directory/by_issue["stereo_polarity_inversion"]["file"]
    polarity,_=sf.read(polarity_path,dtype="float64",always_2d=True)
    if polarity.shape[1]!=2 or not np.allclose(polarity[:,0]+polarity[:,1],0.0,atol=1/32768):
        errors.append("audio diagnosis lab: polarity inversion must cancel in mono sum")
    silent_path=directory/by_issue["right_channel_silent"]["file"]
    silent,_=sf.read(silent_path,dtype="float64",always_2d=True)
    if silent.shape[1]!=2 or not np.allclose(silent[:,1],0.0):
        errors.append("audio diagnosis lab: silent-channel case must have a zero right channel")
    return errors


def validate_frontier(path:Path,expected_lesson:int,executed:bool):
    errors=validate_notebook(path,False)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    if nb.metadata.get("course",{}).get("lesson")!=expected_lesson:
        errors.append(f"{path.name}: frontier lesson metadata mismatch")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["完成标准","练习","离场小测"]:
        if marker not in text:errors.append(f"{path.name}: missing {marker}")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<5:errors.append(f"{path.name}: expected at least 5 code cells")
    if executed:
        missing=[i for i,cell in enumerate(code_cells) if cell.execution_count is None]
        if missing:errors.append(f"{path.name}: unexecuted code cells {missing}")
    else:
        dirty=[i for i,cell in enumerate(code_cells) if cell.execution_count is not None or cell.outputs]
        if dirty:errors.append(f"{path.name}: source notebook stores outputs in cells {dirty}")
    return errors


def validate_source_or_executed(path:Path,executed:bool):
    errors=validate_notebook(path,False)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if not code_cells:errors.append(f"{path.name}: has no code cells")
    if executed:
        missing=[i for i,cell in enumerate(code_cells) if cell.execution_count is None]
        if missing:errors.append(f"{path.name}: unexecuted code cells {missing}")
    else:
        dirty=[i for i,cell in enumerate(code_cells) if cell.execution_count is not None or cell.outputs]
        if dirty:errors.append(f"{path.name}: source notebook stores outputs in cells {dirty}")
    return errors


def validate_beginner_code_companion(path:Path,executed:bool):
    errors=validate_source_or_executed(path,executed)
    if errors:return errors
    nb=nbformat.read(path,as_version=4)
    if nb.metadata.get("course",{}).get("role")!="beginner-code-companion":
        errors.append(f"{path.name}: missing beginner-code-companion metadata")
    code_cells=[cell for cell in nb.cells if cell.cell_type=="code"]
    if len(code_cells)<14:errors.append(f"{path.name}: expected at least 14 detailed code cells")
    text="\n".join(cell.source for cell in nb.cells)
    for marker in ["运行前，先这样读","运行后，逐项核对","常见错误","只改一处的小实验"]:
        if text.count(marker)<14:errors.append(f"{path.name}: expected 14 occurrences of {marker}")
    return errors


def validate_pair(source:Path,executed:Path):
    errors=[]
    try:
        source_nb=nbformat.read(source,as_version=4)
        executed_nb=nbformat.read(executed,as_version=4)
    except Exception as exc:
        return [f"cannot compare {source.name} with {executed.name}: {exc}"]
    if len(source_nb.cells)!=len(executed_nb.cells):
        return [f"{executed.name}: cell count differs from source {source.name}"]
    for index,(source_cell,executed_cell) in enumerate(zip(source_nb.cells,executed_nb.cells)):
        if source_cell.cell_type!=executed_cell.cell_type:
            errors.append(f"{executed.name}: cell {index} type differs from source")
        if source_cell.source!=executed_cell.source:
            errors.append(f"{executed.name}: cell {index} source differs from source notebook")
    return errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-only",action="store_true",help="Do not require or inspect executed copies")
    args=parser.parse_args()
    errors=[]
    for rel in REQUIRED:
        if not (ROOT/rel).exists():errors.append(f"missing required file: {rel}")
    if (ROOT/"data"/"audio_software_lab"/"manifest.json").exists():
        errors.extend(validate_audio_software_assets())
    if (ROOT/"data"/"audio_diagnosis_lab"/"answer_key.json").exists():
        errors.extend(validate_audio_diagnosis_assets())
    stray_executed=sorted(NB_DIR.glob("*_已运行.ipynb"))
    if stray_executed:
        errors.append(
            "executed copies must live under notebooks/_executed: "
            + ", ".join(path.name for path in stray_executed)
        )
    for name in SPECIAL_NOTEBOOKS:
        special=NB_DIR/name
        if not special.exists():errors.append(f"missing special notebook: {name}")
        else:errors.extend(validate_source_or_executed(special,False))
        if not args.source_only:
            executed_special=executed_path(special)
            if not executed_special.exists():errors.append(f"missing executed special notebook: {executed_special.name}")
            else:
                errors.extend(validate_source_or_executed(executed_special,True))
                if special.exists():errors.extend(validate_pair(special,executed_special))
    overview=NB_DIR/"零基础预备课_Python与PyTorch.ipynb"
    if overview.exists():errors.extend(validate_notebook(overview,False))
    beginner_code=NB_DIR/f"{BEGINNER_CODE_STEM}.ipynb"
    if not beginner_code.exists():errors.append(f"missing beginner code companion: {beginner_code.name}")
    else:errors.extend(validate_beginner_code_companion(beginner_code,False))
    if not args.source_only:
        executed_beginner_code=executed_path(beginner_code)
        if not executed_beginner_code.exists():errors.append(f"missing executed beginner code companion: {executed_beginner_code.name}")
        else:
            errors.extend(validate_beginner_code_companion(executed_beginner_code,True))
            if beginner_code.exists():errors.extend(validate_pair(beginner_code,executed_beginner_code))
    for lesson,stem in enumerate(FOUNDATION_STEMS,start=1):
        source=NB_DIR/f"{stem}.ipynb"
        if not source.exists():errors.append(f"missing foundation source: {source.name}")
        else:errors.extend(validate_foundation(source,lesson,False))
        if not args.source_only:
            executed_foundation=executed_path(source)
            if not executed_foundation.exists():errors.append(f"missing executed foundation: {executed_foundation.name}")
            else:
                errors.extend(validate_foundation(executed_foundation,lesson,True))
                if source.exists():errors.extend(validate_pair(source,executed_foundation))
    for lesson,stem in enumerate(AUDIO_FOUNDATION_STEMS,start=1):
        source=NB_DIR/f"{stem}.ipynb"
        if not source.exists():errors.append(f"missing audio foundation source: {source.name}")
        else:errors.extend(validate_audio_foundation(source,lesson,False))
        if not args.source_only:
            executed_audio=executed_path(source)
            if not executed_audio.exists():errors.append(f"missing executed audio foundation: {executed_audio.name}")
            else:
                errors.extend(validate_audio_foundation(executed_audio,lesson,True))
                if source.exists():errors.extend(validate_pair(source,executed_audio))
    for lesson,stem in enumerate(RESEARCH_THINKING_STEMS,start=1):
        source=NB_DIR/f"{stem}.ipynb"
        if not source.exists():errors.append(f"missing research-thinking source: {source.name}")
        else:errors.extend(validate_research_thinking(source,lesson,False))
        if not args.source_only:
            executed_thinking=executed_path(source)
            if not executed_thinking.exists():errors.append(f"missing executed research-thinking lesson: {executed_thinking.name}")
            else:
                errors.extend(validate_research_thinking(executed_thinking,lesson,True))
                if source.exists():errors.extend(validate_pair(source,executed_thinking))
    for lesson,stem in enumerate(COGNITIVE_TOOLKIT_STEMS,start=1):
        source=NB_DIR/f"{stem}.ipynb"
        if not source.exists():errors.append(f"missing cognitive-toolkit source: {source.name}")
        else:errors.extend(validate_cognitive_toolkit(source,lesson,False))
        if not args.source_only:
            executed_cognitive=executed_path(source)
            if not executed_cognitive.exists():errors.append(f"missing executed cognitive-toolkit lesson: {executed_cognitive.name}")
            else:
                errors.extend(validate_cognitive_toolkit(executed_cognitive,lesson,True))
                if source.exists():errors.extend(validate_pair(source,executed_cognitive))
    for lesson,stem in enumerate(INFORMATION_REASONING_STEMS,start=1):
        source=NB_DIR/f"{stem}.ipynb"
        if not source.exists():errors.append(f"missing information-reasoning source: {source.name}")
        else:errors.extend(validate_information_reasoning(source,lesson,False))
        if not args.source_only:
            executed_research=executed_path(source)
            if not executed_research.exists():errors.append(f"missing executed information-reasoning lesson: {executed_research.name}")
            else:
                errors.extend(validate_information_reasoning(executed_research,lesson,True))
                if source.exists():errors.extend(validate_pair(source,executed_research))
    language_model_sources=sorted(p for p in NB_DIR.glob("语言模型零基础_[0-9][0-9]_*.ipynb") if not p.stem.endswith("_已运行"))
    language_model_numbers=[language_model_number(path) for path in language_model_sources]
    last_language_model=max(language_model_numbers,default=0)
    expected_language_models=set(range(1,last_language_model+1))
    if last_language_model<MIN_LM_LAST_LESSON:
        errors.append(f"language-model course ends at lesson {last_language_model}, expected at least {MIN_LM_LAST_LESSON}")
    if set(language_model_numbers)!=expected_language_models:
        errors.append(f"language-model lesson numbers mismatch: missing={sorted(expected_language_models-set(language_model_numbers))}")
    if len(language_model_numbers)!=len(set(language_model_numbers)):
        errors.append("duplicate language-model source lesson numbers")
    for source in language_model_sources:
        errors.extend(validate_source_or_executed(source,False))
        if not args.source_only:
            executed_language_model=executed_path(source)
            if not executed_language_model.exists():errors.append(f"missing executed language-model lesson: {executed_language_model.name}")
            else:
                errors.extend(validate_source_or_executed(executed_language_model,True))
                errors.extend(validate_pair(source,executed_language_model))
    capstone=NB_DIR/f"{CAPSTONE_STEM}.ipynb"
    if not capstone.exists():errors.append(f"missing capstone source: {capstone.name}")
    else:errors.extend(validate_source_or_executed(capstone,False))
    if not args.source_only:
        executed_capstone=executed_path(capstone)
        if not executed_capstone.exists():errors.append(f"missing executed capstone: {executed_capstone.name}")
        else:
            errors.extend(validate_source_or_executed(executed_capstone,True))
            if capstone.exists():errors.extend(validate_pair(capstone,executed_capstone))
    sources=sorted(p for p in NB_DIR.glob("[0-9][0-9]_*.ipynb") if not p.stem.endswith("_已运行"))
    numbers=[lesson_number(p) for p in sources]
    last_lesson=max(numbers,default=0)
    expected=set(range(1,last_lesson+1))
    if last_lesson<MIN_LAST_LESSON:
        errors.append(f"course ends at lesson {last_lesson}, expected at least {MIN_LAST_LESSON}")
    if set(numbers)!=expected:
        errors.append(f"lesson numbers mismatch: missing={sorted(expected-set(numbers))}")
    if len(numbers)!=len(set(numbers)):errors.append("duplicate source lesson numbers")
    for path in sources:
        number=lesson_number(path)
        errors.extend(validate_course_bridge(path))
        if number in UPGRADED_LESSONS:
            errors.extend(validate_notebook(path,True))
        else:
            errors.extend(validate_frontier(path,number,False))
    if not args.source_only:
        executed=[executed_path(source) for source in sources]
        if not all(path.exists() for path in executed):
            missing=[str(path.relative_to(ROOT)) for path in executed if not path.exists()]
            errors.append(f"executed notebook set is incomplete: missing={missing}")
        for source,path in zip(sources,executed):
            if not path.exists():continue
            errors.extend(validate_course_bridge(path))
            number=lesson_number(source)
            if number in UPGRADED_LESSONS:
                errors.extend(validate_notebook(path,False))
            else:
                errors.extend(validate_frontier(path,number,True))
            errors.extend(validate_pair(source,path))
    ignored_tree_parts={".git",".venv",".local_data","__pycache__"}
    for path in ROOT.rglob("*"):
        if ignored_tree_parts.intersection(path.relative_to(ROOT).parts) or not path.is_file():continue
        if path.stat().st_size>95*1024*1024:errors.append(f"file is close to GitHub 100 MB limit: {path.relative_to(ROOT)}")
    if errors:
        print("COURSE VALIDATION FAILED")
        for error in errors:print("-",error)
        return 1
    executed_count=1+len(FOUNDATION_STEMS)+len(AUDIO_FOUNDATION_STEMS)+len(RESEARCH_THINKING_STEMS)+len(COGNITIVE_TOOLKIT_STEMS)+len(INFORMATION_REASONING_STEMS)+len(language_model_sources)+len(expected)+len(SPECIAL_NOTEBOOKS)+1
    print(f"COURSE VALIDATION PASSED: 1 beginner code companion, {len(FOUNDATION_STEMS)} PyTorch foundation, {len(AUDIO_FOUNDATION_STEMS)} audio foundation, {len(RESEARCH_THINKING_STEMS)} research thinking, {len(COGNITIVE_TOOLKIT_STEMS)} cognitive toolkit, {len(INFORMATION_REASONING_STEMS)} information reasoning, {len(language_model_sources)} LM foundation, {len(sources)} ASR lessons, {len(SPECIAL_NOTEBOOKS)} special labs, 1 capstone"+("" if args.source_only else f", {executed_count} executed copies"))
    return 0


if __name__=="__main__":sys.exit(main())
