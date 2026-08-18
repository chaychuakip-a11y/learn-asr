# 数据来源与署名

## Free Spoken Digit Dataset (FSDD)

- 上游仓库：https://github.com/Jakobovski/free-spoken-digit-dataset
- 本实验固定版本：`26eb9aaf76e81b692f806f9140c2d2777410d7a1`
- 固定 ZIP SHA256：`beaca4fb849cad13103b2cffba913b3d7fdb5f15222da4ddd07187b3c38a757a`
- 数据说明：英语数字 0～9 的 8 kHz WAV 录音
- 上游许可：Creative Commons Attribution-ShareAlike 4.0 International
- 文件命名：`{digit}_{speaker}_{index}.wav`

本课程使用或派生了以下教学文件：

- `data/spoken_digits_parts/*_jackson_0.wav`
- `data/fsdd_multispeaker/*.wav`
- `data/spoken_digits_0_to_9_8k.wav`
- `data/spoken_digits_0_to_9_16k.wav`
- 根目录兼容副本 `spoken_digits_0_to_9_16k.wav`
- `.local_data/fsdd-26eb9aaf/recordings/`：本地完整 3,000 条录音缓存，不提交 Git
- `artifacts/fsdd_speaker_disjoint_*`：由固定版本训练得到的 manifest、指标和 CTC checkpoint
- `artifacts/fsdd_loso_results.json` 与 `artifacts/fsdd_loso_checkpoints/`：六折说话人重采样指标和各折选中 checkpoint
- `data/audio_software_lab/05_noise_profile_then_speech_snr10db.wav` 与 `07_real_speech_digit_zero_8khz.wav`：用于 Audacity/Praat GUI 实验的加噪派生语音和原始教学副本
- `data/audio_diagnosis_lab/reference_clean_speech.wav` 与 `case_01.wav`～`case_24.wav`：由课程语音生成的盲诊断练习，继续遵守 FSDD 的 CC BY-SA 4.0

拼接、重采样、加噪、回声和多麦克风信号属于课程生成的教学派生数据。公开分发这些数据时应保留本文件、上游署名和 CC BY-SA 4.0 要求。

## 生成方式

- `make_practice_audio.py`：拼接数字录音并生成练习音频；
- `python -m fsdd_generalization.data`：下载固定版本、校验 SHA256，并验证 3,000 条音频契约；
- 各 Notebook：在内存中生成噪声、回声、增益和多通道模拟；
- `scripts/build_*`：生成课程 Notebook；不下载或使用私有数据。
- `scripts/build_audio_software_lab_assets.py`：生成专业音频软件实验的纯音、DC、削波、双声道延迟和 FSDD 派生语音，并写出测量 manifest。
- `scripts/build_audio_diagnosis_practice.py`：确定性生成盲诊断题、公开 manifest 和答案键；`scripts/audio_diagnosis_quiz.py` 分级核对。

不要向仓库提交包含个人录音、API 密钥、访问 token 或未经授权数据的文件。

## AudioMNIST

- 官方仓库：https://github.com/soerenab/AudioMNIST
- 本实验固定版本：`630d7dab4c040882834de6fa21baf9a60372accd`
- 固定 ZIP SHA256：`f0420069d684baff1688658d1ba53316f9b3f742a7d0ab2963a064c591f9c573`
- 数据说明：60 位说话人的 30,000 条英语数字 0～9 录音，48 kHz mono PCM_16
- 上游许可：MIT
- `.local_data/audiomnist-630d7dab/`：完整本地缓存，不提交 Git
- `artifacts/audiomnist_external_manifest.json`：文件、音频头与 speaker metadata 审计清单

AudioMNIST 只用于预注册后的外部评测。首次评分后它即属于“已接触数据”，不得在其结果上调参后仍把同一批数据称作 untouched test。
