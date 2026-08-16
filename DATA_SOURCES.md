# 数据来源与署名

## Free Spoken Digit Dataset (FSDD)

- 上游仓库：https://github.com/Jakobovski/free-spoken-digit-dataset
- 数据说明：英语数字 0～9 的 8 kHz WAV 录音
- 上游许可：Creative Commons Attribution-ShareAlike 4.0 International
- 文件命名：`{digit}_{speaker}_{index}.wav`

本课程使用或派生了以下教学文件：

- `data/spoken_digits_parts/*_jackson_0.wav`
- `data/fsdd_multispeaker/*.wav`
- `data/spoken_digits_0_to_9_8k.wav`
- `data/spoken_digits_0_to_9_16k.wav`
- 根目录兼容副本 `spoken_digits_0_to_9_16k.wav`

拼接、重采样、加噪、回声和多麦克风信号属于课程生成的教学派生数据。公开分发这些数据时应保留本文件、上游署名和 CC BY-SA 4.0 要求。

## 生成方式

- `make_practice_audio.py`：拼接数字录音并生成练习音频；
- 各 Notebook：在内存中生成噪声、回声、增益和多通道模拟；
- `scripts/build_*`：生成课程 Notebook，不下载私有数据。

不要向仓库提交包含个人录音、API 密钥、访问 token 或未经授权数据的文件。
