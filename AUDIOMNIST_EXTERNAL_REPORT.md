# AudioMNIST 首次外部盲测报告

## 审计链

- protocol commit：`0b303605ea02aa486ba4de36469d0ac42cef9966`
- frozen model commit：`cac8aa682ed9f630623b9b627387cb651855b2f2`
- protocol SHA256：`43a212026a29561f8633ec135281951ec4ad4a919ddea3370904d6a78a7c40f3`
- checkpoint SHA256：`5c74192bfa1b3bbddbf614a08a60b592082e7af0981af8facd3370e84d80e337`
- AudioMNIST revision：`630d7dab4c040882834de6fa21baf9a60372accd`
- archive SHA256：`f0420069d684baff1688658d1ba53316f9b3f742a7d0ab2963a064c591f9c573`
- frozen result SHA256：`73c8d1a3426c709f20675022579b798f5e56b4a8aeb179eee26fb7d196b94d0e`

## 冻结结果

| Track | Utterances | Exact | CER | Speaker CER standard deviation |
|---|---:|---:|---:|---:|
| Single digit | 30,000 | 59.30% | 40.85% | 19.52% |
| Multi digit | 6,000 | 37.75% | 39.84% | 19.94% |

单数字 speaker-macro CER 的 95% t 区间为 35.81%～45.90%。单数字编辑操作包括 4,036 次 substitution、7,964 次 deletion、256 次 insertion；删除是主要失败模式。60 位 speaker 的单数字 exact 从 14.0% 到 92.8%，平均值不能代表任意个人。

60/60 位固定代表录音在 chunk sizes 1、137、400、800、1600 samples 下最终输出一致。这只证明流式实现一致，不证明文本正确。

按预注册区间，59.30% single-digit exact 属于 `weak_transfer_requires_separately_designed_adaptation`。结果未触发重训、模型替换或第二次“盲测”。

## 解释限制

AudioMNIST 是孤立英语数字，不能代表开放词汇、长语音、多语言、噪声远场或生产 ASR。gender、accent、native-speaker 和 age 分桶仅为描述性诊断，样本不平衡且存在 speaker/录音条件混杂，不能直接作因果或总体公平性结论。

AudioMNIST 在本次评分后已成为 contacted data。任何后续适配都必须重新预注册 train/dev/test 边界，并使用另一份未接触外部语料确认最终泛化。
