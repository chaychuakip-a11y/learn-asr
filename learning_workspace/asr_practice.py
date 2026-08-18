"""请亲手补全；验证器不会把参考实现写进这个文件。"""


def sample_count(sample_rate: int, seconds: float) -> int:
    """返回录音采样点数。例如 8 kHz × 0.5 s 应得到 4000。"""

    raise NotImplementedError("请完成 sample_count")


def frame_count(num_samples: int, n_fft: int, hop_length: int) -> int:
    """返回 center=False 且 num_samples >= n_fft 时的完整帧数。"""

    raise NotImplementedError("请完成 frame_count")


def ctc_collapse(frame_ids: list[int], blank_id: int = 0) -> list[int]:
    """先合并相邻重复类别，再删除 blank；返回保留的类别 ID。"""

    raise NotImplementedError("请完成 ctc_collapse")


class StreamingCTCCollapse:
    """逐 chunk 接收帧类别；必须保存跨 chunk 的 previous class。"""

    def __init__(self, blank_id: int = 0):
        raise NotImplementedError("请完成 StreamingCTCCollapse.__init__")

    def accept(self, frame_ids: list[int]) -> list[int]:
        """接收一个 chunk，返回截至当前的完整类别序列。"""

        raise NotImplementedError("请完成 StreamingCTCCollapse.accept")


def add_k_bigram_probability(
    context_count: int,
    bigram_count: int,
    vocabulary_size: int,
    k: float = 0.1,
) -> float:
    """计算 (bigram_count+k)/(context_count+k*vocabulary_size)。"""

    raise NotImplementedError("请完成 add_k_bigram_probability")


def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    """返回 RTF；非法的非正音频时长应抛出 ValueError。"""

    raise NotImplementedError("请完成 real_time_factor")
