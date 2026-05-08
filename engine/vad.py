"""
engine/vad.py
Voice Activity Detection - Silero VAD 封装
流式接口，返回 (is_speech: bool, speech_prob: float)
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

# VAD 参数
VAD_SAMPLE_RATE = 16000
# Silero 需要 30s 的 context 窗口，但我们做流式所以用 30ms per chunk
VAD_CHUNK_SIZE = 480  # 30ms @ 16kHz


@dataclass
class VADResult:
    is_speech: bool
    probability: float  # 0.0 ~ 1.0


class VADEngine:
    """
    Silero VAD 流式封装

    使用方式:
        vad = VADEngine()
        for chunk in audio_chunks:
            result = vad.push(chunk)  # chunk: float32 array
            if result.is_speech:
                print("有人说话")
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        window_size_samples: int = VAD_CHUNK_SIZE,
    ):
        """
        Args:
            threshold: 语音检测阈值 (0.0~1.0)，越高越保守
            min_speech_duration_ms: 最小语音持续时间，越长越严格
            min_silence_duration_ms: 最小静音持续时间，触发结束检测
            window_size_samples: 每帧样本数 (30ms = 480 @ 16kHz)
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples

        self._model = None
        self._reset_state()
        self._model_path = self._get_model_path()

    def _get_model_path(self) -> Path:
        """获取模型缓存路径"""
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        return cache_dir

    def load(self):
        """加载 Silero VAD 模型"""
        if self._model is not None:
            return

        # Silero VAD 自动下载模型
        self._model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
        )
        self._model.eval()

        # 分离 get_speech_timestamps utility
        self._get_speech_ts = utils[0]

    def _reset_state(self):
        """重置内部状态"""
        # Silero VAD 需要维持一个内部状态（previous_hiddens）
        self._hiddens = None
        self._speech_buffer = np.array([], dtype=np.float32)
        self._silence_frames = 0

    def push(self, chunk: np.ndarray) -> VADResult:
        """
        处理一个音频块

        Args:
            chunk: np.ndarray, float32, shape=(n_samples,), 16kHz
                   长度应为 window_size_samples 或其倍数

        Returns:
            VADResult
        """
        if self._model is None:
            self.load()

        # 确保是1D数组
        chunk = np.atleast_1d(chunk).astype(np.float32)

        # 如果长度不够，pad
        if len(chunk) < self.window_size_samples:
            chunk = np.pad(chunk, (0, self.window_size_samples - len(chunk)))

        # 每次只处理 window_size_samples
        if len(chunk) > self.window_size_samples:
            # 逐块处理，累积结果
            for i in range(0, len(chunk), self.window_size_samples):
                sub_chunk = chunk[i:i + self.window_size_samples]
                result = self._process_chunk(sub_chunk)
            return result

        # 转换为 tensor
        tensor = torch.from_numpy(chunk).float()

        # 流式推理
        with torch.no_grad():
            # 如果 hiddens 不为 None，需要传入
            if self._hiddens is not None:
                speech_prob, self._hiddens = self._model(tensor, self._hiddens)
            else:
                speech_prob, self._hiddens = self._model(tensor)

        prob = speech_prob.item()

        # 累积静音帧数
        if prob < self.threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        is_speech = prob >= self.threshold

        return VADResult(is_speech=is_speech, probability=prob)

    def _process_chunk(self, chunk: np.ndarray) -> VADResult:
        """内部：处理单个 window"""
        tensor = torch.from_numpy(chunk).float()

        with torch.no_grad():
            if self._hiddens is not None:
                speech_prob, self._hiddens = self._model(tensor, self._hiddens)
            else:
                speech_prob, self._hiddens = self._model(tensor)

        prob = speech_prob.item()

        if prob < self.threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        return VADResult(is_speech=prob >= self.threshold, probability=prob)

    def reset(self):
        """重置状态（在每次新的听写开始时调用）"""
        self._reset_state()

    def is_speech_active(self) -> bool:
        """当前是否处于语音活动状态"""
        return self._silence_frames < (
            self.min_silence_duration_ms / 30  # 30ms per frame
        )


class FileVADSimulator:
    """
    用于测试的 VAD 模拟器
    把音频文件当作"一直有语音"处理（方便测试 Polish 链路）
    """

    def __init__(self, audio_source):
        self._source = audio_source
        self._eof = False

    def push(self, chunk: np.ndarray) -> VADResult:
        """始终返回 is_speech=True，模拟一直说话"""
        return VADResult(is_speech=True, probability=1.0)

    def reset(self):
        pass


if __name__ == '__main__':
    # 简单测试：生成测试音频并验证 VAD
    import sys

    print('=== VAD 模块测试 ===')

    vad = VADEngine(threshold=0.5)
    vad.load()
    print('✅ Silero VAD 模型加载成功')

    # 生成 3 秒测试音频（模拟人声频谱）
    # 人声主要集中在 100-4000Hz，用正弦波叠加模拟
    import math

    duration = 3.0  # 秒
    t = np.linspace(0, duration, int(VAD_SAMPLE_RATE * duration), dtype=np.float32)

    # 叠加几个频率模拟语音特征
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +   # 基频
        0.2 * np.sin(2 * np.pi * 400 * t) +   # 共振峰
        0.1 * np.sin(2 * np.pi * 800 * t) +   # 高频
        0.05 * np.random.randn(len(t))          # 噪音
    )
    signal = np.clip(signal, -1.0, 1.0)

    # 分块测试
    chunk_size = 480  # 30ms
    speech_chunks = 0
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        result = vad.push(chunk)
        if result.is_speech:
            speech_chunks += 1

    print(f'检测到 {speech_chunks}/{len(signal)//chunk_size} 个语音帧')
    if speech_chunks > 50:
        print('✅ VAD 检测正常（预期>50帧有语音）')
    else:
        print(f'⚠️  VAD 较保守，检测到 {speech_chunks} 帧')

    print('\nVAD 模块测试通过')
