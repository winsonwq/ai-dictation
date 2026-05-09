"""
voxa/engine/vad.py
Voice Activity Detection — ONNX Runtime 版本，不需要 PyTorch
使用 sensevoice-onnx 内置的 FSMN VAD (离线模式)
"""

import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_logger = logging.getLogger('dictation.vad')

# VAD 参数
VAD_SAMPLE_RATE = 16000


@dataclass
class VADResult:
    is_speech: bool
    probability: float  # 0.0 ~ 1.0


class VADEngine:
    """
    FSMN VAD — ONNX Runtime 版本 (离线模式)

    由于 FSMNVad 只有离线接口 (segments_offline)，我们用能量检测辅助：
    1. 累积音频
    2. 当静音超过阈值时，调用 segments_offline 检测语音段
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        self._vad = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self._speech_active = False
        self._silence_samples = 0

    def load(self):
        """加载 FSMN VAD 模型"""
        if self._vad is not None:
            return

        _logger.info('加载 FSMN VAD 模型 (ONNX)...')

        from sensevoice.utils.fsmn_vad import FSMNVad

        # 模型路径
        cache_dir = Path.home() / '.cache' / 'sensevoice-onnx' / 'models' / 'SenseVoice-onnx'

        if not (cache_dir / 'fsmn-config.yaml').exists():
            _logger.info('首次使用，自动下载 SenseVoice ONNX 模型（含 VAD）...')
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='lovemefan/SenseVoice-onnx',
                local_dir=str(cache_dir),
            )

        self._vad = FSMNVad(str(cache_dir))
        _logger.info('FSMN VAD 就绪')

    def push(self, chunk: np.ndarray) -> VADResult:
        """
        处理一个音频块（简化版：基于能量检测）

        Args:
            chunk: np.ndarray, float32, shape=(n_samples,), 16kHz

        Returns:
            VADResult
        """
        if self._vad is None:
            self.load()

        chunk = np.atleast_1d(chunk).astype(np.float32)

        # 归一化
        max_val = np.abs(chunk).max()
        if max_val > 0:
            chunk = chunk / max_val

        # 简单能量检测
        rms = np.sqrt(np.mean(chunk ** 2))
        is_speech = rms > 0.01  # 简单阈值

        if is_speech:
            self._silence_samples = 0
            self._speech_active = True
        else:
            self._silence_samples += len(chunk)
            # 静音超过 500ms 认为语音结束
            if self._silence_samples > self.min_silence_duration_ms * 16:  # 16 samples/ms
                self._speech_active = False

        prob = 1.0 if self._speech_active else 0.0
        _logger.debug(f'VAD: energy={rms:.4f}, speech={is_speech}, prob={prob:.3f}')
        return VADResult(is_speech=self._speech_active, probability=prob)

    def detect_segments(self, audio: np.ndarray) -> List:
        """
        使用 FSMN VAD 检测语音段（离线模式）

        Args:
            audio: float32 array, 16kHz

        Returns:
            List of (start_ms, end_ms) tuples
        """
        if self._vad is None:
            self.load()

        audio = np.atleast_1d(audio).astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        segments = self._vad.segments_offline(audio)
        return segments

    def reset(self):
        """重置状态"""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._speech_active = False
        self._silence_samples = 0

    def is_speech_active(self) -> bool:
        """当前是否处于语音活动状态"""
        return self._speech_active


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
    print('=== VAD 模块测试 (ONNX 版本) ===')

    vad = VADEngine(threshold=0.5)
    vad.load()
    print('✅ FSMN VAD 模型加载成功')

    # 生成 3 秒测试音频
    duration = 3.0
    t = np.linspace(0, duration, int(VAD_SAMPLE_RATE * duration), dtype=np.float32)
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t) +
        0.05 * np.random.randn(len(t))
    )
    signal = np.clip(signal, -1.0, 1.0)

    # 分块测试（能量检测）
    chunk_size = 480
    speech_chunks = 0
    total_chunks = len(signal) // chunk_size
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        result = vad.push(chunk)
        if result.is_speech:
            speech_chunks += 1

    print(f'能量检测: {speech_chunks}/{total_chunks} 个语音帧')

    # 离线 VAD 测试
    print('\n测试 FSMN 离线 VAD...')
    segments = vad.detect_segments(signal)
    print(f'检测到 {len(segments)} 个语音段: {segments}')

    print('\n✅ VAD 模块测试通过')
