"""
voxa/engine/vad.py
Voice Activity Detection — ONNX Runtime 版本，不需要 PyTorch
使用 sensevoice-onnx 内置的 FSMN VAD
"""

import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

_logger = logging.getLogger('dictation.vad')

# VAD 参数
VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # 32ms @ 16kHz


@dataclass
class VADResult:
    is_speech: bool
    probability: float  # 0.0 ~ 1.0


class VADEngine:
    """
    FSMN VAD 流式封装 — ONNX Runtime 版本

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
    ):
        """
        Args:
            threshold: 语音检测阈值 (0.0~1.0)，越高越保守
            min_speech_duration_ms: 最小语音持续时间
            min_silence_duration_ms: 最小静音持续时间，触发结束检测
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        self._vad = None
        self._speech_frames = 0
        self._silence_frames = 0

    def load(self):
        """加载 FSMN VAD 模型"""
        if self._vad is not None:
            return

        _logger.info('加载 FSMN VAD 模型 (ONNX)...')

        from sensevoice.utils.fsmn_vad import FSMNVad, VADXOptions

        # 模型路径
        cache_dir = Path.home() / '.cache' / 'sensevoice-onnx' / 'models' / 'SenseVoice-onnx'

        if not cache_dir.exists():
            _logger.info('首次使用，自动下载 SenseVoice ONNX 模型（含 VAD）...')
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='lovemefan/SenseVoice-onnx',
                local_dir=str(cache_dir),
            )

        vad_model_path = cache_dir / 'fsmn_vad.onnx'
        if not vad_model_path.exists():
            raise FileNotFoundError(f'VAD 模型未找到: {vad_model_path}')

        options = VADXOptions(
            sample_rate=16000,
            detect_mode=1,  # kVadMutipleUtteranceDetectMode
            max_end_silence_time=800,
            max_start_silence_time=3000,
        )

        self._vad = FSMNVad(str(vad_model_path), options)
        _logger.info('FSMN VAD 就绪')

    def push(self, chunk: np.ndarray) -> VADResult:
        """
        处理一个音频块

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

        # FSMN VAD 检测
        # detect() 返回 List[Tuple[start_ms, end_ms]]
        segments = self._vad.detect(chunk)

        # 有语音段返回 True
        has_speech = len(segments) > 0

        # 简单的状态机逻辑
        if has_speech:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1
            self._speech_frames = 0

        # 计算概率（基于状态）
        prob = 1.0 if has_speech else 0.0

        _logger.debug(f'VAD: speech={has_speech}, prob={prob:.3f}')
        return VADResult(is_speech=has_speech, probability=prob)

    def reset(self):
        """重置状态"""
        self._speech_frames = 0
        self._silence_frames = 0

    def is_speech_active(self) -> bool:
        """当前是否处于语音活动状态"""
        return self._speech_frames > 0


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

    # 分块测试
    chunk_size = 480
    speech_chunks = 0
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        result = vad.push(chunk)
        if result.is_speech:
            speech_chunks += 1

    print(f'检测到 {speech_chunks}/{len(signal)//chunk_size} 个语音帧')
    print('\nVAD 模块测试通过')
