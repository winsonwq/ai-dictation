"""
voxa/engine/_asr_sensevoice.py
SenseVoice backend — ONNX Runtime 版本，不需要 PyTorch
模型: lovemefan/SenseVoice-onnx (HuggingFace 自动下载)
参考: Handy 的 transcribe-rs ONNX 方案
"""

import gc
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .asr import ASRResult

_logger = logging.getLogger('dictation.asr.sensevoice')


class SenseVoiceBackend:
    """SenseVoice ASR 后端 — ONNX Runtime，无需 PyTorch"""

    def __init__(self, language: str = 'auto'):
        self.language = language
        self._sample_rate = 16000
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._min_audio = int(self._sample_rate * 0.3)
        self._max_audio = int(self._sample_rate * 30)
        self._model = None
        self._vad = None
        self._frontend = None

    # ---- 生命周期 ----

    def load(self):
        """加载 SenseVoice ONNX 模型"""
        if self._model is not None:
            return

        _logger.info('加载 SenseVoice ONNX 模型...')

        from sensevoice.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
        from sensevoice.utils.frontend import WavFrontend
        from sensevoice.utils.fsmn_vad import FSMNVad, VADXOptions

        # 模型缓存目录
        cache_dir = Path.home() / '.cache' / 'sensevoice-onnx'
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 自动下载模型（如果不存在）
        model_dir = cache_dir / 'models' / ' SenseVoice-onnx'
        if not model_dir.exists():
            _logger.info('首次使用，自动下载 SenseVoice ONNX 模型...')
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id='lovemefan/SenseVoice-onnx',
                local_dir=str(cache_dir / 'models' / 'SenseVoice-onnx'),
            )

        model_dir = cache_dir / 'models' / 'SenseVoice-onnx'

        # 加载 ASR 模型
        self._model = SenseVoiceInferenceSession(
            str(model_dir / 'embedding.npy'),
            str(model_dir / 'sense-voice-encoder.onnx'),
            str(model_dir / 'chn_jpn_yue_eng_ko_spectok.bpe.model'),
            device_id=-1,  # CPU
            intra_op_num_threads=6,
        )

        # 加载音频前端
        self._frontend = WavFrontend(str(model_dir / 'am.mvn'))

        # 加载 VAD
        vad_options = VADXOptions(
            sample_rate=16000,
            detect_mode=1,  # kVadMutipleUtteranceDetectMode
            max_end_silence_time=800,
            max_start_silence_time=3000,
        )
        self._vad = FSMNVad(str(model_dir / 'fsmn_vad.onnx'), vad_options)

        _logger.info('SenseVoice ONNX 就绪')

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)

    def cancel(self):
        self.stop()

    # ---- 转写 ----

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        """
        完整的端到端转写：VAD 检测 + ASR
        """
        if self._model is None:
            self.load()

        # 确保音频是正确格式
        audio = self._ensure_audio_format(audio)

        # VAD 检测语音段
        speech_segments = self._vad.detect(audio)

        if not speech_segments:
            return ASRResult(text='', duration=len(audio) / self._sample_rate)

        # 取最长的语音段进行转写
        best_segment = max(speech_segments, key=lambda s: s[1] - s[0])
        start_ms, end_ms = best_segment
        start_sample = int(start_ms * 16)  # ms -> sample (16k * ms / 1000)
        end_sample = int(end_ms * 16)
        speech_audio = audio[start_sample:end_sample]

        if len(speech_audio) < self._min_audio:
            return ASRResult(text='', duration=len(audio) / self._sample_rate)

        # 提取音频特征
        audio_feat = self._frontend.get_features(speech_audio)

        # ASR 推理
        language_map = {'auto': 0, 'zh': 3, 'en': 4, 'yue': 7, 'ja': 11, 'ko': 12, 'nospeech': 13}
        lang_id = language_map.get(self.language, 0)

        result = self._model(
            audio_feat[None, ...],
            language=lang_id,
            use_itn=True,
        )

        text = result[0] if result else ''

        return ASRResult(
            text=text,
            duration=len(audio) / self._sample_rate,
        )

    def _ensure_audio_format(self, audio: np.ndarray) -> np.ndarray:
        """确保音频是 float32 1D array @ 16kHz"""
        audio = np.atleast_1d(audio).astype(np.float32)
        # 归一化到 [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def release(self):
        """释放模型（GC）"""
        self._model = None
        self._vad = None
        self._frontend = None
        gc.collect()

    # ---- 流式接口（保留，与现有框架兼容）----

    def push_audio(self, chunk: np.ndarray) -> Optional[ASRResult]:
        """流式：收音频块，返回最终结果（如果检测到语音结束）"""
        if not self._running:
            return None

        chunk = np.atleast_1d(chunk).astype(np.float32)
        max_val = np.abs(chunk).max()
        if max_val > 0:
            chunk = chunk / max_val

        with self._lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

            # 超过最大长度，截断
            if len(self._audio_buffer) > self._max_audio:
                self._audio_buffer = self._audio_buffer[-self._max_audio:]

        # 流式 VAD - 检测是否还在说话
        # 这里简化处理，完整实现需要更复杂的状态机
        return None

    def flush(self) -> Optional[ASRResult]:
        """流式：强制输出当前缓冲区的转写结果"""
        if not self._running or len(self._audio_buffer) < self._min_audio:
            return None

        with self._lock:
            audio = self._audio_buffer.copy()
            self._audio_buffer = np.array([], dtype=np.float32)

        return self.transcribe(audio)

    def get_stats(self) -> dict:
        """返回当前状态统计"""
        return {
            'engine': 'sensevoice-onnx',
            'buffer_samples': len(self._audio_buffer),
        }
