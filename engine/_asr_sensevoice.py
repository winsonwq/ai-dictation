"""
engine/_asr_sensevoice.py
SenseVoice backend — 阿里 FunASR SenseVoice，原生中文标点
模型: iic/SenseVoiceSmall (~160MB)，通过 funasr 调用
"""

import gc
import logging
import threading
import time
from typing import Optional

import numpy as np

from engine.asr import ASRResult

_logger = logging.getLogger('dictation.asr.sensevoice')


class SenseVoiceBackend:
    """SenseVoice ASR 后端 — 原生支持中文标点，无需 LLM 后处理"""

    def __init__(self, language: str = 'zh'):
        self.language = language

        self._sample_rate = 16000
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._min_audio = int(self._sample_rate * 0.3)   # 0.3 秒最小
        self._max_audio = int(self._sample_rate * 30)    # 30 秒上限
        self._model = None

    # ---- 生命周期 ----

    def load(self):
        if self._model is not None:
            return

        import os as _os
        import sys as _sys
        from io import StringIO as _StringIO

        _os.environ.setdefault('FUNASR_DISABLE_PROGRESS_BAR', '1')

        # 抑制 modelscope 下载进度条（直接吞掉 stdout/stderr）
        _logger.info('加载 SenseVoice 模型...')
        _old_stdout, _old_stderr = _sys.stdout, _sys.stderr
        _sys.stdout = _sys.stderr = _StringIO()
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model='iic/SenseVoiceSmall',
                disable_update=True,
                device='cpu',
                disable_log=True,
            )

            # 尝试 torch.compile 加速推理（PyTorch 2.0+）
            try:
                import torch
                if hasattr(self._model, 'model') and hasattr(torch, 'compile'):
                    self._model.model = torch.compile(
                        self._model.model, mode='reduce-overhead'
                    )
                    _logger.info('SenseVoice: torch.compile 优化已启用')
            except Exception as _e:
                _logger.debug(f'torch.compile 不可用: {_e}')
        finally:
            _sys.stdout, _sys.stderr = _old_stdout, _old_stderr

        # 预热：首次推理加载模型权重到 PyTorch，后续调用更快
        _logger.info('SenseVoice warm-up...')
        _old_stdout, _old_stderr = _sys.stdout, _sys.stderr
        _sys.stdout = _sys.stderr = _StringIO()
        try:
            dummy = np.zeros(self._min_audio, dtype=np.float32)
            self._model.generate(input=dummy, language='auto', use_itn=True)
        except Exception:
            pass  # warm-up 失败不影响使用
        finally:
            _sys.stdout, _sys.stderr = _old_stdout, _old_stderr

        _logger.info('SenseVoice 就绪')

    def start(self):
        self._running = True
        self._audio_buffer = np.array([], dtype=np.float32)

    def push(self, audio_chunk: np.ndarray) -> Optional[str]:
        if not self._running:
            return None
        with self._lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])
            if len(self._audio_buffer) > self._max_audio:
                self._audio_buffer = self._audio_buffer[-self._max_audio:]
        return None

    def flush(self) -> Optional[ASRResult]:
        if not self._running or self._model is None:
            return None
        t0 = time.time()
        with self._lock:
            buf_len = len(self._audio_buffer)
            if buf_len < self._min_audio:
                return None
            audio_for_flush = self._audio_buffer.copy()
        t_copy = time.time()

        _logger.debug(f'sensevoice flush: buffer={buf_len} (copy: {(t_copy-t0)*1000:.0f}ms)')
        text = self._transcribe(audio_for_flush)

        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)

        if not text:
            return None
        _logger.debug(f'sensevoice result: {text[:50]}...')
        return ASRResult(text=text, end_time=buf_len / self._sample_rate, language=self.language)

    def reset(self):
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)

    def shutdown(self):
        self._running = False
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
            self._model = None
        gc.collect()

    # ---- 内部 ----

    def _transcribe(self, audio: np.ndarray) -> str:
        """调用 SenseVoice 转写，返回带标点的文本"""
        import sys as _sys
        from io import StringIO as _StringIO

        t0 = time.time()
        # SenseVoice 期望 float32，范围 [-1, 1]
        clipped = np.clip(audio, -1.0, 1.0).astype(np.float32)
        t_prep = time.time()

        # 抑制输出（modelscope 进度条可能在推理时闪现）
        _old_stdout, _old_stderr = _sys.stdout, _sys.stderr
        _sys.stdout = _sys.stderr = _StringIO()

        try:
            result = self._model.generate(
                input=clipped,
                language='auto',
                use_itn=True,
            )
        except Exception as e:
            _logger.error(f'SenseVoice 转写错误: {e}')
            return ''
        finally:
            _sys.stdout, _sys.stderr = _old_stdout, _old_stderr
            t_done = time.time()

        prep_ms = (t_prep - t0) * 1000
        infer_ms = (t_done - t_prep) * 1000
        total_ms = (t_done - t0) * 1000
        _logger.debug(f'sensevoice timing: prep={prep_ms:.0f}ms infer={infer_ms:.0f}ms total={total_ms:.0f}ms (samples={len(audio)})')

        if not result or len(result) == 0:
            return ''

        # result 是一个 list[dict]，包含 text / timestamp / emotion
        text = result[0].get('text', '')
        if text:
            # SenseVoice 有时会在末尾加 <|...|> 标签，去掉
            import re
            text = re.sub(r'<\|[^|]*\|>', '', text).strip()
        return text
