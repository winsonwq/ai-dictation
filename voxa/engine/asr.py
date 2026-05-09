"""
voxa/engine/asr.py
ASR 引擎入口 — 根据参数选择后端，统一接口
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

_logger = logging.getLogger('dictation.asr')


@dataclass
class ASRResult:
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = 'zh'
    avg_logprob: float = 0.0


class ASREngine:
    """
    ASR 引擎 — 后端可切换

    backend='whisper'    → WhisperCppBackend (whisper.cpp, Metal GPU)
    backend='sensevoice' → SenseVoiceBackend (阿里 FunASR, 原生中文标点)

    使用方式:
        asr = ASREngine(engine='whisper', model_size='small')
        asr.load()
        asr.start()
        for chunk in audio_chunks:
            asr.push(chunk)
        result = asr.flush()
        asr.shutdown()
    """

    def __init__(self, engine: str = 'whisper', **kwargs):
        """
        Args:
            engine: 'whisper' | 'sensevoice'
            **kwargs: 透传给具体后端 (model_size, language, n_threads, ...)
        """
        self._engine_name = engine

        if engine == 'whisper':
            from ._asr_whisper import WhisperCppBackend
            self._backend = WhisperCppBackend(**kwargs)
        elif engine == 'sensevoice':
            from ._asr_sensevoice import SenseVoiceBackend
            language = kwargs.pop('language', 'zh')  # SenseVoice 只关心 language
            self._backend = SenseVoiceBackend(language=language)
        else:
            raise ValueError(
                f'未知引擎: {engine}，可选: whisper, sensevoice'
            )

    # ---- 代理方法 ----

    def load(self):
        self._backend.load()

    def start(self):
        self._backend.start()

    def push(self, audio_chunk: np.ndarray) -> Optional[str]:
        return self._backend.push(audio_chunk)

    def flush(self, force: bool = False) -> Optional[ASRResult]:
        return self._backend.flush(force=force)

    def reset(self):
        self._backend.reset()

    def shutdown(self):
        self._backend.shutdown()
