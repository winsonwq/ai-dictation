"""
engine/stream_transcriber.py
流式转写组件 - 定时批量转写实现
"""

import logging
import threading
from typing import Callable, Optional

import numpy as np

_logger = logging.getLogger('dictation.stream')


class StreamTranscriber:
    """
    流式转写器 - 基于定时批量转写的流式输出

    算法：
    - _all_audio: 存放从开始说话到现在所有累积的音频片段
    - 每次 feed：新音频追加到 _all_audio
    - 每次 flush：用 _all_audio 中所有音频转写
    - 注意：Whisper 最大处理 30 秒音频，超出部分会被截断
    """

    def __init__(
        self,
        asr_engine,  # ASREngine 实例
        flush_interval: float = 2.0,
        on_transcript: Optional[Callable[[str], None]] = None,
    ):
        self._asr = asr_engine
        self._flush_interval = flush_interval
        self._on_transcript = on_transcript

        self._all_audio = []  # 所有累积的音频片段
        self._lock = threading.Lock()
        self._running = False
        self._timer: Optional[threading.Timer] = None

    def start(self):
        """启动流式转写"""
        _logger.debug(f'StreamTranscriber.start() called, was_running={self._running}')
        self._running = True
        self._all_audio.clear()
        self._schedule_next()

    def stop(self):
        """停止流式转写"""
        _logger.debug(f'StreamTranscriber.stop() called, running={self._running}')
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def feed(self, audio_chunk: np.ndarray):
        """接收音频块，累积到所有音频缓冲区"""
        with self._lock:
            self._all_audio.append(audio_chunk)
            _logger.debug(f'StreamTranscriber.feed(): buffer size={len(self._all_audio)} chunks')

    def _schedule_next(self):
        """调度下一次转写"""
        if not self._running:
            return
        self._timer = threading.Timer(self._flush_interval, self._do_flush)
        self._timer.daemon = True
        self._timer.start()

    def _do_flush(self):
        """执行一次转写"""
        if not self._running:
            return

        # 取出所有累积的音频
        with self._lock:
            if not self._all_audio:
                _logger.debug('StreamTranscriber._do_flush: 无音频，跳过')
                self._schedule_next()
                return
            audio_chunks = list(self._all_audio)
            _logger.debug(f'StreamTranscriber._do_flush: {len(audio_chunks)} chunks, {sum(len(c) for c in audio_chunks)} samples')

        # 用所有累积音频转写
        try:
            audio = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
            if len(audio) == 0:
                self._schedule_next()
                return

            # 重置 ASR，喂入所有累积音频，Whisper 看到完整上下文
            self._asr.reset()
            self._asr.push(audio)
            result = self._asr.flush(force=True)

            if result and result.text:
                text = result.text.strip()
                if self._on_transcript:
                    self._on_transcript(text)
            else:
                _logger.debug('StreamTranscriber: 转写无结果')

        except Exception as e:
            _logger.error(f'StreamTranscriber 转写错误: {e}')

        self._schedule_next()

    def flush_final(self) -> Optional[str]:
        """最终转写 - 用于说话结束后获取完整结果"""
        self.stop()

        with self._lock:
            if not self._all_audio:
                return None
            audio_chunks = list(self._all_audio)

        if not audio_chunks:
            return None

        try:
            audio = np.concatenate(audio_chunks)
            self._asr.reset()
            self._asr.push(audio)
            result = self._asr.flush(force=True)
            if result and result.text:
                return result.text.strip()
        except Exception as e:
            _logger.error(f'StreamTranscriber 最终转写错误: {e}')

        return None
