"""
voxa/core.py
VoxaCore - 核心听写引擎
事件驱动，无 UI 依赖
"""

import sys
import os
import time
import logging
import threading
from enum import Enum
from typing import Optional, Callable

import numpy as np

from .events import EventEmitter
from .engine.audio import MicrophoneCapture, FileAudioSource, SAMPLE_RATE
from .engine.vad import VADEngine
from .engine.asr import ASREngine, ASRResult
from .engine.polish import PolishEngine

_logger = logging.getLogger('voxa.core')


class State(Enum):
    IDLE = 'idle'
    LISTENING = 'listening'
    STREAMING = 'streaming'
    FLUSHING = 'flushing'
    POLISHING = 'polishing'
    ERROR = 'error'


class VoxaCore(EventEmitter):
    """
    听写核心引擎

    事件驱动，无 UI 依赖，可被 TUI、HTTP Server、桌面应用等调用

    事件列表:
        state_change(state)       → 状态变化
        vad.speech_start()        → 检测到语音开始
        vad.speech_end()          → 检测到语音结束
        asr.partial(text)        → ASR 部分结果
        asr.final(result)        → ASR 最终结果 (ASRResult)
        polish.result(text)       → 润色完成
        error(message)            → 错误发生
        audio.chunk(chunk)        → 原始音频块（可用于波形显示）

    使用方式:
        core = VoxaCore()
        core.on('asr.final', lambda r: print(f'转写: {r.text}'))
        core.on('polish.result', lambda t: print(f'润色: {t}'))
        core.start()
    """

    def __init__(
        self,
        engine: str = 'whisper',
        model_size: str = 'small',
        llm_model: str = 'qwen/qwen3.5-plus-02-15',
        disable_polish: bool = False,
        use_file: Optional[str] = None,
        device: Optional[int] = None,
        streaming: bool = False,
        flush_interval: float = 2.0,
    ):
        """
        Args:
            engine: ASR 引擎 ('whisper' | 'sensevoice')
            model_size: Whisper 模型大小 ('tiny' | 'small' | 'medium' | 'large')
            llm_model: 润色模型 ID
            disable_polish: 是否禁用 LLM 润色
            use_file: 使用文件音频源（用于测试）
            device: 麦克风设备索引
            streaming: 是否启用流式转写
            flush_interval: 流式转写刷新间隔（秒）
        """
        super().__init__()
        self.engine = engine
        self.model_size = model_size
        self.llm_model = llm_model
        self.disable_polish = disable_polish
        self.use_file = use_file
        self.device = device
        self.streaming = streaming
        self.flush_interval = flush_interval

        self._state = State.IDLE
        self._audio: Optional[MicrophoneCapture] = None
        self._file_source: Optional[FileAudioSource] = None
        self._vad: Optional[VADEngine] = None
        self._asr: Optional[ASREngine] = None
        self._polish: Optional[PolishEngine] = None

        self._current_partial = ''
        self._current_final = ''
        self._running = False
        self._t_stop: float = 0.0

    # ---- 属性 ----

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_listening(self) -> bool:
        return self._state in (State.LISTENING, State.STREAMING)

    # ---- 公开方法 ----

    def start(self):
        """开始听写"""
        if self._state in (State.LISTENING, State.STREAMING):
            return

        self._load_engines()
        self._set_state(State.LISTENING if not self.streaming else State.STREAMING)
        self._current_partial = ''
        self._current_final = ''
        self._running = True

        # 重置引擎状态
        self._vad.reset()
        self._asr.reset()
        self._asr.start()

        # 启动音频采集
        if self.use_file:
            self._file_source = FileAudioSource(self.use_file)
            self._file_source.load()
        else:
            self._audio = MicrophoneCapture(device_index=self.device)
            self._audio.start()

        # 启动处理线程
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[str]:
        """
        停止听写，返回最终转写文本
        """
        self._t_stop = time.time()
        if self._state not in (State.LISTENING, State.STREAMING):
            return None

        self._running = False

        # 停止音频采集
        if self._audio:
            self._audio.stop()
            self._audio = None

        # 获取 ASR 最终结果
        self._set_state(State.FLUSHING)
        asr_result = self._asr.flush(force=True)
        final_text = ''

        if asr_result:
            final_text = asr_result.text
            self._current_final = final_text
            self.emit('asr.final', asr_result)

        # 停止 ASR
        self._asr.shutdown()

        # LLM 润色
        if final_text and not self.disable_polish:
            self._set_state(State.POLISHING)
            try:
                polished = self._polish.polish(final_text)
                self.emit('polish.result', polished)
                return polished
            except Exception as e:
                _logger.error(f'润色失败: {e}')
                self.emit('error', f'润色失败: {e}')

        self._set_state(State.IDLE)
        return final_text or self._current_final

    def cancel(self):
        """取消当前听写（不触发润色）"""
        self._running = False
        if self._audio:
            self._audio.stop()
            self._audio = None
        if self._asr:
            self._asr.shutdown()
        self._set_state(State.IDLE)
        self._current_partial = ''
        self._current_final = ''

    def get_devices(self):
        """获取可用音频设备列表"""
        from .engine.audio import list_devices
        return list_devices()

    def check_environment(self):
        """检测运行环境"""
        from .engine.audio import check_environment
        return check_environment()

    # ---- 内部方法 ----

    def _load_engines(self):
        """延迟加载各引擎"""
        if self._vad is None:
            self._vad = VADEngine(threshold=0.5)
            self._vad.load()
            _logger.debug('VAD 引擎加载完成')

        if self._asr is None:
            self._asr = ASREngine(engine=self.engine, model_size=self.model_size)
            self._asr.load()
            _logger.debug('ASR 引擎加载完成')

        if not self.disable_polish and self._polish is None:
            self._polish = PolishEngine(model=self.llm_model)
            _logger.debug('LLM 润色引擎加载完成')

    def _set_state(self, state: State):
        self._state = state
        self.emit('state_change', state)

    def _run_loop(self):
        """主处理循环"""
        try:
            self._run_loop_impl()
        except Exception as e:
            _logger.exception(f'处理循环异常: {e}')
            self.emit('error', str(e))
            self._set_state(State.ERROR)

    def _run_loop_impl(self):
        """主处理循环实现"""
        speech_buffer = []
        silence_frames = 0
        max_silence_frames = 50  # 约 1.5 秒静音后触发结束

        while self._running:
            # 读取音频
            if self.use_file:
                chunk = self._file_source.read()
                if chunk is None:
                    # 文件播放完毕
                    break
            else:
                if self._audio is None:
                    break
                chunk = self._audio.read(timeout=0.1)
                if chunk is None:
                    continue

            # 发射音频块事件（用于波形显示）
            self.emit('audio.chunk', chunk)

            # VAD 检测
            vad_result = self._vad.push(chunk)

            if vad_result.is_speech:
                silence_frames = 0
                if not speech_buffer:
                    self.emit('vad.speech_start')
                speech_buffer.append(chunk)
            else:
                silence_frames += 1
                if speech_buffer:
                    speech_buffer.append(chunk)
                    # 静音超过阈值，认为语音结束
                    if silence_frames > max_silence_frames:
                        self._flush_speech(speech_buffer)
                        speech_buffer = []
                        silence_frames = 0

            # ASR 流式处理（非流式模式下每帧都推送）
            if not self.streaming:
                partial = self._asr.push(chunk)
                if partial:
                    self._current_partial = partial
                    self.emit('asr.partial', partial)

        # 处理剩余的语音 buffer
        if speech_buffer and self._running:
            self._flush_speech(speech_buffer)

    def _flush_speech(self, speech_buffer):
        """处理累积的语音片段"""
        audio = np.concatenate(speech_buffer) if speech_buffer else np.array([])
        if len(audio) == 0:
            return

        self.emit('vad.speech_end')

        # 推送剩余音频到 ASR
        self._asr.push(audio)

        # 获取最终结果
        asr_result = self._asr.flush(force=True)
        if asr_result and asr_result.text:
            self._current_final = asr_result.text
            self.emit('asr.final', asr_result)

            # 润色
            if not self.disable_polish:
                self._set_state(State.POLISHING)
                try:
                    polished = self._polish.polish(asr_result.text)
                    self.emit('polish.result', polished)
                except Exception as e:
                    _logger.error(f'润色失败: {e}')
            self._set_state(State.IDLE)
