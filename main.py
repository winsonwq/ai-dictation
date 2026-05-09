#!/usr/bin/env python3
"""
ai-dictation main.py
命令行 AI 听写工具
"""

import os
import sys
import time
import signal
import argparse
import threading
import queue
import logging
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Optional, Union

# 确保 engine 模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 日志配置：写入文件，UI 只显示关键信息
def setup_logging():
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dictation.log')
    # 每次运行从空日志开始
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
        filename=log_file,
        filemode='w',
    )
    return logging.getLogger('dictation')

_logger = setup_logging()

# 延迟导入，避免没有麦克风时 import 报错
from engine.audio import MicrophoneCapture, FileAudioSource, check_environment
from engine.vad import VADEngine, VAD_CHUNK_SIZE
from engine.asr import ASREngine
from engine.polish import PolishEngine, _polish_cache


# ===================== State Machine =====================

class State(Enum):
    IDLE = 'idle'
    LISTENING = 'listening'
    FLUSHING = 'flushing'
    POLISHING = 'polishing'
    ERROR = 'error'


STATE_PROMPTS = {
    State.IDLE: '🎤 按空格开始听写',
    State.LISTENING: '🎙️ 听写中... (按空格停止)',
    State.FLUSHING: '⏳ 转写中...',
    State.POLISHING: '✨ 润色中...',
    State.ERROR: '❌ 错误',
}


# ===================== UI =====================

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class RichUI:
    """Rich 终端 UI - 使用简单打印模式避免刷新闪烁"""

    def __init__(self, engine: str = 'whisper', model: str = 'small'):
        self.console = Console()
        self._engine = engine
        self._model = model
        self._state = State.IDLE
        self._partial = ''
        self._final = ''
        self._polished = ''
        self._lines = []

    def start(self):
        self._lines = []
        self.console.clear()
        self._print_ui()

    def _print_ui(self):
        """打印 UI 到控制台"""
        self.console.clear()

        state_color = {
            State.IDLE: 'white',
            State.LISTENING: 'green',
            State.FLUSHING: 'yellow',
            State.POLISHING: 'blue',
            State.ERROR: 'red',
        }.get(self._state, 'white')

        # 打印状态
        self.console.print(f"[bold]AI Dictation[/bold] [dim]({self._engine}/{self._model})[/dim]")
        self.console.print(f"[{state_color}]{STATE_PROMPTS[self._state]}[/{state_color}]")

        # 打印转写内容
        if self._polished:
            self.console.print(f"[dim]📝 转写: {self._final}[/dim]")
            self.console.print(f"[bold green]✨ 润色: {self._polished}[/bold green]")
        elif self._final:
            timing = f" [dim]({self._elapsed:.1f}s)[/dim]" if getattr(self, '_elapsed', None) else ""
            self.console.print(f"📝 转写: {self._final}{timing}")
        elif self._partial:
            self.console.print(f"📝 {self._partial}", style="dim")

        # 打印历史
        if self._lines:
            self.console.print("─" * 50, style="dim")
            for line in self._lines[-3:]:
                self.console.print(line, style="dim")

        # 打印帮助
        self.console.print("─" * 50, style="dim")
        self.console.print("[dim]空格: 开始/停止听写   |   q: 退出[/dim]")

    def update(self, state: State, partial='', final='', polished='', elapsed=None):
        self._state = state
        if partial:
            self._partial = partial
        if final:
            self._final = final
            self._partial = ''
            self._elapsed = elapsed
        if polished:
            self._polished = polished
        if state == State.IDLE and final:
            # 保存到历史
            from datetime import datetime
            ts = datetime.now().strftime('%H:%M:%S')
            self._lines.append(f'[{ts}] {polished or final}')
            self._final = ''
            self._partial = ''
            self._polished = ''
            self._elapsed = None

        self._print_ui()

    def stop(self):
        pass

    def clear(self):
        self.console.clear()


class PlainUI:
    """无 Rich 依赖的纯文本 UI"""

    def __init__(self, engine: str = 'whisper', model: str = 'small'):
        self._engine = engine
        self._model = model
        self._state = State.IDLE
        self._lines = []
        self._partial = ''
        self._final = ''
        self._polished = ''

    def start(self):
        self._lines = []
        os.system('clear')
        self._print_ui()

    def _print_ui(self):
        """打印 UI"""
        os.system('clear')
        print(f'=== AI Dictation ({self._engine}/{self._model}) ===')
        print(f'状态: {STATE_PROMPTS[self._state]}')

        if self._polished:
            print(f'转写: {self._final}')
            print(f'润色: {self._polished}')
        elif self._final:
            timing = f' ({self._elapsed:.1f}s)' if getattr(self, '_elapsed', None) else ''
            print(f'转写: {self._final}{timing}')
        elif self._partial:
            print(f'转写: {self._partial}')

        if self._lines:
            print('─' * 40)
            for l in self._lines[-3:]:
                print(l)

        print('─' * 40)
        print('空格: 开始/停止   q: 退出')

    def update(self, state: State, partial='', final='', polished='', elapsed=None):
        self._state = state
        if partial:
            self._partial = partial
        if final:
            self._final = final
            self._partial = ''
            self._elapsed = elapsed
        if polished:
            self._polished = polished
        if state == State.IDLE and final:
            from datetime import datetime
            ts = datetime.now().strftime('%H:%M:%S')
            self._lines.append(f'[{ts}] {polished or final}')
            self._final = ''
            self._partial = ''
            self._polished = ''
            self._elapsed = None

        self._print_ui()

    def stop(self):
        pass

    def clear(self):
        os.system('clear')


def get_ui(engine: str = 'whisper', model: str = 'small') -> Union[RichUI, PlainUI]:
    return RichUI(engine=engine, model=model) if HAS_RICH else PlainUI(engine=engine, model=model)


# ===================== Core Engine =====================

class DictationEngine:
    """
    听写引擎
    整合音频采集、VAD、ASR、LLM 润色
    """

    def __init__(
        self,
        engine: str = 'whisper',
        model_size: str = 'small',
        llm_model: str = 'qwen/qwen3.5-plus-02-15',
        disable_polish: bool = False,
        use_file: Optional[str] = None,
        device: Optional[int] = None,
    ):
        self.engine = engine
        self.model_size = model_size
        self.llm_model = llm_model
        self.disable_polish = disable_polish
        self.use_file = use_file
        self.device = device

        self._state = State.IDLE
        self._audio: Optional[MicrophoneCapture] = None
        self._file_source: Optional[FileAudioSource] = None
        self._vad: Optional[VADEngine] = None
        self._asr: Optional[ASREngine] = None
        self._polish: Optional[PolishEngine] = None

        self._audio_buffer = []
        self._current_partial = ''
        self._current_final = ''
        self._speech_active = False
        self._silence_count = 0
        self._running = False

        # 回调
        self.on_transcript = None  # (text: str, is_partial: bool) -> None
        self.on_polish = None     # (text: str) -> None
        self.on_state_change = None

    def _set_state(self, state: State):
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _load_engines(self):
        """加载各引擎（延迟加载）"""
        if self._vad is None:
            self._vad = VADEngine(threshold=0.5)
            self._vad.load()
            _logger.debug('VAD 引擎加载完成')

        if self._asr is None:
            self._asr = ASREngine(
                engine=self.engine,
                model_size=self.model_size,
            )
            self._asr.load()
            _logger.debug('ASR 引擎加载完成')

        if not self.disable_polish and self._polish is None:
            self._polish = PolishEngine(model=self.llm_model)
            _logger.debug('LLM 润色引擎加载完成')

    def start(self):
        """开始听写"""
        if self._state == State.LISTENING:
            return

        self._load_engines()
        self._set_state(State.LISTENING)
        self._current_partial = ''
        self._current_final = ''
        self._speech_active = False
        self._silence_count = 0
        self._audio_buffer = []
        self._running = True

        # 重置 VAD 状态
        self._vad.reset()
        self._asr.reset()
        self._asr.start()  # 启动 ASR

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

    def stop(self):
        """停止听写"""
        self._t_stop = time.time()
        _logger.debug(f'stop() called, current state={self._state}')
        if self._state != State.LISTENING:
            return

        self._running = False
        # 停止音频采集
        if self._audio:
            self._audio.stop()
            self._audio = None
        _logger.debug(f'stop: audio stopped in {(time.time()-self._t_stop)*1000:.0f}ms')

        self._set_state(State.FLUSHING)

    def _run_loop(self):
        """主处理循环"""
        _logger.debug('_run_loop() 开始')
        try:
            if self._audio:
                _logger.debug('_run_loop: 使用 MicrophoneCapture')
                source = self._audio.read_all()
            else:
                _logger.debug('_run_loop: 使用 FileAudioSource')
                source = self._file_source.read_all()

            _logger.debug('_run_loop: 开始迭代 chunks')
            chunk_count = 0
            for chunk in source:
                if not self._running:
                    _logger.debug(f'_run_loop: _running=False, 退出, 处理了 {chunk_count} 个 chunk (loop lag: {(time.time()-self._t_stop)*1000:.0f}ms)')
                    break
                chunk_count += 1
                if chunk_count <= 3 or chunk_count % 10 == 0:
                    _logger.debug(f'_run_loop: 处理 chunk #{chunk_count}, len={len(chunk)}')

                # 1. VAD 检测 + 能量计算
                vad_result = self._vad.push(chunk)
                chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
                if chunk_count <= 3:
                    _logger.debug(f'VAD: is_speech={vad_result.is_speech}, prob={vad_result.probability:.3f}, rms={chunk_rms:.4f}')

                # 2. 语音段开始检测（VAD）
                if vad_result.is_speech:
                    if not self._speech_active:
                        self._speech_active = True
                        _logger.debug(f'语音段开始 (prob={vad_result.probability:.3f})')
                        if self._asr:
                            self._asr.reset()

                # 3. 累积音频到 ASR（仅在语音活动时；push 只累积不做推理）
                if self._asr is not None and self._speech_active:
                    self._asr.push(chunk)

            # Flush 最终结果
            if chunk_count > 0:
                _logger.debug(f'_run_loop: 循环结束, 共处理 {chunk_count} 个 chunk')
            self._flush()

        except Exception as e:
            _logger.error(f'引擎错误: {e}')
            self._set_state(State.ERROR)

    def _flush(self):
        """强制输出最终结果"""
        _logger.debug('_flush() called')
        self._set_state(State.FLUSHING)

        if self._asr:
            t0 = time.time()
            result = self._asr.flush()
            elapsed = time.time() - t0
            if result:
                self._current_final = result.text
                self._last_elapsed = elapsed
                if self.on_transcript:
                    self.on_transcript(result.text, is_partial=False, elapsed=elapsed)

        # 润色
        if self._current_final and not self.disable_polish:
            self._set_state(State.POLISHING)
            # 检查缓存
            cached = _polish_cache.get(self._current_final)
            if cached:
                polished = cached
            else:
                polished = self._polish.polish(self._current_final)
                _polish_cache.set(self._current_final, polished)

            if self.on_polish:
                self.on_polish(polished)

        self._set_state(State.IDLE)

    def shutdown(self):
        """关闭引擎"""
        self._running = False
        if self._audio:
            self._audio.stop()
            self._audio = None
        if self._asr:
            self._asr.shutdown()
            self._asr = None


# ===================== Input Handler =====================

class KeyboardHandler:
    """键盘事件处理（跨平台）"""

    def __init__(self, on_space, on_q):
        self._on_space = on_space
        self._on_q = on_q
        self._running = True

    def _is_interactive(self):
        """检查是否在交互式终端中运行"""
        return sys.stdin.isatty() and sys.stdout.isatty()

    def start(self):
        """启动键盘监听"""
        # 非交互模式
        if not self._is_interactive():
            _logger.warning('非交互模式，无法读取按键')
            _logger.warning('请在真实终端中运行此程序')
            return

        try:
            import tty
            import termios

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)

            def restore():
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            try:
                tty.setcbreak(fd)
                _logger.debug('键盘监听开始')

                while self._running:
                    # 使用 sys.stdin.read 直接读取（非阻塞）
                    import select
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if ready:
                        try:
                            ch = sys.stdin.read(1)
                            if not ch:
                                break
                            _logger.debug(f'键盘读到: {repr(ch)} ord={ord(ch) if ch else 0}')
                            if ch == ' ':
                                _logger.debug('检测到空格键')
                                self._on_space()
                            elif ch in ('q', 'Q'):
                                _logger.debug('检测到 Q 键')
                                self._on_q()
                                break
                            else:
                                _logger.debug(f'未知按键: {repr(ch)}')
                        except (IOError, EOFError):
                            break

            except termios.error as e:
                _logger.error(f'tty 错误: {e}')
            except Exception as e:
                _logger.error(f'键盘处理错误: {e}')
            finally:
                restore()

        except ImportError as e:
            _logger.error(f'键盘模块导入失败: {e}')

    def stop(self):
        self._running = False


# ===================== Main =====================

def parse_args():
    parser = argparse.ArgumentParser(description='AI Dictation CLI')
    parser.add_argument('--engine', default='whisper',
                        choices=['whisper', 'sensevoice'],
                        help='ASR 引擎: whisper (whisper.cpp) | sensevoice (带标点)')
    parser.add_argument('--model', default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper 模型大小')
    parser.add_argument('--llm-model', default='qwen/qwen3.5-plus-02-15',
                        help='LLM 模型')
    parser.add_argument('--polish', action='store_true',
                        help='启用 LLM 润色（默认关闭）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--input-file', type=str,
                        help='从文件读取音频（用于测试，WAV 16kHz mono）')
    parser.add_argument('--device', type=int, default=None,
                        help='麦克风设备编号（用 --check 查看可用设备）')
    parser.add_argument('--check', action='store_true',
                        help='仅检查环境')
    return parser.parse_args()


def cmd_check():
    """环境检测"""
    print('=== AI Dictation 环境检测 ===\n')

    # 音频环境
    print('音频环境:')
    info = check_environment()
    print(f'  运行模式: {info["mode"]}')
    print(f'  PortAudio: {"✅" if info["has_portaudio"] else "❌ (不可用，将使用文件模式)"}')
    print(f'  麦克风: {"✅" if info["has_microphone"] else "❌ (不可用)"}')
    if info['devices']:
        print('  可用设备:')
        for d in info['devices']:
            print(f'    [{d["index"]}] {d["name"]}')
    print()

    # Python 依赖
    print('Python 依赖:')
    deps = [
        ('silero_vad', 'silero-vad'),
        ('sounddevice', 'sounddevice'),
        ('openai', 'openai'),
        ('rich', 'rich'),
        ('torch', 'torch'),
    ]
    for module, package in deps:
        try:
            __import__(module)
            print(f'  ✅ {package}')
        except ImportError:
            print(f'  ❌ {package} (未安装: pip install {package})')
    print()

    # OpenRouter
    print('OpenRouter API:')
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if api_key:
        print(f'  ✅ API key 已设置 ({api_key[:10]}...)')
    else:
        print('  ⚠️  未设置 OPENROUTER_API_KEY 环境变量')

    print()
    print('环境检测完成。运行 python main.py 开始听写。')


def main():
    args = parse_args()

    if args.check:
        cmd_check()
        return

    _logger.info(f'启动中 (引擎: {args.engine}, 模型: {args.model})')

    # 创建引擎
    engine = DictationEngine(
        engine=args.engine,
        model_size=args.model,
        llm_model=args.llm_model,
        disable_polish=not args.polish,
        use_file=args.input_file,
        device=args.device,
    )

    # 预加载引擎，启动后直接可用
    _logger.info('预加载引擎中...')
    engine._load_engines()
    _logger.info('引擎就绪，按空格开始')

    # 创建 UI
    ui = get_ui(engine=args.engine, model=args.model)

    # 事件队列（用于线程间通信）
    event_queue = queue.Queue()
    running = True

    # 回调 - 发送到队列
    def on_transcript(text, is_partial, elapsed=None):
        event_queue.put(('transcript', text, is_partial, elapsed))

    def on_polish(text):
        event_queue.put(('polish', text))

    def on_state_change(state):
        event_queue.put(('state', state))

    engine.on_transcript = on_transcript
    engine.on_polish = on_polish
    engine.on_state_change = on_state_change

    # 键盘处理 - 发送到队列
    _last_space_time = 0.0

    def on_space():
        nonlocal _last_space_time
        _logger.debug('on_space() 被调用')
        now = time.time()
        if now - _last_space_time < 0.3:
            _logger.debug(f'on_space() 忽略（去抖，间隔{now - _last_space_time:.2f}s < 0.3s）')
            return
        _last_space_time = now
        event_queue.put(('key', 'space'))
        _logger.debug(f'on_space() 事件已入队，队列大小={event_queue.qsize()}')

    def on_q():
        _logger.debug('on_q() 被调用')
        event_queue.put(('key', 'quit'))

    # Ctrl+C 处理
    def on_sigint(signum, frame):
        on_q()

    signal.signal(signal.SIGINT, on_sigint)

    # 启动 UI
    ui.start()

    # 启动键盘线程
    kb = KeyboardHandler(on_space=on_space, on_q=on_q)
    kb_thread = threading.Thread(target=kb.start, daemon=True)
    kb_thread.start()

    # 主循环 - 处理事件
    try:
        while running:
            try:
                # 非阻塞获取事件
                event = event_queue.get(timeout=0.1)

                if event[0] == 'key':
                    key = event[1]
                    if key == 'space':
                        _logger.debug(f'空格键: engine._state={engine._state}')
                        if engine._state == State.IDLE:
                            engine.start()
                            ui.update(State.LISTENING)
                        elif engine._state == State.LISTENING:
                            engine.stop()
                    elif key == 'quit':
                        running = False
                        break

                elif event[0] == 'transcript':
                    _, text, is_partial, elapsed = event
                    engine._current_partial = text
                    _logger.debug(f'转写: partial={is_partial}, text={text[:50] if text else ""}... ({elapsed:.1f}s)' if elapsed else '')
                    if is_partial:
                        ui.update(State.LISTENING, partial=text)
                    else:
                        engine._current_final = text
                        ui.update(State.FLUSHING, final=text, elapsed=elapsed)

                elif event[0] == 'polish':
                    text = event[1]
                    engine._current_final = engine._current_final or engine._current_partial
                    engine._current_partial = ''
                    _logger.debug(f'润色完成: {text[:50] if text else ""}...')
                    ui.update(State.IDLE, final=engine._current_final, polished=text)

                elif event[0] == 'state':
                    state = event[1]
                    _logger.debug(f'状态变化: {state}')
                    if state == State.IDLE:
                        engine._current_final = ''
                        engine._current_partial = ''

            except queue.Empty:
                # 超时，继续循环
                pass

    finally:
        engine.shutdown()
        kb.stop()
        print('\n👋 退出')


if __name__ == '__main__':
    main()
