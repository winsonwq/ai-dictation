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
from datetime import datetime
from enum import Enum

# 确保 engine 模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 延迟导入，避免没有麦克风时 import 报错
from engine.audio import MicrophoneCapture, FileAudioSource, FileVADSimulator, check_environment
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
    State.FLUSHING: '⏳ 处理中...',
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
    """Rich 终端 UI"""

    def __init__(self):
        self.console = Console()
        self.live = None
        self._lines = []
        self._state = State.IDLE
        self._partial = ''
        self._final = ''
        self._polished = ''

    def start(self):
        self._lines = []
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self.live.start()

    def _render(self):
        """渲染当前状态"""
        if not HAS_RICH:
            return self._plain_render()

        lines = []

        # Header
        header = Text('AI Dictation', style='bold cyan')
        lines.append(Panel(header, title='ai-dictation', border_style='cyan'))

        # State
        state_color = {
            State.IDLE: 'white',
            State.LISTENING: 'green',
            State.FLUSHING: 'yellow',
            State.POLISHING: 'blue',
            State.ERROR: 'red',
        }.get(self._state, 'white')
        state_text = Text(f'状态: {STATE_PROMPTS[self._state].split("(")[0].strip()}', style=state_color)
        lines.append(state_text)

        # Transcript area
        if self._polished:
            lines.append(Text(f'📝 转写: {self._final}', style='dim'))
            lines.append(Text(f'✨ 润色: {self._polished}', style='bold green'))
        elif self._final:
            lines.append(Text(f'📝 转写: {self._final}', style='white'))
        elif self._partial:
            lines.append(Text(f'📝 {self._partial}', style='dim'))

        # History
        if len(self._lines) > 0:
            lines.append(Text('─' * 50, style='dim'))
            for line in self._lines[-3:]:
                lines.append(Text(line, style='dim'))

        # Help
        lines.append(Text('─' * 50, style='dim'))
        lines.append(Text('空格: 开始/停止听写   |   q: 退出', style='dim'))

        return '\n'.join(str(l) for l in lines)

    def _plain_render(self):
        """纯文本降级渲染"""
        lines = [
            '=== AI Dictation ===',
            f'状态: {STATE_PROMPTS[self._state]}',
        ]
        if self._polished:
            lines.append(f'转写: {self._final}')
            lines.append(f'润色: {self._polished}')
        elif self._final:
            lines.append(f'转写: {self._final}')
        elif self._partial:
            lines.append(f'转写: {self._partial}')
        lines.append('空格: 开始/停止   q: 退出')
        return '\n'.join(lines)

    def update(self, state: State, partial='', final='', polished=''):
        """更新 UI"""
        self._state = state
        if partial:
            self._partial = partial
        if final:
            self._final = final
            self._partial = ''
        if polished:
            self._polished = polished
        if state == State.IDLE and final:
            # 保存到历史
            ts = datetime.now().strftime('%H:%M:%S')
            self._lines.append(f'[{ts}] {polished or final}')
            self._final = ''
            self._partial = ''
            self._polished = ''

        if self.live:
            self.live.update(self._render())

    def stop(self):
        if self.live:
            self.live.stop()

    def clear(self):
        self.console.clear()


class PlainUI:
    """无 Rich 依赖的纯文本 UI"""

    def __init__(self):
        self._state = State.IDLE
        self._lines = []

    def start(self):
        pass

    def update(self, state: State, partial='', final='', polished=''):
        self._state = state
        os.system('clear')
        lines = ['=== AI Dictation ===', '']
        lines.append(STATE_PROMPTS[state])

        if polished:
            lines.append(f'转写: {final}')
            lines.append(f'润色: {polished}')
        elif final:
            lines.append(f'转写: {final}')
        elif partial:
            lines.append(f'转写: {partial}')

        if self._lines:
            lines.append('─' * 40)
            for l in self._lines[-3:]:
                lines.append(l)

        lines.append('─' * 40)
        lines.append('空格: 开始/停止   q: 退出')

        print('\n'.join(lines))

        if state == State.IDLE and final:
            ts = datetime.now().strftime('%H:%M:%S')
            self._lines.append(f'[{ts}] {polished or final}')
            self._final = ''
            self._partial = ''
            self._polished = ''

    def stop(self):
        pass


def get_ui() -> RichUI | PlainUI:
    return RichUI() if HAS_RICH else PlainUI()


# ===================== Core Engine =====================

class DictationEngine:
    """
    听写引擎
    整合音频采集、VAD、ASR、LLM 润色
    """

    def __init__(
        self,
        model_size: str = 'small',
        llm_model: str = 'qwen/qwen2.5-72b-instruct',
        disable_polish: bool = False,
        use_file: Optional[str] = None,
    ):
        self.model_size = model_size
        self.llm_model = llm_model
        self.disable_polish = disable_polish
        self.use_file = use_file

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
            print('[引擎] VAD 加载完成')

        if self._asr is None:
            self._asr = ASREngine(
                model_size=self.model_size,
                device='cpu',
                compute_type='int8',
            )
            self._asr.load()
            print('[引擎] ASR 加载完成')

        if not self.disable_polish and self._polish is None:
            self._polish = PolishEngine(model=self.llm_model)
            print('[引擎] LLM 加载完成')

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

        # 启动音频采集
        if self.use_file:
            self._file_source = FileAudioSource(self.use_file)
            self._file_source.load()
        else:
            self._audio = MicrophoneCapture()
            self._audio.start()

        # 启动处理线程
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止听写"""
        if self._state != State.LISTENING:
            return

        self._running = False
        self._set_state(State.FLUSHING)

    def _run_loop(self):
        """主处理循环"""
        SilenceFrames = 30  # 约 1 秒静音触发 flush

        try:
            if self._audio:
                source = self._audio.read_all()
            else:
                source = self._file_source.read_all()

            for chunk in source:
                if not self._running:
                    break

                # 1. VAD 检测
                vad_result = self._vad.push(chunk)
                is_speech = vad_result.is_speech

                # 2. 状态机
                if is_speech:
                    self._silence_count = 0
                    if not self._speech_active:
                        self._speech_active = True
                else:
                    self._silence_count += 1

                # 3. ASR 推理（每帧都喂）
                if self._asr:
                    partial = self._asr.push(chunk)
                    if partial and partial != self._current_partial:
                        self._current_partial = partial
                        if self.on_transcript:
                            self.on_transcript(partial, is_partial=True)

                # 4. 静音超时自动停止
                if self._speech_active and self._silence_count > SilenceFrames:
                    self._running = False
                    break

            # Flush 最终结果
            self._flush()

        except Exception as e:
            print(f'[引擎] 错误: {e}')
            self._set_state(State.ERROR)

    def _flush(self):
        """强制输出最终结果"""
        self._set_state(State.FLUSHING)

        if self._asr:
            result = self._asr.flush()
            if result:
                self._current_final = result.text
                if self.on_transcript:
                    self.on_transcript(result.text, is_partial=False)

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

    def start(self):
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        def restore():
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        try:
            tty.setcbreak(fd)
            while True:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch == ' ':
                    self._on_space()
                elif ch in ('q', 'Q'):
                    self._on_q()
                    break
        except Exception as e:
            print(f'[键盘] 错误: {e}')
        finally:
            restore()


# ===================== Main =====================

def parse_args():
    parser = argparse.ArgumentParser(description='AI Dictation CLI')
    parser.add_argument('--model', default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper 模型大小')
    parser.add_argument('--llm-model', default='qwen/qwen2.5-72b-instruct',
                        help='LLM 模型')
    parser.add_argument('--no-polish', action='store_true',
                        help='禁用 LLM 润色')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--input-file', type=str,
                        help='从文件读取音频（用于测试，WAV 16kHz mono）')
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
        ('faster_whisper', 'faster-whisper'),
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

    print(f'AI Dictation启动中... (模型: {args.model}, LLM: {args.llm_model})')
    print('按 q 退出\n')

    # 创建引擎
    engine = DictationEngine(
        model_size=args.model,
        llm_model=args.llm_model,
        disable_polish=args.no_polish,
        use_file=args.input_file,
    )

    # 创建 UI
    ui = get_ui()

    # 回调
    def on_transcript(text, is_partial):
        engine._current_partial = text
        if is_partial:
            ui.update(State.LISTENING, partial=text)
        else:
            engine._current_final = text
            ui.update(State.FLUSHING, final=text)

    def on_polish(text):
        engine._current_final = engine._current_final or engine._current_partial
        engine._current_partial = ''
        ui.update(State.IDLE, final=engine._current_final, polished=text)

    def on_state_change(state):
        if state == State.IDLE and engine._current_final:
            engine._current_final = ''
            engine._current_partial = ''
            engine._polished = ''

    engine.on_transcript = on_transcript
    engine.on_polish = on_polish
    engine.on_state_change = on_state_change

    # 键盘处理
    listening = [False]  # 用 list 方便在闭包中修改

    def on_space():
        if engine._state == State.IDLE:
            engine.start()
            listening[0] = True
            ui.update(State.LISTENING)
        elif engine._state == State.LISTENING:
            engine.stop()
            listening[0] = False

    def on_q():
        engine.shutdown()
        ui.stop()
        print('\n👋 退出')
        sys.exit(0)

    # Ctrl+C 处理
    def on_sigint(signum, frame):
        on_q()

    signal.signal(signal.SIGINT, on_sigint)

    # 启动 UI
    ui.start()
    ui.update(State.IDLE)

    # 启动键盘处理
    kb = KeyboardHandler(on_space=on_space, on_q=on_q)

    print('[提示] 按空格开始听写，再按空格停止')
    kb.start()


if __name__ == '__main__':
    main()
