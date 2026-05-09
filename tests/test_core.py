"""
tests/test_core.py
VoxaCore 功能测试

运行方式:
    python -m tests.test_core
"""

import sys
import os
import time
import tempfile

# 确保 voxa 包可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voxa import VoxaCore


def generate_test_audio(duration=3.0, frequency=200):
    """生成测试音频（正弦波）"""
    import numpy as np

    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    signal = np.clip(signal, -1.0, 1.0)
    return signal


def save_wav(filename, audio_data, sample_rate=16000):
    """保存为 WAV 文件"""
    import numpy as np
    import wave

    int16_data = (audio_data * 32767).astype('<i2')
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


class TestResult:
    def __init__(self):
        self.events = []
        self.final_text = None
        self.polished_text = None
        self.error = None

    def on_state_change(self, state):
        self.events.append(('state_change', state.value))

    def on_partial(self, text):
        self.events.append(('partial', text))

    def on_final(self, result):
        self.events.append(('final', result.text))
        self.final_text = result.text

    def on_polish(self, text):
        self.events.append(('polish', text))
        self.polished_text = text

    def on_error(self, msg):
        self.events.append(('error', msg))
        self.error = msg


def test_core_basic():
    """测试：基本听写流程（使用生成的测试音频）"""
    print('=' * 50)
    print('测试 1: 基本听写流程')
    print('=' * 50)

    # 生成测试音频并保存
    audio = generate_test_audio(duration=2.0, frequency=300)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    save_wav(temp_file, audio)
    print(f'生成测试音频: {temp_file} ({len(audio)/16000:.1f}s)')

    result = TestResult()
    core = VoxaCore(use_file=temp_file, disable_polish=True)

    core.on('state_change', result.on_state_change)
    core.on('asr.partial', result.on_partial)
    core.on('asr.final', result.on_final)
    core.on('error', result.on_error)

    print('启动听写...')
    core.start()

    # 等待完成
    for _ in range(50):  # 最多 25 秒
        if core.state.value == 'idle':
            break
        time.sleep(0.5)

    core.stop()
    os.unlink(temp_file)

    print(f'\n事件序列:')
    for event in result.events:
        print(f'  {event}')

    if result.final_text:
        print(f'\n✅ 转写成功: "{result.final_text}"')
    else:
        print(f'\n⚠️  无转写结果（测试音频可能太简单）')

    return result


def test_core_with_vad():
    """测试：VAD 语音活动检测"""
    print('\n' + '=' * 50)
    print('测试 2: VAD 语音活动检测')
    print('=' * 50)

    audio = generate_test_audio(duration=3.0, frequency=200)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    save_wav(temp_file, audio)

    result = TestResult()
    core = VoxaCore(use_file=temp_file, disable_polish=True)

    core.on('state_change', result.on_state_change)
    core.on('asr.final', result.on_final)

    print('启动听写（带 VAD）...')
    core.start()

    for _ in range(50):
        if core.state.value == 'idle':
            break
        time.sleep(0.5)

    core.stop()
    os.unlink(temp_file)

    print(f'\n事件序列:')
    for event in result.events:
        print(f'  {event}')

    return result


def test_core_polish():
    """测试：LLM 润色功能"""
    print('\n' + '=' * 50)
    print('测试 3: LLM 润色功能')
    print('=' * 50)

    # 读取 OPENROUTER_API_KEY
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print('⚠️  未设置 OPENROUTER_API_KEY，跳过润色测试')
        return None

    audio = generate_test_audio(duration=2.0, frequency=250)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    save_wav(temp_file, audio)

    result = TestResult()
    core = VoxaCore(use_file=temp_file, disable_polish=False)

    core.on('state_change', result.on_state_change)
    core.on('asr.final', result.on_final)
    core.on('polish.result', result.on_polish)

    print('启动听写（带润色）...')
    core.start()

    for _ in range(100):  # 润色可能需要更长时间
        if core.state.value == 'idle':
            break
        time.sleep(0.5)

    core.stop()
    os.unlink(temp_file)

    print(f'\n事件序列:')
    for event in result.events:
        print(f'  {event}')

    if result.polished_text:
        print(f'\n✅ 润色成功: "{result.polished_text}"')
    else:
        print(f'\n⚠️  无润色结果')

    return result


def test_event_emitter():
    """测试：事件发射器"""
    print('\n' + '=' * 50)
    print('测试 4: 事件发射器')
    print('=' * 50)

    from voxa import EventEmitter

    emitter = EventEmitter()
    events = []

    def handler1(data):
        events.append(('handler1', data))

    def handler2(data):
        events.append(('handler2', data))

    emitter.on('test', handler1)
    emitter.on('test', handler2)
    emitter.emit('test', 'hello')

    print(f'触发事件后收到: {events}')
    assert len(events) == 2
    assert events[0] == ('handler1', 'hello')
    assert events[1] == ('handler2', 'hello')
    print('✅ 事件发射器正常')

    # 测试 once
    emitter.once('once', handler1)
    emitter.emit('once', 'first')
    emitter.emit('once', 'second')
    once_events = [e for e in events if e[0] == 'handler1' and 'once' in str(e)]
    print(f'once 事件触发次数: {len(once_events)}（应为 1）')
    assert len(once_events) == 1
    print('✅ once 正常工作')

    return True


def test_imports():
    """测试：包导入"""
    print('\n' + '=' * 50)
    print('测试 5: 包导入')
    print('=' * 50)

    from voxa import VoxaCore, State, EventEmitter
    from voxa.engine import MicrophoneCapture, VADEngine, ASREngine, PolishEngine

    print('✅ VoxaCore, State, EventEmitter 导入成功')
    print('✅ 引擎模块导入成功')

    return True


def main():
    print('VoxaCore 测试套件')
    print()

    test_imports()
    test_event_emitter()
    test_core_basic()
    test_core_with_vad()
    test_core_polish()

    print('\n' + '=' * 50)
    print('测试完成')
    print('=' * 50)


if __name__ == '__main__':
    main()
