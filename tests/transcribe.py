#!/usr/bin/env python3
"""
tests/transcribe.py
快速转写测试脚本

使用方式:
    python tests/transcribe.py --file audio.wav
    python tests/transcribe.py --file audio.wav --polish
    python tests/transcribe.py --file audio.wav --engine sensevoice
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voxa import VoxaCore


def transcribe(filepath, engine='whisper', model='small', polish=True, streaming=False):
    """转写音频文件并返回结果"""

    print(f'📁 文件: {filepath}')
    print(f'🎙️  引擎: {engine}/{model}')
    print(f'✨ 润色: {"开启" if polish else "关闭"}')
    print()

    result = {'final': None, 'polished': None, 'error': None}
    polish_done = False

    core = VoxaCore(
        engine=engine,
        model_size=model,
        use_file=filepath,
        disable_polish=not polish,
        streaming=streaming,
    )

    def on_state_change(state):
        print(f'[{state.value}]')

    def on_partial(text):
        print(f'\r转写: {text[:50]}...' if len(text) > 50 else f'\r转写: {text}', end='', flush=True)

    def on_final(asr_result):
        print(f'\r✅ 转写完成: {asr_result.text}')
        result['final'] = asr_result.text

    def on_polish(text):
        nonlocal polish_done
        print(f'✨ 润色完成: {text}')
        result['polished'] = text
        polish_done = True

    def on_error(msg):
        print(f'\n❌ 错误: {msg}')
        result['error'] = msg

    core.on('state_change', on_state_change)
    core.on('asr.partial', on_partial)
    core.on('asr.final', on_final)
    core.on('polish.result', on_polish)
    core.on('error', on_error)

    core.start()

    # 等待完成
    start = time.time()
    while core.state.value not in ('idle', 'error'):
        time.sleep(0.5)
        if time.time() - start > 120:  # 2 分钟超时
            print('\n⏰ 超时')
            core.stop()
            break

    if core.state.value != 'error':
        core.stop()

    print()
    return result


def main():
    parser = argparse.ArgumentParser(description='Voxa 转写测试')
    parser.add_argument('--file', required=True, help='音频文件路径 (.wav)')
    parser.add_argument('--engine', default='whisper', choices=['whisper', 'sensevoice'])
    parser.add_argument('--model', default='small', help='模型大小')
    parser.add_argument('--no-polish', action='store_true', help='禁用润色')
    parser.add_argument('--stream', action='store_true', help='流式转写')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'❌ 文件不存在: {args.file}')
        sys.exit(1)

    result = transcribe(
        filepath=args.file,
        engine=args.engine,
        model=args.model,
        polish=not args.no_polish,
        streaming=args.stream,
    )

    if result['error']:
        sys.exit(1)


if __name__ == '__main__':
    main()
