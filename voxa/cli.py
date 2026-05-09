"""
voxa/cli.py
Voxa CLI - 命令行接口

使用方式:
    python -m voxa.cli --help
    python -m voxa.cli --file test.wav
    python -m voxa.cli --stream --file test.wav
"""

import sys
import os
import argparse
import logging

from voxa import VoxaCore

_logger = logging.getLogger('voxa.cli')


def main():
    parser = argparse.ArgumentParser(description='Voxa - AI 听写引擎')
    parser.add_argument('--file', help='音频文件路径（代替麦克风）')
    parser.add_argument('--stream', action='store_true', help='流式转写')
    parser.add_argument('--engine', default='whisper', choices=['whisper', 'sensevoice'])
    parser.add_argument('--model', default='small', help='模型大小')
    parser.add_argument('--no-polish', action='store_true', help='禁用润色')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    core = VoxaCore(
        engine=args.engine,
        model_size=args.model,
        disable_polish=args.no_polish,
        use_file=args.file,
        streaming=args.stream,
    )

    # 事件回调
    def on_state_change(state):
        print(f'[{state.value}]')

    def on_partial(text):
        print(f'\r转写中: {text}', end='', flush=True)

    def on_final(result):
        print(f'\r最终转写: {result.text}')

    def on_polish(text):
        print(f'✨ 润色: {text}')

    def on_error(msg):
        print(f'\n❌ 错误: {msg}', file=sys.stderr)

    core.on('state_change', on_state_change)
    core.on('asr.partial', on_partial)
    core.on('asr.final', on_final)
    core.on('polish.result', on_polish)
    core.on('error', on_error)

    if args.file:
        print(f'📁 使用音频文件: {args.file}')
        if not os.path.exists(args.file):
            print(f'❌ 文件不存在: {args.file}')
            sys.exit(1)
    else:
        print('🎤 使用麦克风录音')

    print('按 Ctrl+C 退出\n')

    try:
        core.start()
        # 等待处理完成（文件模式）或保持运行（麦克风模式）
        import time
        while core.state not in ('idle', 'error'):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\n⏹  停止听写...')
        result = core.stop()
        if result:
            print(f'\n📝 最终结果: {result}')
        sys.exit(0)

    result = core.stop()
    if result:
        print(f'\n📝 最终结果: {result}')


if __name__ == '__main__':
    main()
