"""
voxa - AI 听写引擎

pip install voxa-core

使用方式:
    from voxa import VoxaCore

    core = VoxaCore()
    core.on('asr.final', lambda r: print(f'转写: {r.text}'))
    core.start()
"""

from .core import VoxaCore, State
from .events import EventEmitter
from .engine import (
    MicrophoneCapture,
    FileAudioSource,
    list_devices,
    has_microphone,
    VADEngine,
    VADResult,
    ASREngine,
    ASRResult,
    PolishEngine,
)

__version__ = '0.1.0'
__all__ = [
    'VoxaCore',
    'State',
    'EventEmitter',
    'MicrophoneCapture',
    'FileAudioSource',
    'list_devices',
    'has_microphone',
    'VADEngine',
    'VADResult',
    'ASREngine',
    'ASRResult',
    'PolishEngine',
]
