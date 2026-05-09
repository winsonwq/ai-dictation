"""
engine/_asr_whisper.py
Whisper.cpp backend — 通过 whisper-cli 子进程调用，Metal GPU 加速
"""

import os
import gc
import logging
import subprocess
import tempfile
import shutil
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from engine.asr import ASRResult

_logger = logging.getLogger('dictation.asr.whisper')

WHISPER_CPP_REPO = 'ggerganov/whisper.cpp'
MODEL_FILES = {
    'tiny':     'ggml-tiny.bin',
    'base':     'ggml-base.bin',
    'small':    'ggml-small.bin',
    'medium':   'ggml-medium.bin',
    'large':    'ggml-large-v3.bin',
    'large-v3': 'ggml-large-v3.bin',
}


class WhisperCppBackend:
    """whisper.cpp ASR 后端"""

    def __init__(
        self,
        model_size: str = 'small',
        language: str = 'zh',
        n_threads: int = 0,
        beam_size: int = 1,  # 1=greedy 最快
        cache_dir: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.n_threads = n_threads
        self.beam_size = beam_size
        self.cache_dir = cache_dir or self._default_cache_dir()

        self._sample_rate = 16000
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._min_audio = int(self._sample_rate * 0.5)
        self._max_audio = int(self._sample_rate * 30)
        self._whisper_bin: Optional[str] = None
        self._model_path: Optional[Path] = None

    # ---- 生命周期 ----

    def load(self):
        self._whisper_bin = shutil.which('whisper-cli')
        if self._whisper_bin is None:
            raise RuntimeError(
                '未找到 whisper-cli，请先安装 whisper.cpp:\n'
                '  brew install whisper-cpp'
            )
        self._model_path = self._get_model_path()
        if not self._model_path.exists():
            self._download_model()

        # 预热 Metal GPU：首次推理编译 shader，后续调用瞬间完成
        _logger.debug('whisper warm-up...')
        self._transcribe(np.zeros(16000, dtype=np.float32))  # 1s 静音
        _logger.debug(f'whisper.cpp 就绪 (模型: {self.model_size}, beam={self.beam_size})')

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
        if not self._running:
            return None
        with self._lock:
            buf_len = len(self._audio_buffer)
            if buf_len < self._min_audio:
                return None
            audio_for_flush = self._audio_buffer.copy()

        _logger.debug(f'whisper flush: buffer={buf_len}')
        text = self._transcribe(audio_for_flush)

        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)

        if not text:
            return None
        _logger.debug(f'whisper result: {text[:50]}...')
        return ASRResult(text=text, end_time=buf_len / self._sample_rate, language=self.language)

    def reset(self):
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)

    def shutdown(self):
        self._running = False
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
        gc.collect()

    # ---- 内部 ----

    def _transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        try:
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                clipped = np.clip(audio, -1.0, 1.0)
                wf.writeframes((clipped * 32767).astype(np.int16).tobytes())

            cmd = [
                self._whisper_bin,
                '-m', str(self._model_path),
                '-f', wav_path,
                '-l', self.language,
                '--no-timestamps', '-nt',
                '--no-prints',           # 抑制进度输出
                '-bs', str(self.beam_size),  # beam size
            ]
            if self.n_threads > 0:
                cmd += ['-t', str(self.n_threads)]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                _logger.error(f'whisper-cli error: {result.stderr}')
                return ''
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return ''.join(lines)
        except subprocess.TimeoutExpired:
            _logger.error('whisper-cli timeout')
            return ''
        except Exception as e:
            _logger.error(f'transcribe error: {e}')
            return ''
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _get_model_path(self) -> Path:
        filename = MODEL_FILES.get(self.model_size, f'ggml-{self.model_size}.bin')
        return Path(self.cache_dir) / filename

    @staticmethod
    def _default_cache_dir() -> str:
        return str(Path.home() / '.cache' / 'whisper-cpp-models')

    def _download_model(self):
        from huggingface_hub import hf_hub_download
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        filename = self._model_path.name
        _logger.info(f'下载模型: {filename} ...')
        downloaded = hf_hub_download(repo_id=WHISPER_CPP_REPO, filename=filename,
                                      cache_dir=str(self._model_path.parent))
        if os.path.abspath(downloaded) != os.path.abspath(str(self._model_path)):
            if self._model_path.exists() or self._model_path.is_symlink():
                self._model_path.unlink()
            os.symlink(os.path.abspath(downloaded), str(self._model_path))
        _logger.info(f'模型已缓存: {self._model_path}')
