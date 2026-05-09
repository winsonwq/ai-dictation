"""
engine/_asr_whisper.py
Whisper.cpp backend — whisper-worker 常驻进程 (Metal GPU, 模型常驻)
"""

import os
import gc
import logging
import subprocess
import struct
import shutil
import threading
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
    """whisper.cpp ASR 后端 — whisper-worker 常驻进程"""

    def __init__(
        self,
        model_size: str = 'small',
        language: str = 'zh',
        n_threads: int = 0,
        cache_dir: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self.n_threads = n_threads
        self.cache_dir = cache_dir or self._default_cache_dir()
        self._sample_rate = 16000
        self._running = False
        self._lock = threading.Lock()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._min_audio = int(self._sample_rate * 0.5)
        self._max_audio = int(self._sample_rate * 30)
        self._worker: Optional[subprocess.Popen] = None

    def load(self):
        if self._worker is not None:
            return

        worker_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'whisper-worker')
        if not os.path.exists(worker_bin):
            raise RuntimeError(f'未找到 whisper-worker: {worker_bin}，请先编译')

        model_path = self._get_model_path()
        if not model_path.exists():
            self._download_model()

        self._worker = subprocess.Popen(
            [worker_bin, str(model_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        line = self._worker.stdout.readline().decode().strip()
        _logger.debug(f'whisper-worker 就绪: {line} ({self.model_size})')

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
        if not self._running or self._worker is None:
            return None
        import time as _time; t0 = _time.time()
        with self._lock:
            buf_len = len(self._audio_buffer)
            if buf_len < self._min_audio:
                return None
            audio_for_flush = self._audio_buffer.copy()
        t_copy = _time.time()

        _logger.debug(f'whisper flush: buffer={buf_len} (copy: {(t_copy-t0)*1000:.0f}ms)')
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
        if self._worker:
            try:
                self._worker.stdin.write(b'EXIT\n')
                self._worker.stdin.flush()
                self._worker.wait(timeout=3)
            except Exception:
                self._worker.kill()
            self._worker = None
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
        gc.collect()

    def _transcribe(self, audio: np.ndarray) -> str:
        import time as _time
        t0 = _time.time()

        try:
            audio_f32 = audio.astype(np.float32)
            cmd = f'TRANSCRIBE {self.language} {len(audio_f32)}\n'.encode()
            self._worker.stdin.write(cmd)
            self._worker.stdin.write(audio_f32.tobytes())
            self._worker.stdin.flush()
            t_send = _time.time()

            line = self._worker.stdout.readline().decode().strip()
            t_done = _time.time()

            send_ms = (t_send - t0) * 1000
            infer_ms = (t_done - t_send) * 1000
            total_ms = (t_done - t0) * 1000
            _logger.debug(f'whisper timing: send={send_ms:.0f}ms infer={infer_ms:.0f}ms total={total_ms:.0f}ms (samples={len(audio)})')

            if line.startswith('RESULT '):
                return line[7:]
            return ''
        except Exception as e:
            _logger.error(f'whisper error: {e}')
            return ''

    def _get_model_path(self) -> Path:
        filename = MODEL_FILES.get(self.model_size, f'ggml-{self.model_size}.bin')
        return Path(self.cache_dir) / filename

    @staticmethod
    def _default_cache_dir() -> str:
        return str(Path.home() / '.cache' / 'whisper-cpp-models')

    def _download_model(self):
        from huggingface_hub import hf_hub_download
        model_path = self._get_model_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        filename = model_path.name
        _logger.info(f'下载模型: {filename} ...')
        downloaded = hf_hub_download(repo_id=WHISPER_CPP_REPO, filename=filename,
                                      cache_dir=str(model_path.parent))
        if os.path.abspath(downloaded) != os.path.abspath(str(model_path)):
            if model_path.exists() or model_path.is_symlink():
                model_path.unlink()
            os.symlink(os.path.abspath(downloaded), str(model_path))
        _logger.info(f'模型已缓存: {model_path}')
