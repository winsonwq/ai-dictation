"""
engine/asr.py
ASR 模块 - faster-whisper 封装
支持流式 partial 输出和 final 输出
"""

import os
import gc
import threading
import queue
from dataclasses import dataclass, field
from typing import Iterator, Optional, Awaitable
import numpy as np

# 默认模型
DEFAULT_MODEL = 'Systran/faster-whisper-small'


@dataclass
class ASRResult:
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    language: str = 'zh'
    avg_logprob: float = 0.0


class ASREngine:
    """
    faster-whisper ASR 引擎

    支持:
    - 增量转写（partial）：实时输出中间结果
    - 最终转写（final）：完整句子结果
    - 流式推理：边收音频边输出 token

    使用方式:
        asr = ASREngine(model_size='small')
        asr.start()

        # 喂入音频块
        for chunk in audio_chunks:
            partial = asr.push_partial(chunk)
            if partial:
                print(f'实时: {partial}')

        # 获取最终结果
        final = asr.flush()
        print(f'最终: {final}')

        asr.shutdown()
    """

    def __init__(
        self,
        model_size: str = 'small',
        device: str = 'auto',  # 'cpu', 'cuda', 'auto'
        compute_type: str = 'default',  # 'int8', 'float16', 'default'
        language: str = 'zh',
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_size: whisper 模型大小 (tiny/base/small/medium/large)
            device: 运行设备 'cpu'/'cuda'/'auto'
            compute_type: 计算精度 'int8'/'float16'/'default'
            language: 语言代码，默认中文
            cache_dir: 模型缓存目录
        """
        self.model_size = model_size
        self.device = device if device != 'auto' else self._detect_device()
        self.compute_type = compute_type
        self.language = language
        self.cache_dir = cache_dir

        self._model = None
        self._sample_rate = 16000
        self._running = False
        self._lock = threading.Lock()

        # 流式推理状态
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_partial = ''

    def _detect_device(self) -> str:
        """自动检测设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'

    def _get_model_name(self) -> str:
        """获取 HuggingFace 模型名"""
        # faster-whisper 使用 HuggingFace 格式的模型
        size_to_model = {
            'tiny': 'Systran/faster-whisper-tiny',
            'base': 'Systran/faster-whisper-base',
            'small': 'Systran/faster-whisper-small',
            'medium': 'Systran/faster-whisper-medium',
            'large': 'Systran/faster-whisper-large',
        }
        return size_to_model.get(self.model_size, f'Systran/faster-whisper-{self.model_size}')

    def load(self):
        """加载模型（延迟加载，首次使用时）"""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        model_path = self._get_model_name()

        print(f'[ASR] 加载模型 {self.model_size} on {self.device}...')

        try:
            self._model = WhisperModel(
                model_size_or_path=model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as e:
            # 如果 HuggingFace 下载失败，尝试从本地缓存
            if 'not found' in str(e).lower() or '404' in str(e):
                print(f'[ASR] 模型下载失败，尝试本地缓存...')
                self._model = WhisperModel(
                    model_size_or_path=self.model_size,  # 尝试直接用 size name
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                raise

        print(f'[ASR] 模型加载完成')

    def start(self):
        """开始推理（初始化流式状态）"""
        if self._model is None:
            self.load()
        self._running = True
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_partial = ''

    def push(self, audio_chunk: np.ndarray) -> Optional[str]:
        """
        喂入一段音频，返回 partial 转写结果

        Args:
            audio_chunk: float32 array, 16kHz 单声道

        Returns:
            当前帧的 partial 转写文本，或 None
        """
        if not self._running or self._model is None:
            return None

        with self._lock:
            # 累积音频
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

            # 每 3 秒音频做一次推理（避免过于频繁）
            min_audio_for_infer = self._sample_rate * 3
            if len(self._audio_buffer) < min_audio_for_infer:
                return None

            try:
                # 流式转写
                segments, info = self._model.transcribe(
                    self._audio_buffer,
                    language=self.language,
                    beam_size=5,
                    vad_filter=False,  # 我们自己处理 VAD
                    task='transcribe',
                )

                # 取最后一个 segment 作为 partial
                partial_texts = []
                for seg in segments:
                    partial_texts.append(seg.text)

                if partial_texts:
                    self._last_partial = ''.join(partial_texts)
                    return self._last_partial

            except Exception as e:
                print(f'[ASR] 转写错误: {e}', file=__import__('sys').stderr)

            return None

    def flush(self) -> Optional[ASRResult]:
        """
        flush remaining audio and return final transcription
        应当在说话结束后调用，强制输出最终结果
        """
        if not self._running or self._model is None:
            return None

        with self._lock:
            if len(self._audio_buffer) < 1600:  # < 100ms 音频
                return None

            try:
                segments, info = self._model.transcribe(
                    self._audio_buffer,
                    language=self.language,
                    beam_size=5,
                    vad_filter=False,
                    task='transcribe',
                )

                texts = [seg.text for seg in segments]
                final_text = ''.join(texts).strip()

                if not final_text:
                    return None

                result = ASRResult(
                    text=final_text,
                    start_time=0.0,
                    end_time=len(self._audio_buffer) / self._sample_rate,
                    language=info.language if hasattr(info, 'language') else self.language,
                )

                # 清空 buffer
                self._audio_buffer = np.array([], dtype=np.float32)
                self._last_partial = ''

                return result

            except Exception as e:
                print(f'[ASR] Final 转写错误: {e}', file=__import__('sys').stderr)
                return None

    def reset(self):
        """重置流式状态（每次新的听写开始时）"""
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
            self._last_partial = ''

    def shutdown(self):
        """关闭引擎，释放资源"""
        self._running = False
        with self._lock:
            self._audio_buffer = np.array([], dtype=np.float32)
            self._model = None
            gc.collect()


class ASRStreamer:
    """
    异步 ASR 处理器
    在后台线程运行转写，通过 queue 输出结果
    """

    def __init__(self, asr_engine: ASREngine, output_queue: queue.Queue):
        self._asr = asr_engine
        self._queue = output_queue
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self, audio_source):
        """
        开始异步转写

        Args:
            audio_source: 可迭代的音频源，每次 yield 一个 chunk (np.ndarray)
        """
        self._running = True
        self._asr.start()
        self._thread = threading.Thread(
            target=self._run,
            args=(audio_source,),
            daemon=True,
        )
        self._thread.start()

    def _run(self, audio_source):
        """后台运行循环"""
        try:
            for chunk in audio_source:
                if not self._running:
                    break

                # 非阻塞检查停止信号
                try:
                    stop = self._queue.get_nowait()
                    if stop == '__STOP__':
                        break
                except queue.Empty:
                    pass

                # 推入 ASR
                partial = self._asr.push(chunk)
                if partial:
                    try:
                        self._queue.put_nowait(('partial', partial))
                    except queue.Full:
                        pass

        except Exception as e:
            print(f'[ASRStreamer] 错误: {e}')
        finally:
            self._running = False

    def flush(self) -> Optional[ASRResult]:
        """获取最终结果"""
        return self._asr.flush()

    def stop(self):
        """停止"""
        self._running = False
        try:
            self._queue.put_nowait('__STOP__')
        except queue.Full:
            pass


if __name__ == '__main__':
    print('=== ASR 模块测试 ===')

    # 生成测试音频（模拟语音）
    print('生成测试音频...')
    import math

    duration = 5.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # 模拟中文语音特征（不同频率切换）
    freqs = [200, 400, 600, 800, 1000, 1200, 1500, 1800]
    signal = np.zeros_like(t)
    chunk_len = len(t) // len(freqs)
    for i, freq in enumerate(freqs):
        start = i * chunk_len
        end = start + chunk_len
        signal[start:end] = 0.3 * np.sin(2 * np.pi * freq * t[start:end])

    # 添加一些随机性模拟真实语音
    signal += 0.05 * np.random.randn(len(signal))
    signal = np.clip(signal, -1.0, 1.0)

    # 保存测试文件
    import wave
    test_file = '/tmp/test_asr.wav'
    with wave.open(test_file, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes((signal * 32767).astype(np.int16).tobytes())
    print(f'测试音频已保存: {test_file}')

    # 测试 ASR
    print('\n初始化 ASR 引擎...')
    asr = ASREngine(model_size='small', device='cpu', compute_type='int8')
    asr.load()

    print('开始转写...')
    asr.start()

    # 分块喂入
    chunk_size = 4800  # 300ms
    partial_count = 0
    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i + chunk_size]
        partial = asr.push(chunk)
        if partial:
            print(f'  partial: {partial}')
            partial_count += 1

    print(f'\n得到 {partial_count} 次 partial 结果')

    final = asr.flush()
    if final:
        print(f'最终结果: {final.text}')
    else:
        print('最终结果为空（模型太小或音频太人造）')

    asr.shutdown()
    print('\nASR 模块测试完成')
