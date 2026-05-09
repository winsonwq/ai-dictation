"""
engine/audio.py
音频采集模块 - 跨平台麦克风采集
WSL2 环境下自动回退到文件输入模式（用于测试）
"""

import sys
import logging
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Callable, Iterator
import numpy as np

_logger = logging.getLogger('dictation.audio')

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512  # 每次读取的样本数（32ms @ 16kHz，与 Silero VAD 要求一致）


class AudioError(Exception):
    """音频相关错误"""
    pass


@dataclass
class AudioDevice:
    index: int
    name: str
    channels: int
    sample_rate: int


def list_devices() -> list[AudioDevice]:
    """列出所有可用音频输入设备"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        # query_devices() 返回 DeviceList，支持索引和属性访问
        result = []
        for i, d in enumerate(devices):
            # 支持属性和字典两种访问方式
            max_channels = getattr(d, 'max_input_channels', 0)
            if isinstance(d, dict):
                max_channels = d.get('max_input_channels', 0)
            if max_channels > 0:
                name = getattr(d, 'name', f'Device {i}')
                sample_rate = int(getattr(d, 'default_samplerate', 16000))
                if isinstance(d, dict):
                    name = d.get('name', f'Device {i}')
                    sample_rate = int(d.get('default_samplerate', 16000))
                result.append(AudioDevice(
                    index=i,
                    name=name,
                    channels=max_channels,
                    sample_rate=sample_rate,
                ))
        return result
    except OSError as e:
        if 'PortAudio' in str(e) or 'No Default Input Device' in str(e):
            return []
        raise AudioError(f'无法查询音频设备: {e}') from e


def has_microphone() -> bool:
    """检测是否有可用麦克风"""
    return len(list_devices()) > 0


class MicrophoneCapture:
    """
    麦克风音频采集器
    使用 sounddevice 流式采集，输出 16kHz 单声道 PCM
    """

    def __init__(self, device_index: Optional[int] = None):
        self._stream = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._device_index = device_index
        self._device_sr = SAMPLE_RATE  # 设备原始采样率

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice 音频回调，每次 frames 个样本"""
        if status:
            _logger.warning(f'音频回调警告: {status}')
        chunk = indata[:, 0].copy()  # 单声道 float32

        # 诊断：记录原始音频强度
        rms_original = np.sqrt(np.mean(chunk ** 2)).item()
        max_original = np.max(np.abs(chunk)).item()

        # 重采样到 16kHz
        if self._device_sr != SAMPLE_RATE:
            chunk = self._resample(chunk, self._device_sr, SAMPLE_RATE)

        rms_after = np.sqrt(np.mean(chunk ** 2)).item() if len(chunk) > 0 else 0
        max_after = np.max(np.abs(chunk)).item() if len(chunk) > 0 else 0

        # 只在音量变化时记录（避免日志太多）
        if not hasattr(self, '_last_rms') or abs(rms_after - self._last_rms) > 0.001:
            _logger.debug(f'音频: 原始rms={rms_original:.4f} resample后rms={rms_after:.4f} len={len(chunk)}')
            self._last_rms = rms_after

        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass  # 丢帧防止阻塞

    def _resample(self, data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """线性插值重采样"""
        if from_sr == to_sr:
            return data
        n = int(len(data) * to_sr / from_sr)
        indices = np.linspace(0, len(data) - 1, n)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    def start(self):
        """启动采集"""
        if self._running:
            return

        try:
            import sounddevice as sd
        except OSError as e:
            raise AudioError(
                f'无法加载 sounddevice (PortAudio 未安装)\n'
                f'在 WSL2 环境下请使用 --input-file 模式测试'
            ) from e

        device_info = None
        if self._device_index is not None:
            device_info = sd.query_devices(self._device_index)
        else:
            try:
                device_info = sd.query_devices(kind='input')
            except sd.PortAudioError:
                raise AudioError('未找到默认麦克风设备')

        if device_info is None or device_info.get('max_input_channels', 0) == 0:
            raise AudioError('所选设备不是输入设备')

        # 确保 16kHz 采样
        target_sr = int(device_info.get('default_samplerate', 16000))
        self._device_sr = target_sr  # 保存设备采样率

        # 计算 blocksize：确保重采样后约 512 样本（Silero VAD 要求）
        if target_sr != SAMPLE_RATE:
            computed_blocksize = int(512 * target_sr / SAMPLE_RATE)
            computed_blocksize = ((computed_blocksize + 15) // 16) * 16
            blocksize = max(computed_blocksize, 512)
        else:
            blocksize = CHUNK_SIZE

        self._stream = sd.InputStream(
            device=self._device_index,
            channels=CHANNELS,
            samplerate=target_sr,
            blocksize=blocksize,
            dtype='float32',
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """后台读取循环"""
        import sounddevice as sd
        try:
            while self._running:
                sd.sleep(int(CHUNK_SIZE * 1000 / SAMPLE_RATE))
        except Exception:
            pass

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        读取一段音频数据
        返回: np.ndarray (float32, shape=(n_samples,)) 或 None（超时或停止）
        """
        if not self._running:
            _logger.debug('read() 返回 None: _running=False')
            return None
        try:
            chunk = self._queue.get(timeout=timeout)
            _logger.debug(f'read() 获取 chunk: len={len(chunk)}, qsize={self._queue.qsize()}')
            return chunk
        except queue.Empty:
            _logger.debug('read() 超时: queue.Empty')
            return None

    def read_all(self) -> Iterator[np.ndarray]:
        """持续读取直到 stop()"""
        _logger.debug('read_all() 开始')
        while self._running:
            chunk = self.read(timeout=0.1)  # 100ms timeout，减少停止延迟
            if chunk is not None:
                yield chunk
        _logger.debug('read_all() 结束')

    def stop(self):
        """停止采集"""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class FileAudioSource:
    """
    文件音频源（用于 WSL2 测试）
    从 wav/pcm 文件读取音频数据
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._data: Optional[np.ndarray] = None
        self._pos = 0

    def load(self) -> int:
        """加载音频文件，返回样本数"""
        import wave

        with wave.open(self.filepath, 'rb') as wf:
            assert wf.getnchannels() == 1, '只支持单声道'
            assert wf.getsampwidth() == 2, '只支持 16bit'
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            int16_data = np.frombuffer(raw, dtype=np.int16)
            self._data = int16_data.astype(np.float32) / 32768.0

            # 重采样如果需要
            if sample_rate != SAMPLE_RATE:
                self._data = self._resample(self._data, sample_rate, SAMPLE_RATE)

        return len(self._data)

    @staticmethod
    def _resample(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """简单线性插值重采样"""
        if from_sr == to_sr:
            return data
        n = int(len(data) * to_sr / from_sr)
        indices = np.linspace(0, len(data) - 1, n)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    def read(self) -> Optional[np.ndarray]:
        """读取一块音频，返回 None 表示文件结束"""
        if self._data is None:
            return None
        chunk_size = SAMPLE_RATE // 10  # 每次 100ms
        if self._pos >= len(self._data):
            return None
        end = min(self._pos + chunk_size, len(self._data))
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def read_all(self) -> Iterator[np.ndarray]:
        """持续读取直到文件结束"""
        while True:
            chunk = self.read()
            if chunk is None:
                break
            yield chunk


def check_environment() -> dict:
    """检测运行环境，返回诊断信息"""
    info = {
        'has_microphone': False,
        'has_portaudio': False,
        'devices': [],
        'mode': 'file',  # 'microphone' | 'file'
    }

    try:
        import sounddevice as sd
        info['has_portaudio'] = True
        info['devices'] = [
            {'index': d.index, 'name': d.name, 'channels': d.channels}
            for d in list_devices()
        ]
        info['has_microphone'] = len(info['devices']) > 0
        info['mode'] = 'microphone' if info['has_microphone'] else 'file'
    except OSError:
        info['mode'] = 'file'

    return info


if __name__ == '__main__':
    print('=== 音频环境检测 ===')
    info = check_environment()
    print(f'运行模式: {info["mode"]}')
    print(f'PortAudio: {"✅" if info["has_portaudio"] else "❌"}')
    print(f'麦克风: {"✅" if info["has_microphone"] else "❌"}')
    if info['devices']:
        print('可用设备:')
        for d in info['devices']:
            print(f'  [{d["index"]}] {d["name"]} ({d["channels"]} channels)')
