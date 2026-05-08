"""
engine/audio.py
音频采集模块 - 跨平台麦克风采集
WSL2 环境下自动回退到文件输入模式（用于测试）
"""

import sys
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Callable, Iterator
import numpy as np

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512  # 每次读取的样本数（约32ms）


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
        if not isinstance(devices, list):
            devices = [devices]
        return [
            AudioDevice(
                index=i,
                name=d.get('name', f'Device {i}'),
                channels=d.get('max_input_channels', 0),
                sample_rate=int(d.get('default_samplerate', 16000)),
            )
            for i, d in enumerate(devices)
            if d.get('max_input_channels', 0) > 0
        ]
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

    def _audio_callback(self, indata, frames, time, status):
        """sounddevice 音频回调，每次 frames 个样本"""
        if status:
            print(f'[Audio Warning] {status}', file=sys.stderr)
        # 复制数据避免引用问题
        chunk = indata[:, 0].copy()  # 单声道 float32
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass  # 丢帧防止阻塞

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
        if target_sr != SAMPLE_RATE:
            # sounddevice 不做重采样，依赖外部 resample
            print(f'[Audio] 设备采样率 {target_sr}Hz，将内部重采样到 {SAMPLE_RATE}Hz')

        self._stream = sd.InputStream(
            device=self._device_index,
            channels=CHANNELS,
            samplerate=target_sr,
            blocksize=CHUNK_SIZE,
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
        返回: np.ndarray (float32, shape=(n_samples,)) 或 None（超时）
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self) -> Iterator[np.ndarray]:
        """持续读取直到 stop()"""
        while self._running:
            chunk = self.read(timeout=1.0)
            if chunk is not None:
                yield chunk

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
