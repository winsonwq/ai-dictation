# Voxa

事件驱动的 AI 听写引擎，支持多引擎本地语音转文字。

**特点**：
- 🎯 零依赖 Python 包（`pip install voxa-core`）
- 🔧 双引擎：whisper.cpp (GPU) / SenseVoice ONNX (CPU)
- 🌊 流式转写 + HTTP API
- 📦 无 PyTorch！使用 ONNX Runtime

## 快速开始

### 安装

```bash
pip install voxa-core
```

### Python API

```python
from voxa import VoxaCore

core = VoxaCore(engine='sensevoice')
core.on('asr.final', lambda text: print(f"转写: {text}"))
core.start()
```

### HTTP API

```bash
pip install voxa-core
voxa serve  # 启动 HTTP 服务器

# 转写音频
curl -X POST http://localhost:8765/api/transcribe \
  -F "audio=@recording.wav"
```

### CLI

```bash
voxa --engine sensevoice  # 开始听写（按 Ctrl+C 停止）
```

## 引擎

| 引擎 | 加速 | 模型大小 | 标点 | 依赖 |
|------|------|---------|------|------|
| whisper.cpp | GPU (Metal/CUDA) | ~140MB (small) | ❌ | libwhisper |
| SenseVoice | CPU (ONNX Runtime) | ~230MB (int8) | ✅ | onnxruntime |

## 项目结构

```
voxa/
├── voxa/
│   ├── __init__.py      # 导出 VoxaCore, events
│   ├── core.py          # VoxaCore 事件驱动引擎
│   ├── events.py        # EventEmitter
│   ├── server.py        # HTTP 服务器
│   ├── cli.py           # CLI 入口
│   └── engine/
│       ├── asr.py       # ASR 引擎抽象
│       ├── _asr_whisper.py    # whisper.cpp 后端
│       ├── _asr_sensevoice.py # SenseVoice ONNX 后端
│       └── vad.py       # FSMN VAD (ONNX)
├── tests/
│   ├── test_core.py     # 核心测试
│   └── test_http.py     # HTTP API 测试
├── pyproject.toml
└── requirements.txt
```

## 架构

```
用户音频
    ↓
┌─────────────────────────────────────┐
│           VoxaCore                  │
│  (事件驱动引擎，EventEmitter)        │
└─────────────────────────────────────┘
    ↓
┌─────────────┐    ┌─────────────────┐
│   VAD       │ →  │   ASR Engine    │
│  FSMN ONNX  │    │ (whisper/Sense) │
└─────────────┘    └─────────────────┘
    ↓
事件: vad.speech_start, vad.speech_end,
     asr.partial, asr.final, error
```

## 安装依赖

### macOS

```bash
brew install whisper-cpp
pip install voxa-core
```

### Linux

```bash
# 安装 whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && mkdir build && cd build
cmake .. && make -j && sudo make install

pip install voxa-core
```

### Windows

```powershell
# 安装 whisper.cpp (需要 MSYS2 或 WSL)
# 或者使用 WSL

pip install voxa-core
```

## 开发

```bash
git clone https://github.com/winsonwq/voxa
cd voxa
pip install -e .

# 运行测试
python -m pytest tests/

# 启动 HTTP 服务器
python -m voxa serve
```

## 许可证

MIT
