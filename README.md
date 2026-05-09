# AI Dictation

命令行 AI 听写工具，支持实时语音转文字 + LLM 润色。

## 快速开始

### 0. 安装 whisper.cpp

```bash
brew install whisper-cpp
```

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 配置代理（可选，WSL2 环境需要）

```bash
export https_proxy="http://172.22.176.1:7890"
export http_proxy="http://172.22.176.1:7890"
```

### 3. 设置 OpenRouter API Key

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### 4. 运行

```bash
python main.py
```

按空格开始听写，再按空格停止。自动润色输出。

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--engine` | ASR 引擎: `whisper` (whisper.cpp) 或 `sensevoice` (SenseVoice, 带标点) |
| `--model` | Whisper 模型大小（仅 whisper 引擎） |
| `--llm-model` | LLM 模型，默认 `qwen/qwen3.5-plus-02-15` |
| `--polish` | 启用 LLM 润色（默认关闭） |
| `--input-file <file>` | 从文件读取音频（用于测试，WAV 16kHz mono） |
| `--check` | 仅检查环境 |
| `--debug` | 调试模式 |

## WSL2 环境说明

WSL2 没有直接访问 Windows 麦克风的能力。测试方式：

```bash
# 生成测试音频
python -c "
import numpy as np, wave
sr = 16000
t = np.linspace(0, 3, sr*3, dtype=np.float32)
signal = 0.3 * np.sin(2*np.pi*250*t) + 0.1*np.random.randn(len(t))
signal = np.clip(signal, -1, 1)
with wave.open('/tmp/test.wav', 'wb') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
    f.writeframes((signal*32767).astype(np.int16).tobytes())
"

# 使用文件模式运行
python main.py --input-file /tmp/test.wav
```

## 项目结构

```
ai-dictation/
├── main.py              # CLI 入口
├── engine/
│   ├── audio.py         # 音频采集
│   ├── vad.py           # Silero VAD
│   ├── asr.py           # ASR 入口（后端可切换）
│   ├── _asr_whisper.py   # whisper.cpp 后端
│   ├── _asr_sensevoice.py # SenseVoice 后端
│   ├── polish.py        # LLM 润色
│   └── rpc.py           # JSON-RPC 协议
├── logs/                # 转写历史
├── requirements.txt
└── SPEC.md              # 设计规格
```

## 模块自测

```bash
# VAD 模块
python engine/vad.py

# ASR 模块（使用缓存的 base 模型）
python engine/asr.py

# Polish 模块
python engine/polish.py

# 环境检测
python main.py --check
```
