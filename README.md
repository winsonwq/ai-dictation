# AI Dictation

命令行 AI 听写工具，支持多引擎本地语音转文字。

**引擎**：whisper.cpp (Metal GPU, ~0.4s) / SenseVoice (CPU, 自带标点, ~0.8s)

## 快速开始

### 1. 安装 whisper.cpp（macOS）

```bash
brew install whisper-cpp
```

> Linux/Windows: 从 [whisper.cpp](https://github.com/ggerganov/whisper.cpp) 源码编译安装

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt

# SenseVoice 引擎需要 (Python 3.12+)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. 运行

```bash
python main.py                    # whisper small (Metal GPU, 0.4-0.5s)
python main.py --model medium     # 更高精度
.venv/bin/python main.py --engine sensevoice  # 自带标点
```

按空格开始听写，再按空格停止。

## 流式转写模式

```bash
python main.py --stream              # 启用流式转写（每 2s 刷新一次）
python main.py --stream --flush-interval 1.0  # 更快的刷新间隔
```

流式转写会在你说话过程中实时显示转写结果，而不是等到说完后才显示。适合长段演讲场景。

## 性能

| 引擎 | 模型 | 短句 (~3s) | 长句 (~15s) | 标点 |
|------|------|-----------|------------|------|
| whisper | small (487MB) | ~0.2s | ~0.5s | 无 |
| whisper | medium (1.5GB) | ~0.4s | ~1.0s | 无 |
| SenseVoice | Small (160MB) | ~0.4s | ~0.8s | ✓ |

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--engine` | ASR 引擎: `whisper` (默认) 或 `sensevoice` |
| `--model` | 模型大小: tiny/base/small/medium/large（仅 whisper） |
| `--polish` | 启用 LLM 润色（默认关闭） |
| `--input-file <file>` | 从文件读取音频（WAV 16kHz mono） |
| `--check` | 仅检查环境 |
| `--debug` | 调试模式 |
| `--stream` | 启用流式转写（边说边转写） |
| `--flush-interval <秒>` | 流式转写刷新间隔（默认 2.0） |

## 架构

```
麦克风 → Silero VAD → audio buffer (float32)
                         │
                    [按空格停止]
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
        WhisperCppBackend     SenseVoiceBackend
              │                     │
              ↓                     ↓
        whisper-worker         funasr (PyTorch)
        (C, 常驻, stdin管道)   (Python, 常驻内存)
              │                     │
              ↓                     ↓
        whisper.cpp            SenseVoiceSmall
        (Metal GPU)            (CPU)
```

## 项目结构

```
ai-dictation/
├── main.py                # CLI 入口
├── whisper-worker.c       # 微型 C 常驻进程
├── engine/
│   ├── asr.py             # ASR 入口（后端可切换）
│   ├── _asr_whisper.py    # whisper.cpp 后端
│   ├── _asr_sensevoice.py # SenseVoice 后端
│   ├── audio.py           # 音频采集
│   ├── vad.py             # Silero VAD
│   ├── stream_transcriber.py  # 流式转写组件
│   └── polish.py          # LLM 润色
├── requirements.txt
└── SPEC.md
```
