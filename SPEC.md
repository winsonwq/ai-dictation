# SPEC: ai-dictation

## Objective

**What:** 一个命令行优先的 AI 听写工具，通过麦克风实时采集语音，实时转写为文字，并经 LLM 智能纠错/润色后输出。

**User:** 阿秋（技术用户，命令行熟练）

**Status:** v1.0 功能完成，已支持流式转写

---

## Success Criteria

| 需求 | 状态 | 备注 |
|------|------|------|
| 启动后按空格开始听写，再按空格停止 | ✅ 完成 | |
| 说话结束后显示转写结果 | ✅ 完成 | Whisper ~1s, SenseVoice ~1.2s |
| LLM 润色结果显示 | ✅ 完成 | 默认关闭，需 `--polish` |
| 全程无崩溃，异常有清晰报错 | ✅ 完成 | |
| Ctrl+C 干净退出 | ✅ 完成 | |
| 流式转写（边说边转） | ✅ 完成 | 需 `--stream` 参数 |
| SenseVoice 引擎 | ✅ 完成 | 需 `--engine sensevoice` |
| 自动静音检测停止 | ❌ 未实现 | 目前需手动按空格停止 |
| 中英文混合场景优化 | ❌ 未实现 | |

---

## Tech Stack

| Layer | Technology | Status | Notes |
|-------|-----------|--------|-------|
| Audio Capture | sounddevice | ✅ | 16kHz mono 采集 |
| VAD | Silero VAD | ✅ | 语音活动检测 |
| ASR | faster-whisper | ✅ | CTranslate2 INT8 量化 |
| ASR | SenseVoice | ✅ | 阿里 FunASR，原生中文标点 |
| LLM Polish | OpenRouter API | ✅ | 默认 qwen/qwen3.5-plus-02-15 |
| CLI UI | Rich | ✅ | 终端彩色实时输出 |
| Streaming | StreamTranscriber | ✅ | 定时批量转写组件 |
| IPC | stdin/stdout JSON-RPC | ❌ 未实现 | TTY 检测未做 |

**Python:** 3.10+

---

## Commands

```bash
# 基本用法
python main.py                              # Whisper small 模型
python main.py --model base                 # 指定模型大小
python main.py --engine sensevoice          # 使用 SenseVoice 引擎（原生标点）

# 流式转写
python main.py --stream                     # 启用流式转写（每 2s 刷新）
python main.py --stream --flush-interval 1.0 # 更快的刷新间隔

# LLM 润色
python main.py --polish                      # 启用 LLM 润色

# 其他
python main.py --check                       # 环境检测
python main.py --debug                       # 调试模式
python main.py --no-polish                   # 禁用润色
python main.py --llm-model qwen/qwen3.5-plus-02-15  # 指定 LLM 模型
```

---

## Project Structure

```
ai-dictation/
├── main.py                     # CLI 入口，状态机，Rich UI
├── engine/
│   ├── audio.py               # 音频采集（sounddevice）
│   ├── vad.py                 # Silero VAD 封装
│   ├── asr.py                 # ASR 引擎统一接口
│   ├── _asr_whisper.py        # Whisper.cpp 后端
│   ├── _asr_sensevoice.py     # SenseVoice 后端
│   ├── stream_transcriber.py  # 流式转写组件
│   └── polish.py              # LLM polishing（OpenRouter）
├── docs/                       # 设计文档
├── requirements.txt
├── SPEC.md
└── README.md
```

---

## State Machine

```
                    ┌─[--stream]──────────────┐
                    ↓                          ↓
IDLE → (空格) → LISTENING → (空格) → FLUSHING → POLISHING → IDLE
                       ↓
              [检测到静音<2s] → FLUSHING

STREAMING → (空格) → FLUSHING → POLISHING → IDLE
```

| State | 显示 | 行为 |
|-------|------|------|
| IDLE | `🎤 按空格开始` | 等待空格 |
| LISTENING | `🎙️ 听写中...` | 采集 + VAD + ASR |
| STREAMING | `📝 转写中...` | 采集 + 流式转写（每 2s 刷新） |
| FLUSHING | `⏳ 转写中...` | 等待 ASR 最终结果 |
| POLISHING | `✨ 润色中...` | LLM polish |
| ERROR | `❌ 错误` | 等待按键清除 |

---

## Data Flow

### 非流式模式
```
[麦克风 16kHz mono]
    ↓
[Silero VAD] ─── 语音活动 ───→ [ASR 累积音频]
    ↓
[按空格] → [ASR flush] → [转写结果] → [LLM polish]
```

### 流式模式 (--stream)
```
[麦克风 16kHz mono]
    ↓
[Silero VAD] ─── 语音活动 ───→ [StreamTranscriber 累积音频]
    ↓
[定时器 2s] → [ASR flush] → [流式显示]
    ↓
[按空格] → [flush_final] → [最终结果]
```

---

## LLM Polish Prompt

```
你是一个专业的语音转写后处理器。你的任务：
1. 修正明显的语音识别错误
2. 添加适当的标点符号（，。！？：；""）
3. 适当分段（超过40字考虑分句）
4. 保持原意，不添加额外内容
5. 不改变人名、地名、术语

输入：语音转写的原始文本（可能没有标点）
输出：处理后的文本（直接输出，不要解释）
```

---

## Boundaries

**Always:**
- 启动时检测麦克风，无麦克风时报错退出
- 按 `Ctrl+C` 或 `q` 干净退出（停止所有线程，释放音频设备）
- 模型文件缓存到 `~/.cache/huggingface/`

**Never:**
- 不保存音频原始文件（隐私）
- 不打印 API key 或 token

---

## Open Questions (已关闭)

| # | 问题 | 结论 |
|---|------|------|
| 1 | faster-whisper 流式输出是否稳定？ | ✅ chunk-by-chunk 更可靠，已实现 StreamTranscriber |
| 2 | LLM polish 是否区分 partial/final？ | ❌ 简化处理，统一用 final 结果润色 |
| 3 | 中文+英文混合场景优化？ | ❌ 未实现 |

---

## Roadmap

- [ ] 自动静音检测停止（无需手动按空格）
- [ ] Tauri GUI 封装
- [ ] 中英文混合场景优化
- [ ] 历史记录导出
- [ ] 多语言支持

---

*Last updated: 2026-05-09*
