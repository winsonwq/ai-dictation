# SPEC: ai-dictation

## Objective

**What:** 一个命令行优先的 AI 听写工具，通过麦克风实时采集语音，实时转写为文字，并经 LLM 智能纠错/润色后输出。

**Why:** 技术验证阶段 — 验证 faster-whisper + LLM polishing 的端到端体验，确认延迟和体验达标后再考虑 UI 层（Tauri）封装。

**User:** 阿秋（技术用户，命令行熟练）

**Success Criteria:**
- [ ] 启动后按空格开始听写，再按空格停止
- [ ] 说话结束后 1.5s 内显示转写结果
- [ ] LLM 纠错结果在转写后 2s 内显示
- [ ] 全程无崩溃，异常有清晰报错
- [ ] Ctrl+C 干净退出

---

## Tech Stack

| Layer | Technology | Version/Notes |
|-------|-----------|---------------|
| Audio Capture | sounddevice | `sounddevice` pip 包，跨平台音频采集 |
| VAD | Silero VAD | `silero-vad` pip 包，轻量流式 VAD |
| ASR | faster-whisper | CTranslate2 INT8 量化版，中文模型 |
| LLM Polish | OpenRouter API | 统一接口，支持任意模型，默认 qwen/qwen3.5-plus-02-15 |
| CLI UI | Rich | `rich` 库，终端彩色实时输出 |
| IPC | stdin/stdout JSON-RPC | TTY 检测，自动选择管道模式 |

**Python:** 3.10+

---

## Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 Whisper 模型（首次运行自动下载）
python -m faster_whisper import 2025  # 可选：指定模型大小

# 运行（默认使用 faster-whisper small + Silero VAD）
python main.py

# 指定模型大小
python main.py --model base  # tiny/base/small/medium/large

# 指定 LLM 模型
python main.py --llm-model qwen/qwen3.5-plus-02-15

# 禁用 LLM polish（只看原始转写）
python main.py --no-polish

# 调试模式（显示原始音频幅度、VAD 状态）
python main.py --debug
```

---

## Project Structure

```
ai-dictation/
├── main.py                 # CLI 入口，参数解析，Rich UI 驱动
├── engine/
│   ├── __init__.py
│   ├── audio.py            # 音频采集（sounddevice）
│   ├── vad.py              # Silero VAD 封装
│   ├── asr.py              # faster-whisper 封装，流式推理
│   ├── polish.py           # LLM polishing（OpenRouter）
│   └── rpc.py              # stdin/stdout JSON-RPC 协议
├── models/                 # Whisper 模型缓存目录
├── logs/                   # 转写历史日志
├── requirements.txt
├── SPEC.md
└── README.md
```

---

## Code Style

### 模块职责（单一出口）

```python
# engine/vad.py
class VADEngine:
    """Silero VAD 封装，流式接口，返回 (is_speech: bool, prob: float)"""
    def push(self, audio_chunk: np.ndarray) -> Tuple[bool, float]: ...
    def reset(self) -> None: ...

# engine/asr.py
class ASREngine:
    """faster-whisper 封装，支持流式增量输出"""
    def start(self) -> None: ...
    def push(self, audio: np.ndarray) -> Iterator[str]: ...  # yields partial transcripts
    def stop(self) -> str: ...  # final transcript
    def shutdown(self) -> None: ...

# engine/polish.py
class PolishEngine:
    """LLM 文本纠错/润色"""
    def polish(self, text: str) -> str: ...  # blocking
    def polish_async(self, text: str) -> Awaitable[str]: ...  # non-blocking
```

### 输出格式约定

```
[12:34:56] 🎤 开始听写... (按空格停止)
[12:35:02] 📝 转写: 今天天气不错
[12:35:03] ✨ 润色: 今天天气不错。
[12:35:08] 🎤 开始听写... (按空格停止)
```

### 异常处理

- 所有子模块初始化失败必须抛 `DictationError(msg: str, cause: Exception | None)`
- `main.py` 顶层 catch 并打印友好错误，不堆栈

---

## Data Flow

```
[麦克风 PCM 16kHz mono]
    ↓ sounddevice stream
[音频缓冲 (ring buffer, 30s)]
    ↓
[Silero VAD] ─── 检测到语音活动 ───→ 触发 ASR
    ↓                                         ↓
[无活动超时]                              [faster-whisper 流式推理]
    ↓                                         ↓
[自动 flush 触发停止]                     [输出 partial transcript]
                                                  ↓
                                           [LLM polish 异步]
                                                  ↓
                                           [输出 polished text]
```

---

## State Machine

```
IDLE → (空格按下) → LISTENING → (空格按下/超时) → POLISHING → IDLE
                         ↓
                   (检测到静音<2s) → FLUSHING → POLISHING
```

| State | 显示 | 行为 |
|-------|------|------|
| IDLE | `🎤 按空格开始` | 等待 VAD/空格 |
| LISTENING | `🎙️ 听写中...` | 采集 + VAD + ASR 流式 |
| FLUSHING | `⏳ 处理中...` | 等待 ASR 输出 final |
| POLISHING | `✨ 润色中...` | LLM polish |
| ERROR | `❌ 错误信息` | 等待按键清除 |

---

## LLM Polish Prompt（系统提示词）

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
- 模型文件缓存到 `~/.cache/huggingface/` 或项目 `models/` 目录

**Ask first:**
- 修改 LLM API 接口（目前用 OpenRouter）
- 增加新依赖（需更新 requirements.txt）
- 修改 VAD/ASR 核心逻辑

**Never:**
- 不保存音频原始文件（隐私）
- 不打印 API key 或 token
- 不在非 TTY 环境下产生非 JSON 输出（stdout 重定向时自动切换为 JSON 模式）

---

## Success Criteria（验证步骤）

```bash
# 1. 环境检测
python main.py --check  # 检测麦克风、依赖、模型下载状态

# 2. 基本转写测试（说一句话，等待自动停止）
python main.py  # 预期：3s 内出转写结果

# 3. LLM polish 测试
python main.py  # 预期：转写后 2s 内出润色结果

# 4. 多次会话测试
python main.py  # 连续说 3 段话，验证状态机恢复

# 5. 错误恢复测试
python main.py  # 拔掉麦克风，验证报错
```

---

## Open Questions

1. faster-whisper 的流式输出（streaming mode）是否足够稳定？还是用 chunk-by-chunk 更可靠？
2. LLM polish 是否需要区分"实时 partial"和"最终文本"两种处理策略？
3. 是否需要支持中文+英文混合场景的自动检测？

---

*Last updated: 2026-05-08 by 百万*
