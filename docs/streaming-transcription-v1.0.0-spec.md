# 流式实时转写功能规格 (Streaming Real-time Transcription)

## 元信息

| 字段 | 值 |
|------|-----|
| 版本 | v1.1.2 |
| 日期 | 2026/05/09 |
| 状态 | Ready for Implementation |
| 项目路径 | /Users/aqiu/projects/ai-dictation |

---

## 1. 背景

当前 ai-dictation 采用"说完再转写"模式：用户按住空格说话，松开后等待 0.4-0.5s 得到完整转写结果。这种模式适合短句，但对于长段演讲，用户无法实时看到进度，产生焦虑感。

**用户痛点**：
- 长语音转写时，不知道要等多久
- 无法确认麦克风是否在正常工作
- 打断后无法续写，需要整句重说

**目标**：实现"边说边转写"，在用户说话过程中实时显示部分转写结果，提升交互体验。

---

## 2. 目标

| 目标 | 描述 | 可衡量指标 |
|------|------|-----------|
| 流式输出 | 语音识别过程中持续输出已识别文本 | 每 flush_interval（默认 2s）更新一次显示 |
| 延迟可控 | 从语音输入到显示的延迟 | ≤ flush_interval + 转写时间（默认 ≤ 3s） |
| 引擎兼容 | whisper.cpp 和 SenseVoice 均支持流式 | 两个引擎共用同一套 StreamTranscriber |
| 体验平滑 | 转写过程有明确的视觉反馈 | 显示"正在转写..."状态 |

---

## 3. 范围

### 3.1 做 (Scope In)

- 定时批量转写：每 flush_interval 触发一次完整转写，输出增量结果
- 增量文本显示：已识别部分实时刷新显示
- 状态指示：区分"等待说话"、"正在说话"、"正在转写"三种状态
- 模式切换：可通过命令行参数选择是否启用流式模式
- 引擎兼容：whisper.cpp 和 SenseVoice 共用同一套实现

### 3.2 不做 (Scope Out)

- 不修改 VAD 逻辑（Silero VAD 保持现有实现）
- 不修改 whisper.cpp 官方 C 代码（方案 A 不需要）
- 不支持多语言混合识别（保持中文优先）
- 不修改 polishing（LLM润色）逻辑

---

## 4. 方案

### 4.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      主流程状态机                            │
├─────────────────────────────────────────────────────────────┤
│  IDLE → LISTENING → STREAMING → FLUSHING → POLISHING → IDLE │
│                    ↑                                        │
│                    └──── 流式状态 (边说边转写)                │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 实现方案对比

有两种实现思路：

#### 方案 A：定时批量转写（推荐）

**原理**：不依赖引擎原生流式 API，纯粹通过定时触发实现"边说边转写"效果。

```
[音频采集] → [缓冲区累积] → [每 2s 触发完整转写] → [输出增量结果] → [继续累积]
```

**优点**：
- 不依赖引擎 API，两个引擎实现方式完全一致
- 实现简单，复用现有批量转写接口
- 调试容易，每次都是完整的批量转写

**缺点**：
- 会有少量重复输出（前后两次转写可能有重叠文本）
- 延迟取决于定时间隔（2s 间隔 → 最差 2s 延迟）

**关键参数**：
- `flush_interval`: 定时触发间隔，建议 2s

#### 方案 B：引擎原生流式 API

依赖 whisper.cpp 的 `whisper_init_streaming()` 或 FunASR 的流式接口。

**优点**：
- 真正的增量输出，无重复

**缺点**：
- 两个引擎需要分别实现
- whisper.cpp 流式 API 需要修改 whisper-worker.c 通信协议
- SenseVoice 可能无原生流式接口

**结论**：**推荐先实现方案 A**，简单可靠，两个引擎一套代码，效果差距用户感知不明显。

### 4.3 核心变更点（方案 A：定时批量转写）

#### 4.3.1 定时批量转写实现

**核心算法**：
- 维护两个缓冲区：`_audio_buffer`（本次间隔新音频）和 `_all_audio_buffer`（所有累积音频）
- **每次 flush 都用 ALL 累积音频转写**，而非只转写最新片段
- `_audio_buffer` 在每次 flush 后清空，`_all_audio_buffer` 持续累积

```python
# stream_transcriber.py (新增)
class StreamTranscriber:
    def __init__(self, asr_engine, flush_interval=2.0):
        self.asr_engine = asr_engine
        self.flush_interval = flush_interval
        self._audio_buffer = []      # 本次间隔新音频
        self._all_audio_buffer = []  # 所有累积音频（不清空）

    def feed(self, audio_chunk: np.ndarray):
        """接收音频块，同时写入两个缓冲区"""
        self._audio_buffer.append(audio_chunk)
        self._all_audio_buffer.append(audio_chunk)

    def _do_flush(self):
        """定时触发：清空本次缓冲区，用所有累积音频转写"""
        if not self._audio_buffer:
            return

        # 清空本次缓冲区（保留所有累积音频）
        self._audio_buffer = []

        # 用 ALL 累积音频进行转写（完整上下文）
        audio = np.concatenate(self._all_audio_buffer)
        self.asr_engine.reset()
        self.asr_engine.push(audio)
        result = self.asr_engine.flush(force=True)
        # ... 处理结果
```

**关键点**：
- 无论 flush_interval 是多少，每次触发都用**所有累积音频**转写
- Whisper/SenseVoice 看到完整上下文，转写结果更准确
- 只在说话结束时才真正"停止"

**状态机改造**：
```python
class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    STREAMING = auto()      # 新增：边说边转写中，定时触发转写
    FLUSHING = auto()       # 保留：最终转写（说话结束后）
    POLISHING = auto()
```

**定时触发逻辑**：
- 方式1：使用 `threading.Timer` 每 2s 触发一次
- 方式2：复用 VAD，在 VAD 检测到静音时触发（更智能）

#### 4.3.2 UI 显示变更

```python
# main.py 状态机新增 STREAMING 状态
class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    STREAMING = auto()      # 新增：边说边转写中
    FLUSHING = auto()
    POLISHING = auto()
```

显示逻辑：
- STREAMING 状态：显示 `正在转写: <增量文本>`，同一行持续更新
- 使用 Rich 库的 `rewindable` 控制台输出，同一行刷新

### 4.4 交互流程对比

**当前模式 (说完再转写)**：
```
[说话中...] → [松开空格] → [转写中 0.5s] → [输出完整结果]
```

**流式模式 (边说边转写)**：
```
[说话中...] → [开始转写] → [输出部分1] → [输出部分2] → ... → [说完] → [最终结果]
```

### 4.5 命令行接口

```bash
# 默认行为：不加参数时等同于 --no-stream（保持向后兼容）
python main.py

# 启用流式转写
python main.py --stream

# 显式关闭流式转写
python main.py --no-stream

# 同时指定引擎和流式
python main.py --engine sensevoice --stream

# 指定流式刷新间隔（方案 A 专用）
python main.py --stream --flush-interval 2.0
```

**默认行为**：不指定 `--stream` 或 `--no-stream` 时，默认关闭流式模式（向后兼容）

### 4.6 性能目标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 首字延迟 | ≤ flush_interval + 转写时间 | 由刷新间隔决定（默认 2s + 0.5s ≈ 2.5s 最差） |
| 增量更新间隔 | = flush_interval | 定时触发间隔（默认 2s） |
| 内存占用 | ≤ +30MB | 音频缓冲（2s × 16kHz × 4字节 ≈ 128KB，额外缓冲开销小） |
| CPU 占用 | ≤ +30% | 定时转写带来的额外开销 |

---

## 5. 任务拆解

> **说明**：基于方案 A（定时批量转写），实现大幅简化，不需要调研引擎原生流式 API。

### 5.1 任务列表 (Task Breakdown)

- [ ] **T1: 核心组件实现**
  - [ ] T1.1 新增 `StreamTranscriber` 类（音频缓冲、定时触发、转写）
  - [ ] T1.2 实现 `--flush-interval` 参数解析

- [ ] **T2: 状态机改造**
  - [ ] T2.1 新增 `STREAMING` 状态
  - [ ] T2.2 修改 LISTENING → STREAMING 转换逻辑
  - [ ] T2.3 实现定时触发转写循环

- [ ] **T3: UI 改造**
  - [ ] T3.1 增量文本刷新显示（Rich 同行动态更新）
  - [ ] T3.2 状态指示器更新（显示"正在转写..."）

- [ ] **T4: CLI 参数**
  - [ ] T4.1 添加 `--stream` / `--no-stream` 参数
  - [ ] T4.2 添加 `--flush-interval` 参数

- [ ] **T5: 集成测试**
  - [ ] T5.1 whisper.cpp 流式模式 E2E 测试
  - [ ] T5.2 SenseVoice 流式模式 E2E 测试
  - [ ] T5.3 性能基准测试（首字延迟、内存）

### 5.2 任务依赖

```
T1 → T2 → T3
T1, T2, T3 → T5.1 (whisper.cpp E2E)
T1, T2, T3 → T5.2 (SenseVoice E2E)
T5.1, T5.2 → T5.3
```

**注意**：T5.1 和 T5.2 可并行进行，两个引擎共用 StreamTranscriber

---

## 6. 测试验证

### 6.1 功能测试

| 测试用例 | 输入 | 预期输出 | 通过标准 |
|---------|------|---------|---------|
| TC-01 | whisper.cpp 流式模式，说 5 秒话 | 说话过程中看到增量文本输出 | 流式输出 ≥ 2 次 |
| TC-02 | SenseVoice 流式模式，说 5 秒话 | 说话过程中看到增量文本输出 | 流式输出 ≥ 2 次 |
| TC-03 | `--no-stream` 模式，说 5 秒话 | 说完后才显示完整结果 | 说话过程中无输出，说完后才输出 |
| TC-04 | 流式模式下快速打断（<1s） | 正常输出短句 | 输出正确 |
| TC-05 | 连续长语音（>30s） | 持续流式输出无中断 | 流式输出正常，无内存泄漏 |

### 6.2 边界测试

| 测试用例 | 场景 | 通过标准 |
|---------|------|---------|
| BC-01 | 极短语音（<0.5s） | 不崩溃，正常输出 |
| BC-02 | 极长语音（>60s） | 流式正常，无内存泄漏 |
| BC-03 | 无声环境 | 不触发误识别 |

### 6.3 性能测试

| 测试指标 | 测试方法 | 目标值 |
|---------|---------|--------|
| 首字延迟 | 从开始说话到显示第一个字的时间 | ≤ flush_interval + 转写时间（≤ 3s） |
| 内存增长 | 60s 连续流式转写后内存对比 | ≤ +30MB |
| CPU 峰值 | 流式转写时 CPU 占用 | ≤ +30% |

---

## 7. 成功标准

| 标准 | 验证方法 |
|------|---------|
| ✅ whisper.cpp 支持流式输出 | 启用 `--stream` 后说话过程看到增量文本（≥2次） |
| ✅ SenseVoice 支持流式输出 | 启用 `--stream` 后说话过程看到增量文本（≥2次） |
| ✅ 向后兼容 | 不加 `--stream` 参数时行为与原来一致 |
| ✅ 首字延迟 ≤ 3s | 使用 `--stream` 模式测量（flush_interval + 转写时间） |
| ✅ 内存增长 ≤ 30MB | 60s 连续流式转写后内存对比基准 |
| ✅ 文档更新 | README 说明新增 `--stream` 和 `--flush-interval` 参数 |

---

## Changelog

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0.0 | 2026/05/09 | 初始版本 |
| v1.0.1 | 2026/05/09 | 修复质量检查问题：<br>- 统一首字延迟指标为 ≤1.5s<br>- 修复测试用例与性能目标不一致<br>- 修复拼写错误，明确 CLI 默认行为 |
| v1.1.0 | 2026/05/09 | **重大架构变更**：<br>- 新增方案 A（定时批量转写）：简单可靠，不依赖引擎原生流式 API<br>- 方案 B（原生流式 API）降级为备选<br>- 任务拆解大幅简化：T1~T5 简化版<br>- 新增 `--flush-interval` 参数控制刷新间隔<br>- 性能目标更新：增量间隔由 flush_interval 决定 |
| v1.1.1 | 2026/05/09 | 统一文档围绕方案 A：<br>- 目标章节延迟指标与 flush_interval 一致<br>- Scope In/Out 移除方案 B 残留描述<br>- 性能测试与方案 A 目标对齐（≤3s, +30MB, +30%） |
| v1.1.2 | 2026/05/09 | 细化算法说明：<br>- 明确每次 flush 用 ALL 累积音频转写（非仅最新片段）<br>- 维护双缓冲区：`_audio_buffer`（本次）和 `_all_audio_buffer`（累积）<br>- 说明持续累积不清空的语义 |
