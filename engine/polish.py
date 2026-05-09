"""
engine/polish.py
LLM 文本润色/纠错模块
通过 OpenRouter API 调用 LLM
"""

import os
import asyncio
import logging
import time
from typing import Optional
import threading

_logger = logging.getLogger('dictation.polish')

# 默认 prompt
SYSTEM_PROMPT = """你是一个专业的语音转写后处理器。你的任务：
1. 修正明显的语音识别错误
2. 添加适当的标点符号（，。！？：；""）
3. 适当分段（超过40字考虑分句）
4. 保持原意，不添加额外内容
5. 不改变人名、地名、术语
6. 如果文本已经很好，直接返回原文本，不要多余解释

输入：语音转写的原始文本（可能没有标点）
输出：处理后的文本（直接输出，不要解释，不要加引号）"""

DEFAULT_MODEL = 'qwen/qwen3.5-plus-02-15'
DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'


class PolishEngine:
    """
    LLM 文本润色引擎

    使用 OpenRouter API，支持流式和非流式两种模式

    使用方式:
        polish = PolishEngine(api_key='...')
        result = await polish.polish("今天天气不错")
        print(result)  # "今天天气不错。"
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ):
        """
        Args:
            api_key: OpenRouter API key，默认从环境变量 OPENROUTER_API_KEY 读取
            model: 模型 ID，默认 qwen/qwen3.5-plus-02-15
            base_url: API 地址
            timeout: 请求超时（秒）
            temperature: 采样温度，越低越确定性
            max_tokens: 最大输出 token 数
        """
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError('需要提供 OpenRouter API key，或设置 OPENROUTER_API_KEY 环境变量')

        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        """延迟初始化 client"""
        if self._client is None:
            from openai import OpenAI
            with self._lock:
                if self._client is None:
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        timeout=self.timeout,
                        default_headers={
                            'HTTP-Referer': 'https://ai-dictation.local',
                            'X-Title': 'ai-dictation',
                        },
                    )
        return self._client

    def polish(self, text: str) -> str:
        """
        同步润色

        Args:
            text: 原始转写文本

        Returns:
            润色后的文本
        """
        if not text or not text.strip():
            return text

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            _logger.error(f'润色失败: {e}')
            return text  # 失败时返回原文

    async def polish_async(self, text: str) -> str:
        """
        异步润色（不阻塞事件循环）

        Args:
            text: 原始转写文本

        Returns:
            润色后的文本
        """
        if not text or not text.strip():
            return text

        loop = asyncio.get_event_loop()

        def _call():
            client = self._get_client()
            return client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        try:
            response = await loop.run_in_executor(None, _call)
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            _logger.error(f'异步润色失败: {e}')
            return text

    def polish_stream(self, text: str):
        """
        流式润色

        Yields:
            文本片段
        """
        if not text or not text.strip():
            yield text
            return

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': text},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except Exception as e:
            _logger.error(f'流式润色失败: {e}')
            yield text


class PolishCache:
    """
    简单的润色结果缓存
    避免对相同文本重复调用 LLM
    """

    def __init__(self, max_size: int = 100):
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._max_size = max_size

    def get(self, text: str) -> Optional[str]:
        """命中缓存返回结果，否则 None"""
        return self._cache.get(text)

    def set(self, text: str, result: str):
        """存入缓存"""
        if text in self._cache:
            self._cache.move_to_end(text)
            return
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[text] = result

    def clear(self):
        self._cache.clear()


# 全局缓存实例
_polish_cache = PolishCache()


def set_cache_size(size: int):
    """设置缓存大小"""
    global _polish_cache
    _polish_cache = PolishCache(max_size=size)


if __name__ == '__main__':
    import sys

    print('=== Polish 模块测试 ===')

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        # 尝试从配置文件读取
        key_file = os.path.expanduser('~/.openclaw/configs/github-token.txt')
        if os.path.exists(key_file):
            with open(key_file) as f:
                # 飞书 token 文件里的 OpenRouter key 格式
                content = f.read()
                if 'sk-or-v1' in content:
                    api_key = content.strip().split('\n')[0]
        if not api_key:
            print('❌ 未找到 OpenRouter API key，跳过测试')
            sys.exit(0)

    engine = PolishEngine(api_key=api_key)

    # 测试用例
    test_cases = [
        '今天天气不错',
        '我想吃火锅明天有空吗',
        '这个项目大概需要两周时间',
    ]

    for text in test_cases:
        print(f'\n原文: {text}')
        start = time.time()
        result = engine.polish(text)
        elapsed = time.time() - start
        print(f'润色: {result} ({elapsed:.1f}s)')

    print('\n✅ Polish 模块测试通过')
