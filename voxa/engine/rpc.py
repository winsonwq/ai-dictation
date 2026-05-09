"""
voxa/engine/rpc.py
stdin/stdout JSON-RPC 协议
用于 Tauri 进程间通信

当 stdout 被重定向时自动切换为 JSON 模式，
否则保持 TTY 交互模式。
"""

import sys
import json
import uuid
from typing import Any, Callable, Optional
from dataclasses import dataclass


@dataclass
class RPCRequest:
    id: str
    method: str
    params: dict


@dataclass
class RPCResponse:
    id: str
    result: Any = None
    error: Optional[str] = None


class RPCHandler:
    """
    JSON-RPC 请求处理器

    注册方法后，可以通过 stdin 接收命令并通过 stdout 响应

    使用方式:
        rpc = RPCHandler()
        rpc.register('start', start_handler)
        rpc.register('stop', stop_handler)
        rpc.run()  # 事件循环
    """

    def __init__(self):
        self._handlers: dict[str, Callable] = {}
        self._json_mode = self._detect_json_mode()

    def _detect_json_mode(self) -> bool:
        """
        检测是否应该使用 JSON 模式
        如果 stdout 被重定向（不是 TTY），使用 JSON 模式
        """
        return not sys.stdout.isatty()

    def register(self, method: str, handler: Callable):
        """注册一个 RPC 方法"""
        self._handlers[method] = handler

    def _handle_request(self, req: RPCRequest) -> RPCResponse:
        """处理单个请求"""
        handler = self._handlers.get(req.method)
        if handler is None:
            return RPCResponse(
                id=req.id,
                error=f"Method not found: {req.method}"
            )

        try:
            result = handler(**req.params)
            return RPCResponse(id=req.id, result=result)
        except Exception as e:
            return RPCResponse(
                id=req.id,
                error=f"{type(e).__name__}: {e}"
            )

    def _send_response(self, resp: RPCResponse):
        """发送响应"""
        if self._json_mode:
            line = json.dumps({
                'id': resp.id,
                'result': resp.result,
                'error': resp.error,
            })
            sys.stdout.write(line + '\n')
            sys.stdout.flush()
        else:
            # TTY 模式下只打印信息
            if resp.error:
                print(f'[RPC Error] {resp.error}', file=sys.stderr)

    def _read_requests(self):
        """读取并处理请求"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                req = RPCRequest(
                    id=data.get('id', str(uuid.uuid4())),
                    method=data.get('method', ''),
                    params=data.get('params', {}),
                )
                resp = self._handle_request(req)
                self._send_response(resp)
            except json.JSONDecodeError as e:
                print(f'[RPC] 无效的 JSON: {e}', file=sys.stderr)

    def run(self):
        """运行事件循环"""
        if self._json_mode:
            self._read_requests()
        else:
            # TTY 模式：输出提示信息
            print('[RPC] 运行在 TTY 模式，仅支持交互调用', file=sys.stderr)
            print('[RPC] 使用 stdin 重定向来发送 JSON-RPC 命令', file=sys.stderr)

    def emit(self, event: str, data: Any):
        """
        主动发送事件（服务端推送到客户端）
        用于实时转写结果的流式推送
        """
        if self._json_mode:
            line = json.dumps({
                'event': event,
                'data': data,
            })
            sys.stdout.write(line + '\n')
            sys.stdout.flush()


# 全局 handler
_handler: Optional[RPCHandler] = None


def get_handler() -> RPCHandler:
    global _handler
    if _handler is None:
        _handler = RPCHandler()
    return _handler


def register(method: str, handler: Callable):
    get_handler().register(method, handler)


if __name__ == '__main__':
    # 简单测试
    rpc = RPCHandler()

    def echo(message: str) -> str:
        return f'echo: {message}'

    rpc.register('echo', echo)
    print('测试 RPC...')
    rpc.run()
