"""
voxa/server.py
HTTP Server - 供外部集成的 HTTP 接口

启动方式:
    python -m voxa.server
    python -m voxa.server --port 8765

API:
    POST /api/start          开始听写
    POST /api/stop           停止听写
    GET  /api/status         当前状态
    GET  /api/health         健康检查

    SSE /events              实时事件流
"""

import sys
import json
import threading
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

_logger = logging.getLogger('voxa.server')


class VoxaServer:
    """
    Voxa HTTP Server

    提供 REST API + SSE 事件流
    """

    def __init__(self, core, port: int = 8765):
        self.core = core
        self.port = port
        self._sse_clients: list = []
        self._server: Optional[HTTPServer] = None

        # 订阅核心事件，广播给所有 SSE 客户端
        self.core.on('state_change', self._broadcast_state)
        self.core.on('asr.partial', lambda text: self._broadcast('asr.partial', {'text': text}))
        self.core.on('asr.final', self._broadcast_asr_final)
        self.core.on('polish.result', lambda text: self._broadcast('polish.result', {'text': text}))
        self.core.on('vad.speech_start', lambda: self._broadcast('vad.speech_start', {}))
        self.core.on('vad.speech_end', lambda: self._broadcast('vad.speech_end', {}))
        self.core.on('error', lambda msg: self._broadcast('error', {'message': msg}))

    def _broadcast_state(self, state):
        self._broadcast('state_change', {'state': state.value})

    def _broadcast_asr_final(self, result):
        self._broadcast('asr.final', {
            'text': result.text,
            'language': result.language,
            'avg_logprob': result.avg_logprob,
        })

    def _broadcast(self, event: str, data: dict):
        """向所有 SSE 客户端广播消息"""
        message = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        dead = []
        for client in self._sse_clients:
            try:
                client['queue'].put(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self._sse_clients.remove(client)

    def add_sse_client(self, client):
        self._sse_clients.append(client)

    def remove_sse_client(self, client):
        if client in self._sse_clients:
            self._sse_clients.remove(client)

    def start(self):
        """在后台线程启动服务器"""
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        _logger.info(f'HTTP Server 启动: http://localhost:{self.port}')

    def _serve(self):
        self._server = HTTPServer(('localhost', self.port), self._make_handler())
        self._server.serve_forever()

    def _make_handler(self):
        """创建请求处理器"""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                _logger.debug(f"{self.client_address[0]} {format % args}")

            def send_json(self, code: int, data: dict):
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def send_sse(self, code: int):
                self.send_response(code)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == '/api/health':
                    self.send_json(200, {'status': 'ok', 'state': server.core.state.value})

                elif path == '/api/status':
                    self.send_json(200, {
                        'state': server.core.state.value,
                        'is_listening': server.core.is_listening,
                    })

                elif path == '/events':
                    # SSE 事件流
                    import queue
                    client_queue = queue.Queue()
                    client = {'queue': client_queue}
                    server.add_sse_client(client)

                    self.send_sse(200)

                    try:
                        while True:
                            msg = client_queue.get(timeout=30)
                            if msg is None:  # 客户端断开
                                break
                            self.wfile.write(msg.encode())
                            self.wfile.flush()
                    except Exception:
                        pass
                    finally:
                        server.remove_sse_client(client)

                else:
                    self.send_json(404, {'error': 'Not found'})

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path

                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length) if content_length > 0 else b''
                try:
                    data = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    self.send_json(400, {'error': 'Invalid JSON'})
                    return

                if path == '/api/start':
                    if server.core.is_listening:
                        self.send_json(200, {'status': 'already_listening'})
                    else:
                        # 在独立线程中启动，避免阻塞 HTTP 请求
                        thread = threading.Thread(target=server.core.start, daemon=True)
                        thread.start()
                        self.send_json(200, {'status': 'started'})

                elif path == '/api/stop':
                    # stop() 可能在润色，需要时间，在线程中执行
                    result = [None]
                    def do_stop():
                        result[0] = server.core.stop()
                    t = threading.Thread(target=do_stop, daemon=True)
                    t.start()
                    t.join(timeout=60)
                    self.send_json(200, {'status': 'stopped', 'text': result[0] or ''})

                elif path == '/api/cancel':
                    server.core.cancel()
                    self.send_json(200, {'status': 'cancelled'})

                else:
                    self.send_json(404, {'error': 'Not found'})

        return Handler

    def stop(self):
        """停止服务器"""
        if self._server:
            self._server.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Voxa HTTP Server')
    parser.add_argument('--port', type=int, default=8765, help='HTTP 端口 (默认 8765)')
    parser.add_argument('--engine', default='whisper', help='ASR 引擎')
    parser.add_argument('--model', default='small', help='模型大小')
    parser.add_argument('--no-polish', action='store_true', help='禁用润色')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    from voxa import VoxaCore

    core = VoxaCore(
        engine=args.engine,
        model_size=args.model,
        disable_polish=args.no_polish,
    )

    server = VoxaServer(core, port=args.port)
    server.start()

    print(f'Voxa HTTP Server 运行中: http://localhost:{args.port}')
    print(f'API 文档: http://localhost:{args.port}/api/health')
    print('按 Ctrl+C 停止')

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\n停止服务器...')
        server.stop()


if __name__ == '__main__':
    main()
