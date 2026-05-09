"""
voxa/server.py
HTTP Server - 简化的转写接口

API:
    POST /api/transcribe        单次转写（上传音频文件）
    POST /api/transcribe/stream 流式转写（SSE 事件流）
    GET  /api/health            健康检查
"""

import sys
import os
import json
import tempfile
import threading
import argparse
import logging
import wave
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

_logger = logging.getLogger('voxa.server')


class VoxaServer:
    """
    Voxa HTTP Server

    提供两个转写接口，无状态设计
    """

    def __init__(self, core, port: int = 8765):
        self.core = core
        self.port = port
        self._sse_clients: list = []
        self._server: Optional[HTTPServer] = None

    def start(self):
        self._server, actual_port = self._find_available_port()
        if self._server is None:
            raise RuntimeError(f'无法找到可用端口 ({self.port} - {self.port + 99})')
        self.port = actual_port
        _logger.info(f'HTTP Server 启动: http://localhost:{actual_port}')
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)
        self._thread.start()

    def _find_available_port(self):
        for port in range(self.port, self.port + 100):
            try:
                server = HTTPServer(('localhost', port), self._make_handler())
                return server, port
            except OSError as e:
                if e.errno == 98:
                    continue
                raise
        return None, None

    def _serve_forever(self):
        self._server.serve_forever()

    def _make_handler(self):
        server = self
        core = self.core

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

            def send_error_json(self, code: int, message: str):
                self.send_json(code, {'error': message})

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path == '/api/health':
                    self.send_json(200, {'status': 'ok'})
                else:
                    self.send_error_json(404, 'Not found')

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path == '/api/transcribe':
                    self._handle_transcribe(core, server)
                elif path == '/api/transcribe/stream':
                    self._handle_transcribe_stream(core, server)
                else:
                    self.send_error_json(404, 'Not found')

            def _handle_transcribe(self, core, server):
                """单次转写：接收音频文件，返回转写结果"""
                import cgi

                content_type = self.headers.get('Content-Type', '')
                if 'multipart/form-data' in content_type:
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
                    )
                    # Python 3.10 compatibility: use ['audio'] instead of getfile()
                    if 'audio' not in form:
                        self.send_error_json(400, '缺少 audio 文件')
                        return
                    audio_field = form['audio']
                    if not hasattr(audio_field, 'file') or audio_field.file is None:
                        self.send_error_json(400, 'audio 文件无效')
                        return
                    audio_data = audio_field.file.read()
                else:
                    content_length = int(self.headers.get('Content-Length', 0))
                    audio_data = self.rfile.read(content_length) if content_length > 0 else b''

                if not audio_data:
                    self.send_error_json(400, '音频数据为空')
                    return

                # 保存到临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name

                try:
                    result_text = [None]
                    polish_text = [None]
                    error_msg = [None]

                    def do_transcribe():
                        try:
                            from voxa import VoxaCore
                            _core = VoxaCore(
                                engine=core.engine,
                                model_size=core.model_size,
                                llm_model=core.llm_model,
                                disable_polish=core.disable_polish,
                                use_file=temp_path,
                                streaming=False,
                            )
                            _core.on('asr.final', lambda r: result_text.__setitem__(0, r.text))
                            _core.on('polish.result', lambda t: polish_text.__setitem__(0, t))
                            _core.on('error', lambda m: error_msg.__setitem__(0, m))
                            _core.start()
                            import time
                            for _ in range(200):
                                if _core.state.value in ('idle', 'error'):
                                    break
                                time.sleep(0.5)
                            _core.stop()
                        except Exception as e:
                            error_msg.__setitem__(0, str(e))
                            _logger.error(f'转写失败: {e}')

                    t = threading.Thread(target=do_transcribe)
                    t.start()
                    t.join(timeout=120)

                    if error_msg[0]:
                        self.send_error_json(500, error_msg[0])
                        return

                    self.send_json(200, {
                        'text': result_text[0] or '',
                        'polished': polish_text[0],
                    })

                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

            def _handle_transcribe_stream(self, core, server):
                """
                流式转写：接收音频，返回 SSE 事件流
                与 /api/transcribe 相同，但实时返回 partial 结果
                """
                content_type = self.headers.get('Content-Type', '')
                if 'multipart/form-data' in content_type:
                    import cgi
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
                    )
                    if 'audio' not in form:
                        self.send_error_json(400, '缺少 audio 文件')
                        return
                    audio_field = form['audio']
                    if not hasattr(audio_field, 'file') or audio_field.file is None:
                        self.send_error_json(400, 'audio 文件无效')
                        return
                    audio_data = audio_field.file.read()
                else:
                    content_length = int(self.headers.get('Content-Length', 0) or 0)
                    audio_data = self.rfile.read(content_length) if content_length > 0 else b''

                if not audio_data:
                    self.send_error_json(400, '音频数据为空')
                    return

                # 保存到临时文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_data)
                    temp_path = f.name

                try:
                    result_queue = __import__('queue').Queue()
                    running = [True]

                    def do_transcribe():
                        try:
                            from voxa import VoxaCore
                            _core = VoxaCore(
                                engine=core.engine,
                                model_size=core.model_size,
                                llm_model=core.llm_model,
                                disable_polish=core.disable_polish,
                                use_file=temp_path,
                                streaming=True,
                            )
                            _core.on('asr.partial', lambda t: result_queue.put(('partial', t)))
                            _core.on('asr.final', lambda r: result_queue.put(('final', r.text)))
                            _core.on('polish.result', lambda t: result_queue.put(('polish', t)))
                            _core.on('vad.speech_start', lambda: result_queue.put(('vad_start', '')))
                            _core.on('vad.speech_end', lambda: result_queue.put(('vad_end', '')))
                            _core.on('error', lambda m: result_queue.put(('error', m)))
                            _core.start()
                            import time
                            for _ in range(600):
                                if not running[0]:
                                    break
                                if _core.state.value in ('idle', 'error'):
                                    break
                                time.sleep(0.5)
                            _core.stop()
                        except Exception as e:
                            _logger.error(f'流式转写失败: {e}')
                            result_queue.put(('error', str(e)))
                        finally:
                            running[0] = False

                    t = threading.Thread(target=do_transcribe)
                    t.start()

                    # 发送 SSE 响应
                    self.send_sse(200)
                    import time as _time
                    while running[0]:
                        try:
                            event_type, data = result_queue.get(timeout=0.5)
                            if event_type == 'error':
                                msg = f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                                self.wfile.write(msg.encode())
                                self.wfile.flush()
                                break
                            elif event_type == 'partial':
                                msg = f"event: partial\ndata: {json.dumps({'text': data})}\n\n"
                                self.wfile.write(msg.encode())
                                self.wfile.flush()
                            elif event_type == 'final':
                                msg = f"event: final\ndata: {json.dumps({'text': data})}\n\n"
                                self.wfile.write(msg.encode())
                                self.wfile.flush()
                            elif event_type == 'polish':
                                msg = f"event: polish\ndata: {json.dumps({'text': data})}\n\n"
                                self.wfile.write(msg.encode())
                                self.wfile.flush()
                                break
                            elif event_type in ('vad_start', 'vad_end'):
                                msg = f"event: {event_type}\ndata: {json.dumps({})}\n\n"
                                self.wfile.write(msg.encode())
                                self.wfile.flush()
                        except __import__('queue').Empty:
                            if not running[0]:
                                break
                            continue

                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

        return Handler

    def stop(self):
        if self._server:
            self._server.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Voxa HTTP Server')
    parser.add_argument('--port', type=int, default=8765, help='HTTP 端口 (默认 8765)')
    parser.add_argument('--engine', default='whisper', choices=['whisper', 'sensevoice'])
    parser.add_argument('--model', default='small', help='模型大小')
    parser.add_argument('--llm-model', default='qwen/qwen3.5-plus-02-15', help='润色模型')
    parser.add_argument('--no-polish', action='store_true', help='禁用润色')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    from voxa import VoxaCore

    # VoxaCore 实例，只用于传递配置
    core = VoxaCore(
        engine=args.engine,
        model_size=args.model,
        llm_model=args.llm_model,
        disable_polish=args.no_polish,
    )

    server = VoxaServer(core, port=args.port)
    server.start()

    actual_port = server.port
    print(f'Voxa HTTP Server 运行中: http://localhost:{actual_port}')
    print(f'接口:')
    print(f'  POST /api/transcribe         - 单次转写（上传音频文件）')
    print(f'  POST /api/transcribe/stream - 流式转写（SSE 事件流）')
    print(f'  GET  /api/health            - 健康检查')
    print('按 Ctrl+C 停止')

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\n停止服务器...')
        server.stop()


if __name__ == '__main__':
    main()
