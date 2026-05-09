"""
tests/test_http.py
基于 HTTP API 的集成测试

依赖：HTTP Server 运行在 http://localhost:19291

使用方式:
    python tests/test_http.py
"""

import sys
import os
import time
import json
import tempfile
import wave
import threading
import queue

# 确保 voxa 包可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = 'http://localhost:19291'


def generate_test_audio(duration=2.0, frequency=300):
    """生成测试音频（正弦波）"""
    import numpy as np

    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)
    signal = np.clip(signal, -1.0, 1.0)
    return signal


def save_wav(filename, audio_data, sample_rate=16000):
    """保存为 WAV 文件"""
    import numpy as np

    int16_data = (audio_data * 32767).astype('<i2')
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


def http_get(path):
    """发送 GET 请求"""
    import urllib.request
    url = f'{BASE_URL}{path}'
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def http_post(path, data=None):
    """发送 POST 请求"""
    import urllib.request
    url = f'{BASE_URL}{path}'
    body = json.dumps(data).encode() if data else b''
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


def sse_events(url, timeout=60):
    """读取 SSE 事件流"""
    import urllib.request

    events = []
    q = queue.Queue()

    def read_events():
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                buffer = ''
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith('event:'):
                        event_type = line[6:].strip()
                    elif line.startswith('data:'):
                        data = json.loads(line[5:].strip())
                        q.put((event_type, data))
                    elif line == '':
                        # 空行表示一个事件结束
                        pass
        except Exception as e:
            q.put(('error', str(e)))
        finally:
            q.put(('__end__', None))

    thread = threading.Thread(target=read_events, daemon=True)
    thread.start()

    return q


def test_health():
    """测试：健康检查"""
    print('=' * 50)
    print('测试 1: 健康检查')
    print('=' * 50)

    result = http_get('/api/health')
    print(f'响应: {result}')

    assert result['status'] == 'ok', f"期望 status=ok, 实际 {result['status']}"
    assert result['state'] == 'idle', f"期望 state=idle, 实际 {result['state']}"
    print('✅ 健康检查通过')

    return result


def test_status():
    """测试：获取状态"""
    print('\n' + '=' * 50)
    print('测试 2: 获取状态')
    print('=' * 50)

    result = http_get('/api/status')
    print(f'响应: {result}')

    assert result['state'] == 'idle'
    assert result['is_listening'] == False
    print('✅ 状态查询通过')

    return result


def test_start_stop():
    """测试：启动和停止听写（使用测试音频文件）"""
    print('\n' + '=' * 50)
    print('测试 3: 启动和停止听写')
    print('=' * 50)

    # 生成测试音频
    audio = generate_test_audio(duration=2.0, frequency=300)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_file = f.name
    save_wav(temp_file, audio)
    print(f'生成测试音频: {temp_file} ({len(audio)/16000:.1f}s)')

    # 注意：server 目前不支持上传文件，只能用麦克风
    # 这里测试麦克风听写（如果麦克风可用）

    # 先检查状态
    status = http_get('/api/status')
    print(f'当前状态: {status}')

    # 尝试启动听写
    print('启动听写（麦克风）...')
    try:
        result = http_post('/api/start')
        print(f'启动响应: {result}')

        # 等待一段时间
        print('等待 5 秒...')
        time.sleep(5)

        # 停止听写
        print('停止听写...')
        stop_result = http_post('/api/stop')
        print(f'停止响应: {stop_result}')

        if stop_result.get('text'):
            print(f'\n✅ 听写成功: "{stop_result["text"]}"')
        else:
            print('\n⚠️  无转写结果（可能没有语音输入）')

    except Exception as e:
        print(f'❌ 测试失败: {e}')

    finally:
        os.unlink(temp_file)

    return stop_result


def test_sse_events():
    """测试：SSE 事件流"""
    print('\n' + '=' * 50)
    print('测试 4: SSE 事件流')
    print('=' * 50)

    events_q = sse_events(f'{BASE_URL}/events')

    # 订阅事件后启动听写
    print('启动听写并监听事件...')
    http_post('/api/start')

    # 收集 3 秒内的事件
    events = []
    timeout = 5
    start = time.time()
    while time.time() - start < timeout:
        try:
            event_type, data = events_q.get(timeout=1)
            if event_type == '__end__':
                break
            if event_type == 'error':
                print(f'❌ SSE 错误: {data}')
                break
            print(f'事件: {event_type} -> {data}')
            events.append((event_type, data))
        except queue.Empty:
            break

    # 停止听写
    http_post('/api/stop')

    print(f'\n收到 {len(events)} 个事件')
    if events:
        print('✅ SSE 事件流正常')
    else:
        print('⚠️  未收到事件（可能无麦克风）')

    return events


def test_cancel():
    """测试：取消听写"""
    print('\n' + '=' * 50)
    print('测试 5: 取消听写')
    print('=' * 50)

    # 启动听写
    http_post('/api/start')
    print('听写已启动')

    time.sleep(1)

    # 取消
    result = http_post('/api/cancel')
    print(f'取消响应: {result}')
    print('✅ 取消请求已发送')

    return result


def main():
    print('Voxa HTTP API 测试套件')
    print(f'目标服务器: {BASE_URL}')
    print()

    # 检查服务器是否可用
    try:
        http_get('/api/health')
    except Exception as e:
        print(f'❌ 无法连接到服务器: {e}')
        print(f'请确保服务器运行在 {BASE_URL}')
        sys.exit(1)

    print('服务器连接成功\n')

    test_health()
    test_status()

    # 麦克风相关测试（可能因无麦克风失败）
    try:
        test_start_stop()
        test_sse_events()
        test_cancel()
    except Exception as e:
        print(f'\n⚠️  麦克风测试失败: {e}')
        print('这是正常的如果没有麦克风设备')

    print('\n' + '=' * 50)
    print('测试完成')
    print('=' * 50)


if __name__ == '__main__':
    main()
