"""
voxa/events.py
简单的事件发射器
"""

from typing import Callable, Dict, List


class EventEmitter:
    """
    轻量级事件发射器

    使用方式:
        emitter = EventEmitter()
        emitter.on('data', lambda x: print(x))
        emitter.emit('data', 'hello')  # → prints 'hello'
        emitter.off('data', handler)   # 移除监听器
    """

    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, handler: Callable):
        """注册事件监听器"""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(handler)
        return handler  # 返回 handler，方便 off() 使用

    def off(self, event: str, handler: Callable):
        """移除事件监听器"""
        if event not in self._listeners:
            return
        self._listeners[event] = [h for h in self._listeners[event] if h != handler]

    def once(self, event: str, handler: Callable):
        """注册只触发一次的事件监听器"""
        def wrapper(*args, **kwargs):
            self.off(event, wrapper)
            handler(*args, **kwargs)
        return self.on(event, wrapper)

    def emit(self, event: str, *args, **kwargs):
        """触发事件"""
        if event not in self._listeners:
            return
        for handler in self._listeners[event]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                # 不让一个 handler 的错误影响其他 handler
                import logging
                logging.getLogger('voxa.events').error(f'Event handler error: {e}')

    def listener_count(self, event: str) -> int:
        """返回某个事件的监听器数量"""
        return len(self._listeners.get(event, []))

    def event_names(self) -> List[str]:
        """返回所有已注册的事件名"""
        return list(self._listeners.keys())
