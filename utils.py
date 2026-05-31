"""
Вспомогательные функции: расстояния, центры bbox, FPS-счётчик, webhook.
"""

import time
import math
import requests
import config


def bbox_center(bbox):
    """Вычислить центр bounding box. bbox = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_dist(p1, p2):
    """Евклидово расстояние между двумя точками (x, y)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def bbox_min_distance(bbox1, bbox2):
    """Минимальное расстояние между двумя bbox (по ближайшим краям).
    Если боксы пересекаются — возвращает 0."""
    x1a, y1a, x2a, y2a = bbox1
    x1b, y1b, x2b, y2b = bbox2

    dx = max(0, max(x1a - x2b, x1b - x2a))
    dy = max(0, max(y1a - y2b, y1b - y2a))

    return math.sqrt(dx * dx + dy * dy)


class FPSCounter:
    """Простой счётчик FPS на основе скользящего среднего."""

    def __init__(self, avg_frames=30):
        self.avg_frames = avg_frames
        self.timestamps = []

    def tick(self):
        """Вызывать каждый кадр."""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.avg_frames:
            self.timestamps.pop(0)

    def get_fps(self):
        """Получить текущий FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def send_webhook_alert(track_id, elapsed_sec):
    """Отправить алерт на webhook (если настроен)."""
    if not config.ALERT_WEBHOOK_URL:
        return
    try:
        requests.post(config.ALERT_WEBHOOK_URL, json={
            "type": "unaccompanied_child",
            "track_id": track_id,
            "alone_seconds": round(elapsed_sec, 1),
            "timestamp": time.time()
        }, timeout=2)
    except requests.RequestException:
        pass  # Не падаем из-за проблем с webhook
