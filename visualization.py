"""
Модуль визуализации: отрисовка bbox, меток, алертов на кадре.

Цветовая схема:
- Зелёный — взрослый
- Жёлтый — ребёнок с сопровождением
- Красный (толстая рамка) — ребёнок без сопровождения (алерт)
"""

import cv2
import numpy as np
from models import PersonDetection

# Цвета (BGR)
COLOR_ADULT = (0, 180, 0)          # Зелёный
COLOR_CHILD_OK = (0, 220, 220)     # Жёлтый
COLOR_CHILD_ALERT = (0, 0, 255)    # Красный
COLOR_TEXT_BG = (0, 0, 0)          # Фон текста


def draw_persons(
    frame: np.ndarray,
    persons: list[PersonDetection],
    age_labels: list[str],
    alone_times: dict[int, float],
    alerted_ids: set[int],
    debug_scores: dict[int, dict] | None = None
) -> np.ndarray:
    """
    Нарисовать bbox и метки для всех людей на кадре.

    Args:
        frame: BGR-кадр для рисования (модифицируется in-place).
        persons: Список обнаруженных людей.
        age_labels: Метки "child"/"adult" для каждого.
        alone_times: {track_id: секунды_без_сопровождения}.
        alerted_ids: Множество track_id с активным алертом.

    Returns:
        Аннотированный кадр.
    """
    for person, age_label in zip(persons, age_labels):
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        track_id = person.track_id
        is_child = age_label == "child"
        is_alerted = track_id in alerted_ids

        # Выбираем цвет и толщину рамки
        if is_child and is_alerted:
            color = COLOR_CHILD_ALERT
            thickness = 3
        elif is_child:
            color = COLOR_CHILD_OK
            thickness = 2
        else:
            color = COLOR_ADULT
            thickness = 2

        # Рисуем bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ID на рамке не показываем. Для несопровождаемого ребёнка — таймер до тревоги.
        label = ""
        if is_child:
            alone_sec = alone_times.get(track_id, 0.0)
            if alone_sec > 0:
                label = f"{alone_sec:.0f}s"

        # Рисуем метку с фоном (только если есть что показать)
        if label:
            _draw_label(frame, label, (x1, y1 - 6), color, scale=0.4)

        # Debug-скоры под bbox (только если включены)
        if debug_scores and track_id in debug_scores:
            scores = debug_scores[track_id]
            parts = [f"{k}:{v:.2f}" for k, v in scores.items()]
            debug_text = " ".join(parts)
            _draw_label(frame, debug_text, (x1, y2 + 12), (200, 200, 200), scale=0.35)

    return frame


def draw_alert_banner(frame: np.ndarray, alert_count: int) -> np.ndarray:
    """Нарисовать баннер с количеством активных алертов вверху кадра."""
    if alert_count == 0:
        return frame

    text = f"ALERT: {alert_count} unaccompanied"
    h, w = frame.shape[:2]

    # Красный баннер вверху
    cv2.rectangle(frame, (0, 0), (w, 26), COLOR_CHILD_ALERT, -1)
    cv2.putText(
        frame, text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Показать FPS в правом верхнем углу."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text,
        (w - 90, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (255, 255, 255), 1
    )
    return frame


def _draw_label(frame, text, position, color, scale=0.4):
    """Нарисовать текст с фоном."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Фон под текстом
    cv2.rectangle(
        frame,
        (x, y - text_h - 3),
        (x + text_w + 4, y + 3),
        COLOR_TEXT_BG, -1
    )

    # Текст
    cv2.putText(frame, text, (x + 2, y), font, scale, color, thickness)
