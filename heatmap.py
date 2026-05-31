"""
Модуль тепловой карты (heatmap).

Накапливает позиции, где дети находились без сопровождения,
и визуализирует их как цветовой оверлей на кадре.
"""

import cv2
import numpy as np
from models import PersonDetection
from utils import bbox_center
import config


class HeatmapAccumulator:
    """
    Аккумулятор тепловой карты.

    Каждый кадр:
    1. Применяет decay (затухание) к предыдущим значениям.
    2. Добавляет пятно (Gaussian blob) в точке каждого ребёнка без сопровождения.
    3. Может отрендерить цветной оверлей для наложения на кадр.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.accumulator = np.zeros((height, width), dtype=np.float32)

    def update(self, unaccompanied_children: list[PersonDetection]):
        """Обновить heatmap: затухание + новые точки."""
        # Затухание предыдущих значений
        self.accumulator *= config.HEATMAP_DECAY

        # Рисуем круг прямо в аккумулятор — без аллокации временного массива
        for child in unaccompanied_children:
            cx, cy = bbox_center(child.bbox)
            cv2.circle(
                self.accumulator,
                center=(int(cx), int(cy)),
                radius=config.HEATMAP_RADIUS,
                color=1.0,
                thickness=-1
            )

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Наложить heatmap на кадр.

        Args:
            frame: Исходный BGR-кадр.

        Returns:
            Кадр с наложенной тепловой картой.
        """
        max_val = self.accumulator.max()
        if max_val == 0:
            return frame  # Нечего рисовать

        # Нормализуем в диапазон 0-255
        normalized = np.clip(self.accumulator / max_val, 0, 1)
        heatmap_gray = (normalized * 255).astype(np.uint8)

        # Применяем цветовую карту (синий → красный)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

        # Накладываем на кадр с прозрачностью
        # Рисуем heatmap только там, где есть значения (порог > 5%)
        mask = normalized > 0.05
        result = frame.copy()
        result[mask] = cv2.addWeighted(
            frame[mask], 1 - config.HEATMAP_ALPHA,
            heatmap_color[mask], config.HEATMAP_ALPHA,
            0
        )
        return result

    def reset(self):
        """Сбросить heatmap."""
        self.accumulator[:] = 0
