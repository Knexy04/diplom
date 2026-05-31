"""
Общие структуры данных (dataclasses), используемые во всех модулях.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PersonDetection:
    """Результат детекции одного человека."""
    track_id: int           # Уникальный ID трека
    bbox: tuple             # (x1, y1, x2, y2) — координаты bounding box
    confidence: float       # Уверенность детекции (0..1)
    bbox_height: float      # Высота bbox в пикселях
    keypoints: np.ndarray = field(default=None, repr=False)      # (17, 2) x,y координаты
    keypoints_conf: np.ndarray = field(default=None, repr=False) # (17,) уверенность
    yolo_class: str = None       # Класс от YOLO26: "adult"/"child" (None если не используется)
    yolo_class_conf: float = 0.0  # Confidence детекции от YOLO26
