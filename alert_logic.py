"""
Логика определения "ребёнок без сопровождения взрослого".

Для каждого ребёнка:
1. Ищем ближайшего взрослого (по расстоянию между центрами bbox).
2. Если расстояние > PROXIMITY_RADIUS_PX → ребёнок "без сопровождения".
3. Если ребёнок без сопровождения > ALERT_THRESHOLD_SEC секунд → алерт.
4. Если взрослый подошёл — таймер сбрасывается.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from models import PersonDetection
from utils import bbox_center, bbox_min_distance
import config


@dataclass
class Alert:
    """Информация об алерте."""
    track_id: int           # ID трека ребёнка
    elapsed_sec: float      # Сколько секунд без сопровождения
    status: str             # "NEW" — только сработал, "ONGOING" — продолжается


@dataclass
class ChildState:
    """Состояние отслеживания одного ребёнка."""
    track_id: int
    first_seen_alone: float = None     # Время начала состояния "без сопровождения"
    is_alerted: bool = False           # Алерт уже сработал
    missing_frames: int = 0            # Кадров подряд без детекции (grace period)


class AlertManager:
    """
    Менеджер алертов. Хранит состояние каждого ребёнка
    и определяет, когда нужно генерировать алерт.
    """

    def __init__(self):
        self.states: dict[int, ChildState] = {}

    def update(
        self,
        children: list[PersonDetection],
        adults: list[PersonDetection],
        current_time: float = None,
        frame_idx: int = None
    ) -> list[Alert]:
        """
        Обновить состояния и вернуть список алертов.

        Args:
            children: Список детей в текущем кадре.
            adults: Список взрослых в текущем кадре.
            current_time: Текущее время (по умолчанию time.time()).

        Returns:
            Список Alert для детей, которые без сопровождения.
        """
        if current_time is None:
            current_time = time.time()

        alerts = []
        active_child_ids = set()

        for child in children:
            active_child_ids.add(child.track_id)

            # Расстояние до ближайшего взрослого
            nearest_dist = self._min_distance_to_adults(child, adults)

            # Получаем или создаём состояние для этого ребёнка
            if child.track_id not in self.states:
                self.states[child.track_id] = ChildState(track_id=child.track_id)
            state = self.states[child.track_id]

            if nearest_dist > config.PROXIMITY_RADIUS_PX:
                # Ребёнок без сопровождения
                if state.first_seen_alone is None:
                    state.first_seen_alone = current_time

                elapsed = current_time - state.first_seen_alone

                if elapsed >= config.ALERT_THRESHOLD_SEC:
                    if not state.is_alerted:
                        state.is_alerted = True
                        alerts.append(Alert(child.track_id, elapsed, "NEW"))
                        self._write_journal(child, elapsed, "NEW", frame_idx)
                    else:
                        alerts.append(Alert(child.track_id, elapsed, "ONGOING"))
            else:
                # Взрослый рядом — сбрасываем таймер
                state.first_seen_alone = None
                state.is_alerted = False

        # Grace period: не удаляем сразу, ждём 15 кадров (0.5 сек при 30fps)
        GRACE_FRAMES = 15
        stale_ids = []
        for tid in self.states:
            if tid not in active_child_ids:
                self.states[tid].missing_frames += 1
                if self.states[tid].missing_frames > GRACE_FRAMES:
                    stale_ids.append(tid)
            else:
                self.states[tid].missing_frames = 0
        for tid in stale_ids:
            del self.states[tid]

        return alerts

    def get_child_alone_time(self, track_id: int, current_time: float = None) -> float:
        """Получить время (сек), которое ребёнок без сопровождения. 0 если сопровождён."""
        if current_time is None:
            current_time = time.time()
        state = self.states.get(track_id)
        if state and state.first_seen_alone is not None:
            return current_time - state.first_seen_alone
        return 0.0

    @staticmethod
    def _write_journal(child: PersonDetection, elapsed: float, status: str,
                       frame_idx: int = None):
        """Записать тревожное событие в журнал JSON Lines (ФТ-7).

        Структура записи: временная метка ISO 8601 (UTC), идентификатор трека,
        накопленная длительность отсутствия сопровождения, статус, координаты
        центра bbox и номер кадра. Формат JSONL допускает прямую миграцию в СУБД.
        """
        path = getattr(config, "ALERT_JOURNAL_PATH", None)
        if not path:
            return
        cx, cy = bbox_center(child.bbox)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "track_id": int(child.track_id),
            "duration_sec": round(float(elapsed), 1),
            "status": status,
            "x": int(cx),
            "y": int(cy),
            "frame": int(frame_idx) if frame_idx is not None else None,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Журналирование не должно ронять основной конвейер

    @staticmethod
    def _min_distance_to_adults(child: PersonDetection, adults: list[PersonDetection]) -> float:
        """Минимальное расстояние от ребёнка до ближайшего взрослого (по краям bbox).
        Если боксы пересекаются — расстояние = 0."""
        if not adults:
            return float('inf')
        return min(
            bbox_min_distance(child.bbox, adult.bbox)
            for adult in adults
        )
