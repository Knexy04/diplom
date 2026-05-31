"""Тесты для модуля логики алертов."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import PersonDetection
from alert_logic import AlertManager


def make_person(track_id, cx, cy, height=200):
    """Создать PersonDetection с центром в (cx, cy)."""
    half_w = 50
    half_h = height / 2
    return PersonDetection(
        track_id=track_id,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=0.9,
        bbox_height=height
    )


class TestAlertManager:

    def test_no_alert_with_adult_nearby(self):
        """Ребёнок рядом со взрослым → нет алерта."""
        manager = AlertManager()
        child = make_person(1, 100, 100)
        adult = make_person(2, 150, 100)  # 50px расстояние

        # Прогоняем 10 секунд
        for t in range(100):
            alerts = manager.update([child], [adult], current_time=t * 0.1)
            assert len(alerts) == 0

    def test_alert_after_threshold(self):
        """Ребёнок один → алерт через ALERT_THRESHOLD_SEC."""
        manager = AlertManager()
        child = make_person(1, 100, 100)

        # До порога — нет алертов
        alerts = manager.update([child], [], current_time=0.0)
        assert len(alerts) == 0

        alerts = manager.update([child], [], current_time=4.0)
        assert len(alerts) == 0

        # После порога (5 сек) — алерт NEW
        alerts = manager.update([child], [], current_time=5.5)
        assert len(alerts) == 1
        assert alerts[0].status == "NEW"
        assert alerts[0].track_id == 1

    def test_alert_ongoing(self):
        """После первого алерта — статус ONGOING."""
        manager = AlertManager()
        child = make_person(1, 100, 100)

        manager.update([child], [], current_time=0.0)
        manager.update([child], [], current_time=6.0)  # NEW

        alerts = manager.update([child], [], current_time=7.0)
        assert len(alerts) == 1
        assert alerts[0].status == "ONGOING"

    def test_alert_reset_when_adult_approaches(self):
        """Взрослый подошёл — таймер сбрасывается."""
        manager = AlertManager()
        child = make_person(1, 100, 100)
        adult = make_person(2, 150, 100)  # Рядом

        # 4 секунды один
        manager.update([child], [], current_time=0.0)
        manager.update([child], [], current_time=4.0)

        # Взрослый подошёл — сброс
        alerts = manager.update([child], [adult], current_time=4.5)
        assert len(alerts) == 0

        # Снова один — таймер начинается заново
        alerts = manager.update([child], [], current_time=5.0)
        assert len(alerts) == 0  # Ещё не прошло 5 сек с момента сброса

    def test_adult_too_far(self):
        """Взрослый далеко (> PROXIMITY_RADIUS_PX) — считается "без сопровождения"."""
        manager = AlertManager()
        child = make_person(1, 100, 100)
        adult = make_person(2, 500, 100)  # 400px > 200px радиуса

        manager.update([child], [adult], current_time=0.0)
        manager.update([child], [adult], current_time=6.0)

        alerts = manager.update([child], [adult], current_time=6.0)
        assert len(alerts) == 1

    def test_cleanup_stale_tracks(self):
        """Исчезнувшие дети удаляются из state."""
        manager = AlertManager()
        child = make_person(1, 100, 100)

        manager.update([child], [], current_time=0.0)
        assert 1 in manager.states

        # Ребёнок исчез из кадра
        manager.update([], [], current_time=1.0)
        assert 1 not in manager.states

    def test_get_child_alone_time(self):
        """Проверка get_child_alone_time."""
        manager = AlertManager()
        child = make_person(1, 100, 100)

        manager.update([child], [], current_time=10.0)
        alone = manager.get_child_alone_time(1, current_time=13.0)
        assert abs(alone - 3.0) < 0.01

    def test_no_persons(self):
        """Пустой кадр → нет алертов."""
        manager = AlertManager()
        alerts = manager.update([], [], current_time=0.0)
        assert len(alerts) == 0
