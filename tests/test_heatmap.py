"""Тесты для модуля heatmap."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from models import PersonDetection
from heatmap import HeatmapAccumulator


def make_person(track_id, cx, cy):
    """Создать PersonDetection с центром в (cx, cy)."""
    return PersonDetection(
        track_id=track_id,
        bbox=(cx - 25, cy - 50, cx + 25, cy + 50),
        confidence=0.9,
        bbox_height=100
    )


class TestHeatmapAccumulator:

    def test_initial_state(self):
        """Начальный аккумулятор — все нули."""
        hm = HeatmapAccumulator(100, 100)
        assert hm.accumulator.shape == (100, 100)
        assert hm.accumulator.sum() == 0

    def test_update_adds_values(self):
        """После update с ребёнком — ненулевые значения в аккумуляторе."""
        hm = HeatmapAccumulator(200, 200)
        child = make_person(1, 100, 100)
        hm.update([child])
        assert hm.accumulator.sum() > 0
        # Максимум должен быть рядом с центром (100, 100)
        assert hm.accumulator[100, 100] > 0

    def test_decay(self):
        """Decay уменьшает значения каждый кадр."""
        hm = HeatmapAccumulator(200, 200)
        child = make_person(1, 100, 100)
        hm.update([child])
        val_after_update = hm.accumulator.sum()

        # Обновляем без детей — только decay
        hm.update([])
        val_after_decay = hm.accumulator.sum()

        assert val_after_decay < val_after_update

    def test_accumulation(self):
        """Повторные обновления увеличивают суммарное значение."""
        hm = HeatmapAccumulator(200, 200)
        child = make_person(1, 100, 100)

        hm.update([child])
        total1 = hm.accumulator.sum()

        hm.update([child])
        total2 = hm.accumulator.sum()

        # Суммарная энергия растёт (несмотря на decay, новое пятно добавляется)
        assert total2 > total1

    def test_reset(self):
        """Reset обнуляет аккумулятор."""
        hm = HeatmapAccumulator(200, 200)
        child = make_person(1, 100, 100)
        hm.update([child])
        hm.reset()
        assert hm.accumulator.sum() == 0

    def test_render_overlay_empty(self):
        """Рендер пустого heatmap не меняет кадр."""
        hm = HeatmapAccumulator(200, 200)
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = hm.render_overlay(frame)
        np.testing.assert_array_equal(result, frame)

    def test_render_overlay_nonempty(self):
        """Рендер непустого heatmap отличается от исходного кадра."""
        hm = HeatmapAccumulator(200, 200)
        child = make_person(1, 100, 100)
        hm.update([child])

        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        result = hm.render_overlay(frame)

        # Кадр должен измениться в области heatmap
        assert not np.array_equal(result, frame)
