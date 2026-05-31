"""Тесты для модуля классификации возраста."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from models import PersonDetection
from age_classifier import HeightRatioClassifier


def make_person(track_id, height, y1=100):
    """Создать тестовый PersonDetection с заданной высотой bbox."""
    return PersonDetection(
        track_id=track_id,
        bbox=(100, y1, 200, y1 + height),
        confidence=0.9,
        bbox_height=height
    )


@pytest.fixture
def classifier():
    return HeightRatioClassifier()


@pytest.fixture
def frame_720p():
    """Пустой кадр 1280x720."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestHeightRatioClassifier:
    """Тесты эвристики по высоте bbox."""

    def test_empty_list(self, classifier, frame_720p):
        """Пустой список — пустой результат."""
        assert classifier.classify([], frame_720p) == []

    def test_child_among_adults(self, classifier, frame_720p):
        """Маленький человек среди больших → ребёнок."""
        persons = [
            make_person(1, 300),   # Взрослый
            make_person(2, 280),   # Взрослый
            make_person(3, 120),   # Ребёнок (значительно ниже)
            make_person(4, 310),   # Взрослый
        ]
        labels = classifier.classify(persons, frame_720p)
        assert labels[2] == "child"
        assert labels[0] == "adult"
        assert labels[1] == "adult"
        assert labels[3] == "adult"

    def test_all_adults_similar_height(self, classifier, frame_720p):
        """Все примерно одинаковой высоты → все взрослые."""
        persons = [
            make_person(1, 300),
            make_person(2, 290),
            make_person(3, 310),
        ]
        labels = classifier.classify(persons, frame_720p)
        assert all(label == "adult" for label in labels)

    def test_single_person_always_adult(self, classifier, frame_720p):
        """Один человек в кадре → всегда взрослый (нет контекста для сравнения)."""
        persons = [make_person(1, 200)]
        labels = classifier.classify(persons, frame_720p)
        assert labels[0] == "adult"

        persons = [make_person(1, 400)]
        labels = classifier.classify(persons, frame_720p)
        assert labels[0] == "adult"

    def test_two_children(self, classifier, frame_720p):
        """Два маленьких человека похожей высоты → оба "взрослые" по медиане.
        Это известное ограничение эвристики."""
        persons = [
            make_person(1, 120),
            make_person(2, 130),
        ]
        labels = classifier.classify(persons, frame_720p)
        # Медиана = 125, оба > 0.7 * 125 = 87.5 → оба "adult"
        # Это ожидаемое поведение эвристики
        assert all(label == "adult" for label in labels)
