"""
Модуль классификации возраста: ребёнок или взрослый.

Ансамблевый подход — собираем оценки от всех доступных методов:
1. Pose (пропорции тела из keypoints)
2. ML (нейросеть по лицу, если лицо видно)
3. Height (сравнение высоты bbox с учётом перспективы)

Каждый метод возвращает confidence: 0.0 = точно ребёнок, 1.0 = точно взрослый.
Итоговое решение — среднее по всем доступным оценкам + кэш по track_id.
"""

import os
import time
from abc import ABC, abstractmethod
import cv2
import numpy as np
from models import PersonDetection
import config


# Индексы keypoints (COCO format)
NOSE, LEFT_EYE, RIGHT_EYE = 0, 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

MIN_KEYPOINT_CONF = 0.5


class AgeClassifier(ABC):
    @abstractmethod
    def classify(self, persons: list[PersonDetection], frame: np.ndarray) -> list[str]:
        ...


# =============================================================================
# Отдельные оценщики (каждый возвращает confidence 0..1, где 1 = adult)
# =============================================================================

def estimate_pose_confidence(person: PersonDetection) -> float | None:
    """
    Оценка по пропорциям скелета.
    Возвращает 0.0..1.0 (adult confidence) или None если keypoints плохие.
    """
    kpts = person.keypoints
    conf = person.keypoints_conf
    if kpts is None or conf is None:
        return None

    scores = []  # Набираем оценки от каждой метрики (0 = child, 1 = adult)

    # 1. Размер головы / рост
    head = _get_head_size(kpts, conf)
    torso = _segment_len(kpts, conf, [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)])
    legs = _segment_len(kpts, conf, [(LEFT_HIP, LEFT_ANKLE), (RIGHT_HIP, RIGHT_ANKLE)])

    if torso is not None:
        total_h = (head or 0) + torso + (legs or 0)
        if total_h > 10 and head is not None:
            # head_ratio: дети ~0.20-0.28, взрослые ~0.12-0.16
            head_ratio = head / total_h
            # Линейная шкала: 0.28 → 0.0 (child), 0.14 → 1.0 (adult)
            score = np.clip((0.28 - head_ratio) / 0.14, 0.0, 1.0)
            scores.append(score)

        # 2. Ноги / торс
        if legs is not None and torso > 0:
            leg_torso = legs / torso
            # дети ~0.6-1.0, взрослые ~1.1-1.5
            score = np.clip((leg_torso - 0.6) / 0.7, 0.0, 1.0)
            scores.append(score)

    # 3. Ширина плеч / рост
    shoulders = _point_dist(kpts, conf, LEFT_SHOULDER, RIGHT_SHOULDER)
    if shoulders is not None and torso is not None:
        total_h = (head or 0) + torso + (legs or 0)
        if total_h > 10:
            sh_ratio = shoulders / total_h
            # дети ~0.12-0.20, взрослые ~0.20-0.30
            score = np.clip((sh_ratio - 0.12) / 0.14, 0.0, 1.0)
            scores.append(score)

    if not scores:
        return None

    return float(np.mean(scores))


def estimate_ml_confidence(person: PersonDetection, frame: np.ndarray,
                           ml_session, ml_input_name, face_cascade) -> float | None:
    """
    Оценка по лицу через ML-модель.
    Возвращает 0.0..1.0 (adult confidence) или None если лицо не найдено.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [max(0, int(v)) for v in person.bbox]
    x2, y2 = min(w, x2), min(h, y2)
    body_crop = frame[y1:y2, x1:x2]

    if body_crop.size == 0:
        return None

    gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

    if len(faces) == 0:
        return None  # Лицо не видно — не голосуем

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    bh, bw = body_crop.shape[:2]
    pad = int(max(fw, fh) * 0.2)
    fx, fy = max(0, fx - pad), max(0, fy - pad)
    fw = min(bw - fx, fw + 2 * pad)
    fh = min(bh - fy, fh + 2 * pad)
    face = body_crop[fy:fy+fh, fx:fx+fw]

    if face.size == 0:
        return None

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = cv2.resize(face, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = ((img - MEAN) / STD).transpose(2, 0, 1)[np.newaxis, ...]
    out = ml_session.run(None, {ml_input_name: img})[0]
    prob_adult = float(1.0 / (1.0 + np.exp(-out[0, 0])))

    # Debug: сохранение кропов лица
    debug_dir = "debug_faces"
    if os.path.exists(debug_dir):
        fname = f"{debug_dir}/tid{person.track_id}_p{prob_adult:.2f}.jpg"
        cv2.imwrite(fname, face)

    return prob_adult


def estimate_height_confidence(person: PersonDetection, all_persons: list[PersonDetection],
                                frame_h: int) -> float | None:
    """
    Оценка по высоте bbox с учётом перспективы.
    Сравнивает с другими людьми в кадре.
    """
    if len(all_persons) < 2:
        return None  # Один человек — нет контекста для сравнения

    # Нормализация по перспективе
    def norm_height(p):
        _, y1, _, y2 = p.bbox
        center_y = (y1 + y2) / 2
        perspective = 0.5 + (center_y / frame_h)
        return p.bbox_height / perspective

    norm_h = norm_height(person)
    max_norm = max(norm_height(p) for p in all_persons)

    if max_norm == 0:
        return None

    ratio = norm_h / max_norm
    # ratio: дети ~0.50-0.70, взрослые ~0.80-1.0
    # Используем среднюю высоту вместо max для более стабильного сравнения
    all_norms = sorted([norm_height(p) for p in all_persons], reverse=True)
    # Берём медиану верхних 50% как "типичный взрослый"
    top_half = all_norms[:max(1, len(all_norms) // 2)]
    median_norm = sum(top_half) / len(top_half)
    if median_norm > 0:
        ratio_to_median = norm_h / median_norm
        # Если человек ~90%+ от медианы — скорее всего взрослый
        score = np.clip((ratio_to_median - 0.60) / 0.35, 0.0, 1.0)
    else:
        score = np.clip((ratio - 0.55) / 0.35, 0.0, 1.0)
    return float(score)


def estimate_absolute_height_confidence(person: PersonDetection, frame_h: int) -> float:
    """
    Оценка по абсолютной высоте bbox относительно кадра.
    Работает ВСЕГДА, даже когда в кадре только дети.

    Логика: на типичной камере наблюдения взрослый занимает 50-80% кадра,
    ребёнок 20-45%. Если bbox маленький — скорее ребёнок.

    Учитываем Y-позицию: человек внизу кадра (ближе) должен быть крупнее.
    """
    _, y1, _, y2 = person.bbox
    center_y = (y1 + y2) / 2

    # Нормализуем высоту bbox по перспективе
    perspective = 0.5 + (center_y / frame_h)
    norm_ratio = (person.bbox_height / frame_h) / perspective

    # Если bbox занимает >60% кадра — человек слишком близко, высота ненадёжна
    bbox_fill = person.bbox_height / frame_h
    if bbox_fill > 0.6:
        return None

    # norm_ratio: на далёких камерах наблюдения значения ниже
    # дети ~0.05-0.15, взрослые ~0.15-0.40+
    # Линейная шкала: 0.07 → 0.0 (child), 0.30 → 1.0 (adult)
    score = np.clip((norm_ratio - 0.07) / 0.23, 0.0, 1.0)
    return float(score)


def estimate_head_body_ratio_confidence(person: PersonDetection) -> float | None:
    """
    Оценка по соотношению головы к полному росту.
    Ключевое преимущество: РАБОТАЕТ НЕЗАВИСИМО ОТ ПОЗЫ (сидит, стоит, лежит).

    Пропорции (тело/голова):
    - Младенец:  ~4.0
    - 2 года:    ~5.5
    - 5 лет:     ~5.5
    - 8 лет:     ~6.0
    - Взрослый:  ~7.0-7.5

    Возвращает 0.0..1.0 (0 = child, 1 = adult) или None.
    """
    kpts = person.keypoints
    conf = person.keypoints_conf
    if kpts is None or conf is None:
        return None

    # Оцениваем размер головы по keypoints (нос, глаза, уши)
    head_points = []
    for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
        if conf[idx] > MIN_KEYPOINT_CONF:
            head_points.append(kpts[idx])

    if len(head_points) < 2:
        return None

    head_points = np.array(head_points)
    head_min_y = head_points[:, 1].min()
    head_max_y = head_points[:, 1].max()
    head_width = head_points[:, 0].max() - head_points[:, 0].min()

    # Высота головы ≈ max(ширина головы, вертикальный размер * 1.5)
    # Голова примерно круглая, но keypoints покрывают только верхнюю часть
    head_height = max(head_width * 1.2, (head_max_y - head_min_y) * 2.0)

    if head_height < 5:  # Слишком маленькая — ненадёжно
        return None

    # Полная высота тела = высота bbox
    body_height = person.bbox_height
    if body_height < 20:
        return None

    # Соотношение тело/голова
    ratio = body_height / head_height

    # Рекалибровка: метод систематически недооценивает голову (особенно на
    # дистанции), из-за чего у детей ratio выходит ~7 и читался как взрослый.
    # Сдвигаем центр: 6.0 → 0.0 (child), 8.0 → 1.0 (adult).
    score = np.clip((ratio - 6.0) / 2.0, 0.0, 1.0)
    return float(score)


def is_person_sitting(person: PersonDetection) -> bool:
    """
    Определяет, сидит ли человек.
    Если сидит — height-based методы ненадёжны.
    """
    kpts = person.keypoints
    conf = person.keypoints_conf
    if kpts is None or conf is None:
        return False

    # Проверяем по соотношению ширины и высоты bbox
    x1, y1, x2, y2 = person.bbox
    w = x2 - x1
    h = y2 - y1
    if h < 10:
        return False
    aspect = w / h
    # Сидящий человек: bbox шире чем обычно (aspect > 0.7)
    if aspect > 0.75:
        return True

    # Проверяем по keypoints: если бёдра и колени примерно на одной высоте
    has_hip = conf[LEFT_HIP] > MIN_KEYPOINT_CONF or conf[RIGHT_HIP] > MIN_KEYPOINT_CONF
    has_ankle = conf[LEFT_ANKLE] > MIN_KEYPOINT_CONF or conf[RIGHT_ANKLE] > MIN_KEYPOINT_CONF

    if has_hip and has_ankle:
        hip_y = np.mean([kpts[i][1] for i in [LEFT_HIP, RIGHT_HIP] if conf[i] > MIN_KEYPOINT_CONF])
        ankle_y = np.mean([kpts[i][1] for i in [LEFT_ANKLE, RIGHT_ANKLE] if conf[i] > MIN_KEYPOINT_CONF])
        # Если ноги очень короткие относительно тела — сидит
        leg_len = ankle_y - hip_y
        torso_len = hip_y - y1
        if torso_len > 0 and leg_len / torso_len < 0.4:
            return True

    return False


# =============================================================================
# Вспомогательные функции для pose
# =============================================================================

def _get_head_size(kpts, conf):
    d = _point_dist(kpts, conf, LEFT_EAR, RIGHT_EAR)
    if d is not None:
        return d * 1.2
    d = _point_dist(kpts, conf, LEFT_EYE, RIGHT_EYE)
    if d is not None:
        return d * 2.5
    return None

def _segment_len(kpts, conf, pairs):
    best = None
    for a, b in pairs:
        if conf[a] > MIN_KEYPOINT_CONF and conf[b] > MIN_KEYPOINT_CONF:
            length = abs(kpts[b][1] - kpts[a][1])
            if best is None or length > best:
                best = length
    return best

def _point_dist(kpts, conf, a, b):
    if conf[a] > MIN_KEYPOINT_CONF and conf[b] > MIN_KEYPOINT_CONF:
        return float(np.linalg.norm(kpts[a] - kpts[b]))
    return None


# =============================================================================
# Ансамблевый классификатор
# =============================================================================

class EnsembleAgeClassifier(AgeClassifier):
    """
    Собирает оценки от всех доступных методов, усредняет и принимает решение.

    Методы:
    - Pose (пропорции скелета) — вес 2 (самый надёжный, не зависит от лица/расстояния)
    - ML (нейросеть по лицу) — вес 1 (только когда лицо видно)
    - Height (размер bbox) — вес 1 (только когда 2+ человека в кадре)

    Итоговый score = взвешенное среднее.
    score < 0.5 → "child", иначе "adult".

    Результат кэшируется по track_id для стабильности.
    """

    # Главные — обученные модели (Y26 + лицо): они различают возраст по внешности
    # и надёжны там, где геометрия ломается (близкий план, дальний план, одиночка).
    # Геометрические эвристики (HB/pose/высоты) сильно врут на мелких/крупных детях
    # (дают ложного «взрослого»), поэтому понижены до слабых довесков.
    YOLO_WEIGHT = 2.5         # YOLO26 direct — обученный детектор adult/child, главный голос
    ML_WEIGHT = 2.5         # ML по лицу — самый точный, когда лицо видно
    HEAD_BODY_WEIGHT = 1.5    # head-to-body — недооценивает голову на дистанции → ложный adult
    POSE_WEIGHT = 1.5         # пропорции скелета — ненадёжны в близком/дальнем плане
    HEIGHT_REL_WEIGHT = 1.0   # относительная высота — врёт при малом числе людей/перспективе
    HEIGHT_ABS_WEIGHT = 1.0   # абсолютная высота — самый слабый метод

    def __init__(self):
        # ML-модель (опционально)
        self.ml_session = None
        self.ml_input_name = None
        self.face_cascade = None

        ml_path = config.AGE_MODEL_PATH
        if os.path.exists(ml_path):
            try:
                import onnxruntime as ort
                self.ml_session = ort.InferenceSession(ml_path)
                self.ml_input_name = self.ml_session.get_inputs()[0].name
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                print(f"[ENSEMBLE] ML-модель загружена: {ml_path}")
            except Exception as e:
                print(f"[ENSEMBLE] ML-модель недоступна: {e}")

        # Кэш: {track_id: list[float]} — последние N оценок
        self.track_cache: dict[int, list[float]] = {}
        self.cache_max_size = 10  # Храним последние 10 оценок

        # Антимигание: {track_id: (label, set_at_time)}
        self.label_state: dict[int, tuple[str, float]] = {}

        methods = ["pose", "abs_height"]
        if self.ml_session:
            methods.append("ml")
        methods.append("rel_height")
        print(f"[ENSEMBLE] Методы: {', '.join(methods)}")

        # Хранилище последних debug-score для отображения
        self.debug_scores: dict[int, dict] = {}

    def classify(self, persons: list[PersonDetection], frame: np.ndarray) -> list[str]:
        if not persons:
            return []

        frame_h = frame.shape[0]
        labels = []
        active_ids = set()
        scores_for_group = []  # (track_id, avg_score) для группового контекста

        for person in persons:
            active_ids.add(person.track_id)

            # Собираем оценки от каждого метода
            votes = []  # (confidence, weight)
            debug = {}
            # Если человек сидит — height/HB-методы дают ложный «ребёнок»,
            # отключаем их. Остаются YOLO26 + Pose + ML, плюс «adult-prior»
            # (сидящие в общественных пространствах статистически чаще взрослые).
            sitting = is_person_sitting(person)
            if sitting:
                debug["sit"] = 1
                votes.append((0.70, 1.5))  # adult-prior score 0.7, вес 1.5
                debug["sitP"] = 0.70

            # Обрезан нижней границей кадра (ног не видно) — высотные и
            # пропорциональные методы дают ложного «ребёнка». Определяем по
            # связке: bbox упирается в низ кадра И щиколотки не детектированы.
            # Для таких людей полагаемся на YOLO26-direct и ML по лицу.
            _, _, _, y2b = person.bbox
            ankles_seen = (person.keypoints_conf is not None and
                           (person.keypoints_conf[LEFT_ANKLE] > 0.3 or
                            person.keypoints_conf[RIGHT_ANKLE] > 0.3))
            truncated = (y2b >= frame_h - 3) and not ankles_seen
            if truncated:
                debug["cut"] = 1

            # 0. YOLO26 direct detection (если доступен)
            if person.yolo_class is not None:
                if person.yolo_class == "adult":
                    yolo_score = 0.5 + person.yolo_class_conf * 0.5  # 0.5..1.0
                else:
                    yolo_score = 0.5 - person.yolo_class_conf * 0.5  # 0.0..0.5
                conf_factor = min(1.0, person.yolo_class_conf / 0.6)
                yolo_w = self.YOLO_WEIGHT * conf_factor
                votes.append((yolo_score, yolo_w))
                debug["Y26"] = yolo_score

            # 1. Head-to-body ratio (только для стоящих, не обрезанных)
            if not sitting and not truncated:
                hb_conf = estimate_head_body_ratio_confidence(person)
                if hb_conf is not None:
                    votes.append((hb_conf, self.HEAD_BODY_WEIGHT))
                    debug["HB"] = hb_conf

            # 2. Pose (пропорции скелета) — ненадёжна на обрезанном скелете
            if not truncated:
                pose_conf = estimate_pose_confidence(person)
                if pose_conf is not None:
                    votes.append((pose_conf, self.POSE_WEIGHT))
                    debug["P"] = pose_conf

            # 3. ML (лицо)
            if self.ml_session is not None:
                ml_conf = estimate_ml_confidence(
                    person, frame, self.ml_session, self.ml_input_name, self.face_cascade
                )
                if ml_conf is not None:
                    votes.append((ml_conf, self.ML_WEIGHT))
                    debug["M"] = ml_conf

            # 4-5. Высотные методы — только для стоящих. У сидящего bbox
            # короче из-за позы, не из-за возраста; иначе сидящих взрослых
            # классифицировали бы как детей.
            if not sitting and not truncated:
                rel_conf = estimate_height_confidence(person, persons, frame_h)
                if rel_conf is not None:
                    votes.append((rel_conf, self.HEIGHT_REL_WEIGHT))
                    debug["Rh"] = rel_conf

                abs_conf = estimate_absolute_height_confidence(person, frame_h)
                if abs_conf is not None:
                    votes.append((abs_conf, self.HEIGHT_ABS_WEIGHT))
                    debug["Ah"] = abs_conf

            # Взвешенное среднее
            if votes:
                total_weight = sum(w for _, w in votes)
                avg_score = sum(c * w for c, w in votes) / total_weight
                scores_for_group.append((person.track_id, avg_score))
                debug["AVG"] = avg_score

            self.debug_scores[person.track_id] = debug

        # Групповая коррекция: если >4 людей и >80% помечены как "child",
        # модель скорее всего ошибается — поднимаем всем скоры
        if len(scores_for_group) >= 4:
            child_count = sum(1 for _, s in scores_for_group if s < config.AGE_CHILD_THRESHOLD)
            child_ratio = child_count / len(scores_for_group)
            if child_ratio > 0.75:
                # Маловероятно что >75% людей в ТЦ — дети.
                # Сдвигаем всех к "adult": score = score * 0.5 + 0.35
                corrected = []
                for tid, score in scores_for_group:
                    new_score = score * 0.5 + 0.35
                    corrected.append((tid, new_score))
                    if tid in self.debug_scores:
                        self.debug_scores[tid]["GC"] = 1  # group correction applied
                        self.debug_scores[tid]["AVG"] = new_score
                scores_for_group = corrected

        # Кэшируем скоры
        for tid, score in scores_for_group:
            self._add_to_cache(tid, score)

        # Решение из кэша (сглаженное) с антимиганием
        for person in persons:
            active_ids.add(person.track_id)
            labels.append(self._get_label(person.track_id))

        # Очистка
        for tid in [t for t in self.track_cache if t not in active_ids]:
            del self.track_cache[tid]
        for tid in [t for t in self.label_state if t not in active_ids]:
            del self.label_state[tid]

        return labels

    def _add_to_cache(self, track_id: int, score: float):
        """Добавить оценку в кэш (скользящее окно)."""
        if track_id not in self.track_cache:
            self.track_cache[track_id] = []
        cache = self.track_cache[track_id]
        cache.append(score)
        if len(cache) > self.cache_max_size:
            cache.pop(0)

    def _get_label(self, track_id: int) -> str:
        """Получить метку из кэша (среднее по истории) + антимигание.
        Лейбл не меняется чаще раза в LABEL_HOLD_SEC секунд.
        """
        # Что предлагает текущий score?
        thr = config.AGE_CHILD_THRESHOLD
        cache = self.track_cache.get(track_id)
        if cache:
            avg = sum(cache) / len(cache)
            proposed = "child" if avg < thr else "adult"
        else:
            debug = self.debug_scores.get(track_id, {})
            if "AVG" in debug:
                proposed = "child" if debug["AVG"] < thr else "adult"
            else:
                proposed = "child"  # safe default

        now = time.time()
        prev = self.label_state.get(track_id)
        if prev is None:
            # Первый раз видим трек — фиксируем лейбл
            self.label_state[track_id] = (proposed, now)
            return proposed

        prev_label, set_at = prev
        if proposed == prev_label:
            return prev_label
        # Лейбл хочет смениться — разрешаем только если прошло >= LABEL_HOLD_SEC
        if now - set_at >= config.LABEL_HOLD_SEC:
            self.label_state[track_id] = (proposed, now)
            return proposed
        return prev_label


# =============================================================================
# Простые классификаторы (для совместимости / отладки)
# =============================================================================

class HeightRatioClassifier(AgeClassifier):
    """Только эвристика по высоте bbox."""
    def classify(self, persons, frame):
        if not persons or len(persons) < 2:
            return ["adult"] * len(persons)
        frame_h = frame.shape[0]
        labels = []
        for p in persons:
            conf = estimate_height_confidence(p, persons, frame_h)
            labels.append("child" if conf is not None and conf < 0.5 else "adult")
        return labels


class PoseAgeClassifier(AgeClassifier):
    """Pose (пропорции скелета). Если keypoints плохие — fallback на абсолютную высоту bbox."""
    def __init__(self):
        self.cache = {}
    def classify(self, persons, frame):
        if not persons:
            return []
        frame_h = frame.shape[0]
        labels = []
        active = set()
        for p in persons:
            active.add(p.track_id)
            conf = estimate_pose_confidence(p)
            if conf is None:
                # Fallback: keypoints ненадёжны — используем абсолютную высоту bbox
                conf = estimate_absolute_height_confidence(p, frame_h)
            if conf is not None:
                if p.track_id not in self.cache:
                    self.cache[p.track_id] = []
                self.cache[p.track_id].append(conf)
            c = self.cache.get(p.track_id)
            if c:
                labels.append("child" if sum(c)/len(c) < 0.5 else "adult")
            else:
                labels.append("adult")
        for t in [t for t in self.cache if t not in active]:
            del self.cache[t]
        return labels


# =============================================================================
# Фабрика
# =============================================================================

def create_age_classifier() -> AgeClassifier:
    """Создать классификатор по настройке в config.py."""
    if config.AGE_CLASSIFIER == "ensemble":
        return EnsembleAgeClassifier()
    elif config.AGE_CLASSIFIER == "pose":
        return PoseAgeClassifier()
    elif config.AGE_CLASSIFIER == "heuristic":
        return HeightRatioClassifier()
    # По умолчанию — ensemble
    return EnsembleAgeClassifier()
