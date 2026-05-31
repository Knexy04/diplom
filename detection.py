"""
Модуль детекции людей и трекинга.

Два режима работы:
1. Только pose-модель (YOLOv8-pose) — bbox + keypoints + ByteTrack/BoT-SORT
2. Двухмодельный (YOLO26 + pose) — YOLO26 детектирует adult/child напрямую,
   pose-модель даёт keypoints для ансамбля. BoT-SORT + ReID для трекинга.
"""

import os
from ultralytics import YOLO
import numpy as np
import config
from models import PersonDetection


class PersonDetector:
    """
    Детектор людей с поддержкой YOLO26 (adult/child) + pose keypoints.
    BoT-SORT с ReID для устойчивого трекинга.
    """

    def __init__(self):
        # Pose-модель (всегда загружаем — нужна для keypoints)
        self.pose_model = YOLO(config.YOLO_MODEL)

        # YOLO26 child detector (если обучена)
        self.child_model = None
        if config.USE_YOLO_CHILD_DETECTOR and os.path.exists(config.YOLO_CHILD_MODEL):
            self.child_model = YOLO(config.YOLO_CHILD_MODEL)
            print(f"[DETECT] YOLO child detector загружен: {config.YOLO_CHILD_MODEL}")

        # Определяем конфиг трекера
        self.tracker_config = config.TRACKER_CONFIG
        if not os.path.exists(self.tracker_config):
            # Fallback на встроенный
            self.tracker_config = "botsort.yaml"
            print(f"[DETECT] Кастомный трекер не найден, используем: {self.tracker_config}")

        mode = "Pose-track + YOLO26-class" if self.child_model else "Pose only"
        print(f"[DETECT] Режим: {mode}, Трекер: {self.tracker_config}")
        self._frame_idx = 0
        # Кэш для child-модели (запускаем раз в POSE_SKIP_FRAMES кадров)
        self._last_child_boxes: np.ndarray | None = None
        self._last_child_classes: list[int] | None = None
        self._last_child_confs: list[float] | None = None
        self._last_child_class_names: dict | None = None
        # Кэш yolo_class по track_id — сглаживает между запусками child-модели
        self._yolo_class_cache: dict[int, tuple[str, float]] = {}

    def detect_and_track(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        Обнаружить, оценить позу и отследить людей на кадре.

        При USE_YOLO_CHILD_DETECTOR=True:
        - YOLO26 даёт bbox + класс (adult/child) + трекинг с ReID
        - Pose-модель даёт keypoints (без трекинга, матчим по IoU)

        Returns:
            Список PersonDetection с bbox, keypoints, track_id и yolo_class.
        """
        if self.child_model:
            return self._detect_dual_model(frame)
        else:
            return self._detect_pose_only(frame)

    def _detect_pose_only(self, frame: np.ndarray) -> list[PersonDetection]:
        """Оригинальный режим: только pose-модель."""
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=config.YOLO_DEVICE,
            agnostic_nms=True,
            verbose=False
        )[0]

        persons = []
        if results.boxes is None or results.boxes.id is None:
            return persons

        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.int().cpu().tolist()
        confidences = results.boxes.conf.cpu().tolist()

        has_keypoints = hasattr(results, 'keypoints') and results.keypoints is not None
        if has_keypoints:
            all_kpts = results.keypoints.xy.cpu().numpy()
            all_kpts_conf = results.keypoints.conf.cpu().numpy()

        for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
            x1, y1, x2, y2 = box
            height = y2 - y1

            if height < config.MIN_BBOX_HEIGHT:
                continue

            kpts = all_kpts[i] if has_keypoints else None
            kpts_conf = all_kpts_conf[i] if has_keypoints else None

            persons.append(PersonDetection(
                track_id=int(track_id),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                bbox_height=float(height),
                keypoints=kpts,
                keypoints_conf=kpts_conf
            ))

        return persons

    def _detect_dual_model(self, frame: np.ndarray) -> list[PersonDetection]:
        """
        Двухмодельный режим (pose-driven):
        1. Pose — детекция + трекинг + keypoints (один класс 'person', стабильные ID)
        2. YOLO26 — только классификация adult/child, матчим к pose-боксам по IoU.
           Запускается раз в POSE_SKIP_FRAMES кадров, класс кэшируется по track_id.
        """
        self._frame_idx += 1

        # 1. Pose-модель: detect + track + keypoints на каждом кадре
        pose_results = self.pose_model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=config.YOLO_DEVICE,
            agnostic_nms=True,
            verbose=False
        )[0]

        if pose_results.boxes is None or pose_results.boxes.id is None:
            return []

        boxes = pose_results.boxes.xyxy.cpu().numpy()
        track_ids = pose_results.boxes.id.int().cpu().tolist()
        confidences = pose_results.boxes.conf.cpu().tolist()

        has_keypoints = hasattr(pose_results, 'keypoints') and pose_results.keypoints is not None
        all_kpts = pose_results.keypoints.xy.cpu().numpy() if has_keypoints else None
        all_kpts_conf = pose_results.keypoints.conf.cpu().numpy() if has_keypoints else None

        # 2. YOLO26: классификация adult/child раз в POSE_SKIP_FRAMES кадров
        if self._frame_idx % config.POSE_SKIP_FRAMES == 0:
            child_results = self.child_model(
                frame,
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=config.YOLO_IMGSZ,
                device=config.YOLO_DEVICE,
                agnostic_nms=True,
                verbose=False
            )[0]
            if child_results.boxes is not None and len(child_results.boxes) > 0:
                self._last_child_boxes = child_results.boxes.xyxy.cpu().numpy()
                self._last_child_classes = child_results.boxes.cls.int().cpu().tolist()
                self._last_child_confs = child_results.boxes.conf.cpu().tolist()
                self._last_child_class_names = child_results.names
            else:
                self._last_child_boxes = None
                self._last_child_classes = None
                self._last_child_confs = None

        child_boxes = self._last_child_boxes
        child_classes = self._last_child_classes
        child_confs = self._last_child_confs
        class_names = self._last_child_class_names or {}

        # 3. Сборка результата: pose даёт bbox/track_id/keypoints, YOLO26 — yolo_class по IoU
        persons = []
        active_tids = set()
        for i, (box, track_id, conf) in enumerate(zip(boxes, track_ids, confidences)):
            x1, y1, x2, y2 = box
            height = y2 - y1

            if height < config.MIN_BBOX_HEIGHT:
                continue

            tid = int(track_id)
            active_tids.add(tid)

            kpts = all_kpts[i] if has_keypoints else None
            kpts_conf = all_kpts_conf[i] if has_keypoints else None

            # Матчим pose bbox с child-моделью по IoU → получаем yolo_class
            yolo_class = None
            yolo_class_conf = 0.0
            if child_boxes is not None and len(child_boxes) > 0:
                best_iou = 0
                best_idx = -1
                for j, cbox in enumerate(child_boxes):
                    iou = _compute_iou(box, cbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_iou > 0.3 and best_idx >= 0:
                    cls = child_classes[best_idx]
                    yolo_class = class_names.get(cls, None)
                    yolo_class_conf = float(child_confs[best_idx])
                    self._yolo_class_cache[tid] = (yolo_class, yolo_class_conf)

            # Если в этом кадре не нашли — берём из кэша по track_id
            if yolo_class is None and tid in self._yolo_class_cache:
                yolo_class, yolo_class_conf = self._yolo_class_cache[tid]

            persons.append(PersonDetection(
                track_id=tid,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(conf),
                bbox_height=float(height),
                keypoints=kpts,
                keypoints_conf=kpts_conf,
                yolo_class=yolo_class,
                yolo_class_conf=yolo_class_conf
            ))

        # Чистим кэш yolo_class от устаревших треков
        for tid in [t for t in self._yolo_class_cache if t not in active_tids]:
            del self._yolo_class_cache[tid]

        return persons


def _compute_iou(box1, box2) -> float:
    """Вычислить IoU двух bbox (xyxy format)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0
