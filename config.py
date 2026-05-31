"""
Конфигурация системы детекции детей без сопровождения.
Все параметры собраны в одном месте для удобства настройки.
"""

# --- Источник видео ---
VIDEO_SOURCE = "data/sample.mp4"    # Путь к файлу или 0 для веб-камеры
OUTPUT_VIDEO = "output/result.mp4"  # Путь для сохранения результата (None — не сохранять)

# --- Детекция ---
# Двухмодельный подход:
#   1. YOLO26 (adult/child) — прямая детекция с классом
#   2. YOLOv8-pose — keypoints для ансамбля (head-body ratio, pose)
YOLO_MODEL = "yolo26n-pose.pt"     # pose-модель: bbox + 17 keypoints скелета
YOLO_CHILD_MODEL = "models/yolo_child_detector.pt"  # YOLO26: 2 класса (adult/child)
USE_YOLO_CHILD_DETECTOR = True     # YOLO26s обучена на adult/child dataset
CONFIDENCE_THRESHOLD = 0.40          # Минимальная уверенность детекции (повышен для отсечения шумовых детекций)
YOLO_IMGSZ = 640                    # Разрешение inference (меньше = быстрее, 416/480/640)
POSE_SKIP_FRAMES = 3                # Запускать pose-модель раз в N кадров в dual-mode
YOLO_DEVICE = "mps"                 # Устройство inference: "mps" (Apple Silicon GPU), "cuda", "cpu"

# --- Трекинг (BoT-SORT + ReID) ---
TRACKER_CONFIG = "botsort_reid.yaml"  # BoT-SORT с ReID для переидентификации

# --- Классификация возраста ---
AGE_CLASSIFIER = "ensemble"         # "ensemble" (все методы) / "pose" / "heuristic"
CHILD_HEIGHT_RATIO = 0.65           # Порог нормализованной высоты (child < ratio * max)
LABEL_HOLD_SEC = 2.0                # Минимум секунд между сменами лейбла child↔adult (антимигание)
MIN_BBOX_HEIGHT = 80                # Минимальная высота bbox — отрезает мусор (стулья, неполные фигуры)
AGE_MODEL_PATH = "models/age_classifier_v2.onnx"  # ML-модель v2: порог 12 лет, video augmentations

# --- Логика сопровождения ---
PROXIMITY_RADIUS_PX = 200           # Макс. расстояние (px) для "сопровождения"
ALERT_THRESHOLD_SEC = 5.0           # Секунд без взрослого до срабатывания алерта

# --- Heatmap ---
HEATMAP_ENABLED = True
HEATMAP_DECAY = 0.995               # Коэффициент затухания за кадр
HEATMAP_RADIUS = 50                 # Радиус пятна на heatmap (px)
HEATMAP_ALPHA = 0.4                 # Прозрачность наложения heatmap

# --- Алерты ---
ALERT_CONSOLE = True                # Вывод алертов в консоль
ALERT_WEBHOOK_URL = None            # URL для отправки алерта (None — отключено)

# --- Отображение ---
SHOW_WINDOW = True                  # Показывать окно с видео
SHOW_DEBUG_SCORES = False           # Показывать debug-скоры (Y26/Rh/Ah/AVG) под bbox
SHOW_FPS_OVERLAY = False            # Показывать FPS поверх кадра (есть в боковой панели)
