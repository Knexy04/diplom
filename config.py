"""
Конфигурация системы детекции детей без сопровождения.
Все параметры собраны в одном месте для удобства настройки.
"""

# --- Источник видео ---
# Поддерживаются: локальный файл (MP4/AVI/MOV), индекс веб-камеры (0),
# сетевая трансляция IP-камеры по RTSP, например:
#   VIDEO_SOURCE = "rtsp://login:pass@192.168.1.10:554/Streaming/Channels/101"
VIDEO_SOURCE = "data/sample.mp4"    # Путь к файлу, 0 для веб-камеры или RTSP-URL
RTSP_RECONNECT_SEC = 3.0            # Пауза перед попыткой переподключения к RTSP при разрыве
OUTPUT_VIDEO = "output/result.mp4"  # Путь для сохранения результата (None — не сохранять)

# --- Детекция ---
# Двухмодельный подход:
#   1. YOLO26 (adult/child) — прямая детекция с классом
#   2. YOLOv8-pose — keypoints для ансамбля (head-body ratio, pose)
YOLO_MODEL = "yolo26n-pose.pt"     # pose-модель: bbox + 17 keypoints скелета
YOLO_CHILD_MODEL = "models/yolo_child_detector.pt"  # YOLO26: 2 класса (adult/child)
USE_YOLO_CHILD_DETECTOR = True     # YOLO26s обучена на adult/child dataset
CONFIDENCE_THRESHOLD = 0.25          # Порог детекции — снижен до 0.25 для лучшего обнаружения детей/дальнего плана
YOLO_IMGSZ = 1280                   # Разрешение inference (1280 = +72% детекций мелких/дальних людей; есть запас FPS после отключения CPU-ReID)
POSE_SKIP_FRAMES = 1                # Child-детектор каждый кадр (есть запас FPS; свежий age-голос YOLO26)
# Авто-выбор устройства: cuda (сервер с GPU) → mps (Apple Silicon) → cpu
def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
YOLO_DEVICE = _auto_device()        # "cuda" на сервере с GPU, "mps" на Mac, иначе "cpu"

# --- Трекинг (BoT-SORT + ReID) ---
TRACKER_CONFIG = "botsort_reid.yaml"  # BoT-SORT с ReID для переидентификации

# --- Классификация возраста ---
AGE_CLASSIFIER = "ensemble"         # "ensemble" (все методы) / "pose" / "heuristic"
AGE_CHILD_THRESHOLD = 0.55          # Порог решения: score < threshold → "child" (0.55 = смещение в сторону ребёнка, пропуск ребёнка критичнее)
CHILD_HEIGHT_RATIO = 0.65           # Порог нормализованной высоты (child < ratio * max)
LABEL_HOLD_SEC = 2.0                # Минимум секунд между сменами лейбла child↔adult (антимигание)
MIN_BBOX_HEIGHT = 40                # Минимальная высота bbox — отсекает только самый мелкий мусор
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
ALERT_JOURNAL_PATH = "alerts.jsonl" # Журнал тревожных событий в формате JSON Lines (None — не вести)

# --- Отображение ---
SHOW_WINDOW = True                  # Показывать окно с видео
SHOW_DEBUG_SCORES = True            # Показывать debug-скоры (Y26/HB/P/M/Rh/Ah/AVG) под bbox для диагностики
SHOW_FPS_OVERLAY = False            # Показывать FPS поверх кадра (есть в боковой панели)
