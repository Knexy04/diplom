"""
Главный модуль системы детекции детей без сопровождения взрослых.

Пайплайн обработки каждого кадра:
  Видео → Детекция + Трекинг → Классификация возраста →
  → Проверка сопровождения → Heatmap → Визуализация → Вывод

Запуск:
  python main.py                          # Видео из config.py
  python main.py --source data/test.mp4   # Указать видео
  python main.py --source 0              # Веб-камера
  python main.py --threshold 3.0         # Порог алерта (сек)
  python main.py --radius 150            # Радиус сопровождения (px)
"""

import argparse
import time
import cv2
import config
from detection import PersonDetector
from age_classifier import create_age_classifier
from alert_logic import AlertManager
from heatmap import HeatmapAccumulator
from visualization import draw_persons, draw_alert_banner, draw_fps
from utils import FPSCounter, send_webhook_alert


def parse_args():
    """Разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Система детекции детей без сопровождения взрослых"
    )
    parser.add_argument(
        "--source", default=None,
        help="Источник видео: путь к файлу или 0 для веб-камеры"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help=f"Порог алерта в секундах (по умолчанию {config.ALERT_THRESHOLD_SEC})"
    )
    parser.add_argument(
        "--radius", type=int, default=None,
        help=f"Радиус сопровождения в пикселях (по умолчанию {config.PROXIMITY_RADIUS_PX})"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Не показывать окно (только запись в файл)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Путь для сохранения результата"
    )
    return parser.parse_args()


def apply_args(args):
    """Применить аргументы командной строки к конфигурации."""
    if args.source is not None:
        # Если передано число — это индекс камеры
        try:
            config.VIDEO_SOURCE = int(args.source)
        except ValueError:
            config.VIDEO_SOURCE = args.source

    if args.threshold is not None:
        config.ALERT_THRESHOLD_SEC = args.threshold

    if args.radius is not None:
        config.PROXIMITY_RADIUS_PX = args.radius

    if args.no_display:
        config.SHOW_WINDOW = False

    if args.output is not None:
        config.OUTPUT_VIDEO = args.output


def main():
    """Основной цикл обработки видео."""
    args = parse_args()
    apply_args(args)

    # --- Инициализация компонентов ---
    print(f"[INIT] Загрузка модели {config.YOLO_MODEL}...")
    detector = PersonDetector()

    classifier = create_age_classifier()
    print(f"[INIT] Классификатор: {type(classifier).__name__}")

    alert_manager = AlertManager()
    fps_counter = FPSCounter()

    # --- Открытие видео ---
    print(f"[INIT] Источник видео: {config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {config.VIDEO_SOURCE}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"[INIT] Разрешение: {frame_width}x{frame_height}, FPS видео: {video_fps:.0f}")

    # Heatmap
    heatmap_acc = None
    if config.HEATMAP_ENABLED:
        heatmap_acc = HeatmapAccumulator(frame_width, frame_height)

    # Запись видео
    video_writer = None
    if config.OUTPUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            config.OUTPUT_VIDEO, fourcc, video_fps,
            (frame_width, frame_height)
        )
        print(f"[INIT] Запись в: {config.OUTPUT_VIDEO}")

    print("[START] Обработка видео... (нажмите 'q' для выхода)")
    print(f"[CONFIG] Радиус сопровождения: {config.PROXIMITY_RADIUS_PX}px, "
          f"Порог алерта: {config.ALERT_THRESHOLD_SEC}s")
    print("-" * 60)

    frame_count = 0

    try:
        is_rtsp = isinstance(config.VIDEO_SOURCE, str) and \
            config.VIDEO_SOURCE.lower().startswith("rtsp://")

        while True:
            ret, frame = cap.read()
            if not ret:
                # RTSP-поток мог кратковременно разорваться — пробуем переподключиться
                if is_rtsp:
                    print(f"[RTSP] Поток прерван, переподключение через "
                          f"{config.RTSP_RECONNECT_SEC}s...")
                    cap.release()
                    time.sleep(config.RTSP_RECONNECT_SEC)
                    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
                    if cap.isOpened():
                        continue
                break

            frame_count += 1
            current_time = time.time()
            fps_counter.tick()

            # --- 1. Детекция + Трекинг ---
            persons = detector.detect_and_track(frame)

            # --- 2. Классификация возраста ---
            age_labels = classifier.classify(persons, frame)

            # --- 3. Разделение на детей и взрослых ---
            children = []
            adults = []
            for person, label in zip(persons, age_labels):
                if label == "child":
                    children.append(person)
                else:
                    adults.append(person)

            # --- 4. Логика сопровождения и алертов ---
            alerts = alert_manager.update(children, adults, current_time, frame_idx=frame_count)

            # Собираем информацию для визуализации
            alone_times = {}
            alerted_ids = set()
            unaccompanied_children = []

            for child in children:
                alone_time = alert_manager.get_child_alone_time(child.track_id, current_time)
                if alone_time > 0:
                    alone_times[child.track_id] = alone_time
                    unaccompanied_children.append(child)
                if alert_manager.states.get(child.track_id, None) and \
                   alert_manager.states[child.track_id].is_alerted:
                    alerted_ids.add(child.track_id)

            # --- 5. Консольные алерты ---
            for alert in alerts:
                if alert.status == "NEW" and config.ALERT_CONSOLE:
                    print(f"[ALERT] Ребёнок #{alert.track_id} без сопровождения "
                          f"уже {alert.elapsed_sec:.1f} сек!")
                    send_webhook_alert(alert.track_id, alert.elapsed_sec)

            # --- 6. Heatmap ---
            if heatmap_acc and unaccompanied_children:
                heatmap_acc.update(unaccompanied_children)
            elif heatmap_acc:
                # Только decay, без новых точек
                heatmap_acc.accumulator *= config.HEATMAP_DECAY

            # --- 7. Визуализация ---
            # Наложение heatmap (под bbox)
            if heatmap_acc:
                frame = heatmap_acc.render_overlay(frame)

            # Рисуем bbox и метки
            debug_scores = getattr(classifier, 'debug_scores', None) if config.SHOW_DEBUG_SCORES else None
            frame = draw_persons(frame, persons, age_labels, alone_times, alerted_ids, debug_scores)

            # Баннер алертов
            frame = draw_alert_banner(frame, len(alerted_ids))

            # FPS
            if config.SHOW_FPS_OVERLAY:
                frame = draw_fps(frame, fps_counter.get_fps())

            # --- 8. Вывод ---
            if config.SHOW_WINDOW:
                cv2.imshow("Child Safety Monitor", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[STOP] Остановлено пользователем.")
                    break

            if video_writer:
                video_writer.write(frame)

    except KeyboardInterrupt:
        print("\n[STOP] Прервано (Ctrl+C).")

    finally:
        # Освобождаем ресурсы
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print("-" * 60)
        print(f"[DONE] Обработано кадров: {frame_count}")
        if config.OUTPUT_VIDEO and video_writer:
            print(f"[DONE] Результат сохранён: {config.OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
