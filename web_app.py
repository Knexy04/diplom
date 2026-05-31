"""
Веб-интерфейс системы детекции детей без сопровождения.
Запуск: streamlit run web_app.py
"""

import time
import tempfile
import cv2
import streamlit as st
import numpy as np

import config
from detection import PersonDetector
from age_classifier import create_age_classifier
from alert_logic import AlertManager
from heatmap import HeatmapAccumulator
from visualization import draw_persons, draw_alert_banner, draw_fps
from utils import FPSCounter


# --- Настройка страницы ---
st.set_page_config(
    page_title="Child Safety Monitor",
    page_icon="👶",
    layout="wide"
)

st.title("Система детекции детей без сопровождения")
st.caption("Дипломный проект — компьютерное зрение для безопасности детей")


# --- Сайдбар: настройки ---
st.sidebar.header("Настройки")

source_type = st.sidebar.radio(
    "Источник видео",
    ["Загрузить файл", "Веб-камера", "Файл из data/"]
)

proximity_radius = st.sidebar.slider(
    "Радиус сопровождения (px)", 50, 500, config.PROXIMITY_RADIUS_PX, step=10,
    help="Максимальное расстояние между ребёнком и взрослым для считания 'сопровождённым'"
)

alert_threshold = st.sidebar.slider(
    "Порог алерта (сек)", 1.0, 30.0, config.ALERT_THRESHOLD_SEC, step=0.5,
    help="Сколько секунд ребёнок должен быть один, чтобы сработал алерт"
)

classifier_type = "ensemble"  # ensemble — единственный надёжный режим

show_heatmap = st.sidebar.checkbox("Показывать heatmap", value=True)
show_debug = st.sidebar.checkbox("Debug-скоры (Y26/Rh/Ah/AVG)", value=False)

# Применяем настройки
config.PROXIMITY_RADIUS_PX = proximity_radius
config.ALERT_THRESHOLD_SEC = alert_threshold
config.AGE_CLASSIFIER = classifier_type
config.HEATMAP_ENABLED = show_heatmap
config.SHOW_DEBUG_SCORES = show_debug


# --- Получение источника видео ---
video_path = None

if source_type == "Загрузить файл":
    uploaded = st.sidebar.file_uploader(
        "Выберите видео", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.flush()
        video_path = tmp.name

elif source_type == "Веб-камера":
    video_path = 0

elif source_type == "Файл из data/":
    import glob, os
    video_files = glob.glob("data/*.mp4") + glob.glob("data/*.avi") + glob.glob("data/*.mov")
    if video_files:
        video_path = st.sidebar.selectbox("Файл", video_files)
    else:
        st.sidebar.warning("В папке data/ нет видеофайлов")


# --- Получаем инфо о видео для слайдера перемотки ---
video_info = {}
if video_path is not None and video_path != 0:
    _cap = cv2.VideoCapture(video_path)
    if _cap.isOpened():
        video_info["total_frames"] = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_info["fps"] = _cap.get(cv2.CAP_PROP_FPS) or 30
        video_info["duration"] = video_info["total_frames"] / video_info["fps"]
    _cap.release()


# --- Сайдбар: управление воспроизведением ---
st.sidebar.header("Воспроизведение")

start_sec = 0.0
if video_info.get("duration", 0) > 0:
    start_sec = st.sidebar.slider(
        "Начать с (сек)", 0.0, video_info["duration"],
        0.0, step=0.5,
        format="%.1f сек"
    )

frame_skip = st.sidebar.selectbox(
    "Скорость обработки",
    [("Каждый кадр", 1), ("Каждый 2-й", 2), ("Каждый 3-й", 3), ("Каждый 5-й", 5)],
    format_func=lambda x: x[0]
)
skip_n = frame_skip[1]


# --- Основной интерфейс ---
col_video, col_info = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    # Слайдер перемотки — обновляется во время воспроизведения
    seek_placeholder = st.empty()
    progress_bar = st.empty()

with col_info:
    st.subheader("Статистика")
    fps_display = st.empty()
    persons_display = st.empty()
    children_display = st.empty()
    adults_display = st.empty()
    time_display = st.empty()

    st.subheader("Алерты")
    alerts_container = st.container(height=300)


# --- Кнопки управления ---
col_start, col_stop = st.columns(2)

with col_start:
    start_btn = st.button("▶ Запустить", type="primary", use_container_width=True)

with col_stop:
    stop_btn = st.button("⏹ Остановить", use_container_width=True)


# --- Обработка видео ---
if start_btn and video_path is not None:
    # Инициализация
    with st.spinner("Загрузка модели..."):
        detector = PersonDetector()
        classifier = create_age_classifier()

    alert_manager = AlertManager()
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Не удалось открыть видео: {video_path}")
        st.stop()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Ресайз для обработки: ограничиваем ширину 640px (быстрее YOLO + меньше трафик Streamlit)
    process_width = min(frame_width, 640)
    scale = process_width / frame_width
    process_height = int(frame_height * scale)

    # Перемотка к стартовой позиции
    if start_sec > 0:
        start_frame = int(start_sec * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    heatmap_acc = None
    if config.HEATMAP_ENABLED:
        heatmap_acc = HeatmapAccumulator(process_width, process_height)

    alert_log = []
    frame_count = 0
    ui_update_every = 3  # Обновлять метрики раз в N обработанных кадров
    ui_frame_count = 0
    last_display_time = 0.0
    display_interval = 0.1  # Обновлять видео в UI не чаще 10 раз в секунду

    # Основной цикл
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_count += 1

        # Пропуск кадров для ускорения
        if skip_n > 1 and frame_count % skip_n != 0:
            continue

        current_time = time.time()
        fps_counter.tick()
        ui_frame_count += 1

        # Ресайз до рабочего разрешения
        if scale < 1.0:
            frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_LINEAR)

        # --- Пайплайн ---
        persons = detector.detect_and_track(frame)
        age_labels = classifier.classify(persons, frame)

        children = []
        adults = []
        for person, label in zip(persons, age_labels):
            if label == "child":
                children.append(person)
            else:
                adults.append(person)

        alerts = alert_manager.update(children, adults, current_time)

        alone_times = {}
        alerted_ids = set()
        unaccompanied_children = []

        for child in children:
            alone_time = alert_manager.get_child_alone_time(child.track_id, current_time)
            if alone_time > 0:
                alone_times[child.track_id] = alone_time
                unaccompanied_children.append(child)
            state = alert_manager.states.get(child.track_id)
            if state and state.is_alerted:
                alerted_ids.add(child.track_id)

        # Алерты
        for alert in alerts:
            if alert.status == "NEW":
                alert_log.append(
                    f"🔴 **Ребёнок #{alert.track_id}** без сопровождения "
                    f"({alert.elapsed_sec:.1f} сек)"
                )

        # Heatmap
        if heatmap_acc and unaccompanied_children:
            heatmap_acc.update(unaccompanied_children)
        elif heatmap_acc:
            heatmap_acc.accumulator *= config.HEATMAP_DECAY

        # Визуализация
        if heatmap_acc:
            frame = heatmap_acc.render_overlay(frame)
        debug_scores = getattr(classifier, 'debug_scores', None) if config.SHOW_DEBUG_SCORES else None
        frame = draw_persons(frame, persons, age_labels, alone_times, alerted_ids, debug_scores)
        frame = draw_alert_banner(frame, len(alerted_ids))
        if config.SHOW_FPS_OVERLAY:
            frame = draw_fps(frame, fps_counter.get_fps())

        # --- Отображение (троттлинг — не чаще display_interval) ---
        now = time.time()
        if now - last_display_time >= display_interval:
            last_display_time = now
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[1] > 720:
                new_w = 720
                new_h = int(frame_rgb.shape[0] * 720 / frame_rgb.shape[1])
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Статистика и UI обновляются реже, чем кадры — не узкое место
        if ui_frame_count % ui_update_every == 0:
            current_sec = current_frame / video_fps
            fps_display.metric("FPS", f"{fps_counter.get_fps():.1f}")
            persons_display.metric("Людей в кадре", len(persons))
            children_display.metric("Детей", len(children))
            adults_display.metric("Взрослых", len(adults))
            if total_frames > 0:
                total_sec = total_frames / video_fps
                time_display.metric("Время", f"{current_sec:.1f} / {total_sec:.1f} сек")

            if alert_log:
                with alerts_container:
                    for msg in alert_log[-10:]:
                        st.markdown(msg)

            if total_frames > 0:
                progress_bar.progress(current_frame / total_frames)

    cap.release()
    progress_bar.empty()
    st.success(f"Обработка завершена. Кадров: {frame_count}")

    # Итоговая heatmap
    if heatmap_acc and heatmap_acc.accumulator.max() > 0:
        st.subheader("Итоговая тепловая карта")
        blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        heatmap_img = heatmap_acc.render_overlay(blank)
        st.image(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB), use_container_width=True)

elif start_btn and video_path is None:
    st.warning("Выберите источник видео в боковой панели")
