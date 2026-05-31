"""
FastAPI-сервер с MJPEG-стримом для системы детекции детей без сопровождения.
Замена Streamlit-интерфейса: больше нет WebSocket-проблем, кадры идут как
multipart/x-mixed-replace JPEG-поток (нативно поддерживается всеми браузерами).

Запуск:
    uvicorn server:app --host 0.0.0.0 --port 8501
"""
import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

import config
from detection import PersonDetector
from age_classifier import create_age_classifier
from alert_logic import AlertManager
from heatmap import HeatmapAccumulator
from visualization import draw_persons, draw_alert_banner, draw_fps
from utils import FPSCounter


app = FastAPI(title="Система детекции детей без сопровождения")

# ====== Глобальное состояние ======
class State:
    def __init__(self):
        self.video_path: str | None = None
        self.lock = threading.Lock()
        self.last_frame_jpeg: bytes | None = None
        self.stats = {
            "fps": 0.0,
            "persons": 0,
            "children": 0,
            "adults": 0,
            "alerted": 0,
            "current_sec": 0.0,
            "total_sec": 0.0,
            "status": "idle",   # idle | running | done | error
        }
        self.alerts: list[dict] = []
        self.proximity_radius = config.PROXIMITY_RADIUS_PX
        self.alert_threshold = config.ALERT_THRESHOLD_SEC
        self.worker_thread: threading.Thread | None = None
        self.stop_flag = False
        self.paused = False
        self.seek_to_sec: float | None = None


state = State()


def _processing_loop():
    """Фоновая обработка видео — пишет JPEG-кадры в state.last_frame_jpeg."""
    cap = cv2.VideoCapture(state.video_path)
    if not cap.isOpened():
        state.stats["status"] = "error"
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    state.stats["total_sec"] = total_frames / video_fps if total_frames > 0 else 0
    state.stats["status"] = "running"

    # Ресайз обработки до 640 px по ширине (как в web_app.py)
    process_width = min(frame_width, 640)
    scale = process_width / frame_width
    process_height = int(frame_height * scale)

    # Применяем настройки на уровне config (так делает и web_app.py)
    config.PROXIMITY_RADIUS_PX = state.proximity_radius
    config.ALERT_THRESHOLD_SEC = state.alert_threshold
    detector = PersonDetector()
    classifier = create_age_classifier()
    alert_manager = AlertManager()
    heatmap_acc = None
    if config.HEATMAP_ENABLED:
        heatmap_acc = HeatmapAccumulator(process_width, process_height)

    fps_counter = FPSCounter()
    state.alerts.clear()
    frame_count = 0

    while cap.isOpened():
        if state.stop_flag:
            break
        # Пауза — крутимся вхолостую
        if state.paused:
            time.sleep(0.1)
            continue
        # Перемотка
        if state.seek_to_sec is not None:
            target_frame = int(state.seek_to_sec * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            state.seek_to_sec = None
            # Сбрасываем менеджер тревог, чтобы не было ложных алертов от рассинхрона
            alert_manager.states.clear()
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = time.time()
        fps_counter.tick()

        # Ресайз кадра до рабочего разрешения (640px) — ускоряет в 3-5 раз
        if scale < 1.0:
            frame = cv2.resize(frame, (process_width, process_height),
                               interpolation=cv2.INTER_LINEAR)

        # --- Пайплайн ---
        persons = detector.detect_and_track(frame)
        age_labels = classifier.classify(persons, frame)

        children, adults = [], []
        for p, lbl in zip(persons, age_labels):
            (children if lbl == "child" else adults).append(p)

        alerts = alert_manager.update(children, adults, current_time)
        alone_times = {}
        alerted_ids = set()
        unaccompanied_children = []
        for child in children:
            alone = alert_manager.get_child_alone_time(child.track_id, current_time)
            if alone > 0:
                alone_times[child.track_id] = alone
                unaccompanied_children.append(child)
            st = alert_manager.states.get(child.track_id)
            if st and st.is_alerted:
                alerted_ids.add(child.track_id)

        for alert in alerts:
            if alert.status == "NEW":
                state.alerts.insert(0, {
                    "track_id": alert.track_id,
                    "elapsed_sec": round(alert.elapsed_sec, 1),
                    "ts": time.strftime("%H:%M:%S"),
                })
                state.alerts = state.alerts[:30]

        if heatmap_acc and unaccompanied_children:
            heatmap_acc.update(unaccompanied_children)
        elif heatmap_acc:
            heatmap_acc.accumulator *= config.HEATMAP_DECAY

        # --- Визуализация ---
        if heatmap_acc:
            frame = heatmap_acc.render_overlay(frame)
        frame = draw_persons(frame, persons, age_labels, alone_times, alerted_ids, None)
        frame = draw_alert_banner(frame, len(alerted_ids))
        frame = draw_fps(frame, fps_counter.get_fps())

        # --- JPEG (повышенное качество отображения) ---
        if frame.shape[1] < 1280:
            display_scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame,
                               (1280, int(frame.shape[0] * display_scale)),
                               interpolation=cv2.INTER_CUBIC)
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            with state.lock:
                state.last_frame_jpeg = jpg.tobytes()

        # --- Stats ---
        state.stats.update({
            "fps": round(fps_counter.get_fps(), 1),
            "persons": len(persons),
            "children": len(children),
            "adults": len(adults),
            "alerted": len(alerted_ids),
            "current_sec": round(current_frame / video_fps, 1),
        })

    cap.release()
    final_status = "stopped" if state.stop_flag else "done"
    state.stats["status"] = final_status
    state.stats["fps"] = 0.0
    # Очищаем последний кадр, чтобы стрим показывал плейсхолдер
    with state.lock:
        state.last_frame_jpeg = None


# ====== Маршруты ======

INDEX_HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Детекция детей без сопровождения</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background:#f4f4f6; }
  header { background:#1f2430; color:#fff; padding:14px 20px; }
  header h1 { margin:0; font-size:18px; }
  .wrap { display:grid; grid-template-columns: 1fr 320px; gap:16px; padding:16px; }
  .video-card { background:#000; border-radius:12px; overflow:hidden; aspect-ratio: 16/9; display:flex; align-items:center; justify-content:center; }
  .video-card img { width:100%; height:100%; object-fit:contain; display:block; }
  .panel { background:#fff; border-radius:12px; padding:14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  .panel h2 { margin:0 0 10px; font-size:14px; color:#666; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  .stat { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #f0f0f3; font-size:14px; }
  .stat:last-child { border:0; }
  .stat strong { font-size:18px; font-weight:600; }
  button { background:#3563ff; color:#fff; border:0; padding:10px 16px; border-radius:8px; font-size:14px; cursor:pointer; }
  button:hover { background:#2050e6; }
  button.stop { background:#dc3545; }
  input[type=file], input[type=number] { width:100%; padding:8px; border:1px solid #ddd; border-radius:6px; font-size:14px; box-sizing:border-box; }
  label { display:block; margin-top:10px; font-size:13px; color:#555; }
  .alerts { max-height:300px; overflow-y:auto; }
  .alert-item { padding:6px 8px; border-left:3px solid #dc3545; background:#fff3f3; margin-bottom:6px; border-radius:4px; font-size:13px; }
  .alert-item .ts { color:#999; font-size:11px; }
  .row { display:flex; gap:8px; }
  .row > * { flex:1; }
  .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
  .status-running { background:#22c55e; animation:pulse 1.5s infinite; }
  .status-idle { background:#9ca3af; }
  .status-done { background:#3b82f6; }
  .status-error { background:#ef4444; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
</style>
</head>
<body>
<header>
  <h1>Система детекции детей без сопровождения</h1>
</header>
<div class="wrap">
  <div>
    <div class="video-card">
      <img id="video-img" src="/stream" alt="видео не запущено">
    </div>
    <div class="timeline" style="margin-top:10px; background:#fff; border-radius:10px; padding:10px 14px; box-shadow:0 1px 3px rgba(0,0,0,.06);">
      <div style="display:flex; align-items:center; gap:10px;">
        <button id="pause-btn" type="button" onclick="togglePause()" style="background:#6b7280; padding:6px 12px;">⏸ Пауза</button>
        <span id="time-label" style="font-size:13px; color:#555; min-width:110px;">0 / 0 сек</span>
        <input id="seek-bar" type="range" min="0" max="0" value="0" step="0.1"
               style="flex:1; cursor:pointer;" oninput="onSeekInput(this)" onchange="onSeekChange(this)">
      </div>
    </div>
  </div>
  <div>
    <div class="panel">
      <h2>Статус</h2>
      <div class="stat"><span><span class="status-dot status-idle" id="status-dot"></span><span id="status">idle</span></span></div>
      <div class="stat"><span>FPS</span><strong id="fps">0</strong></div>
      <div class="stat"><span>Людей</span><strong id="persons">0</strong></div>
      <div class="stat"><span>Детей</span><strong id="children">0</strong></div>
      <div class="stat"><span>Взрослых</span><strong id="adults">0</strong></div>
      <div class="stat"><span>Тревог активно</span><strong id="alerted">0</strong></div>
    </div>

    <div class="panel" style="margin-top:14px">
      <h2>Источник</h2>
      <form id="form-upload" enctype="multipart/form-data" method="POST">
        <label>Загрузить файл</label>
        <input type="file" name="file" accept="video/*">
        <label>или путь к файлу на сервере</label>
        <input type="text" name="path" placeholder="data/1.mp4" value="data/1.mp4">
        <label>Радиус сопровождения (px)</label>
        <input type="number" name="radius" value="200" step="10" min="50" max="500">
        <label>Порог тревоги (сек)</label>
        <input type="number" name="threshold" value="5" step="0.5" min="1" max="30">
        <div class="row" style="margin-top:14px">
          <button type="submit">▶ Старт</button>
          <button type="button" class="stop" onclick="stop()">⏹ Стоп</button>
        </div>
      </form>
    </div>

    <div class="panel" style="margin-top:14px">
      <h2>Тревоги</h2>
      <div class="alerts" id="alerts"><em style="color:#999;font-size:13px">Пока тревог нет</em></div>
    </div>
  </div>
</div>

<script>
let seekDragging = false;
let lastTotal = 0;

async function refresh() {
  try {
    const r = await fetch('/stats');
    const s = await r.json();
    document.getElementById('fps').textContent = s.fps;
    document.getElementById('persons').textContent = s.persons;
    document.getElementById('children').textContent = s.children;
    document.getElementById('adults').textContent = s.adults;
    document.getElementById('alerted').textContent = s.alerted;
    document.getElementById('status').textContent = s.status;
    document.getElementById('status-dot').className = 'status-dot status-' + s.status;

    // timeline
    const seek = document.getElementById('seek-bar');
    if (s.total_sec && s.total_sec !== lastTotal) {
      seek.max = s.total_sec;
      lastTotal = s.total_sec;
    }
    if (!seekDragging) {
      seek.value = s.current_sec;
    }
    document.getElementById('time-label').textContent =
      `${(+s.current_sec).toFixed(1)} / ${(+s.total_sec).toFixed(1)} сек`;

    const alertsBox = document.getElementById('alerts');
    if (s.alerts && s.alerts.length) {
      alertsBox.innerHTML = s.alerts.map(a =>
        `<div class="alert-item"><strong>Ребёнок #${a.track_id}</strong> без сопровождения ${a.elapsed_sec} сек<br><span class="ts">${a.ts}</span></div>`
      ).join('');
    } else {
      alertsBox.innerHTML = '<em style="color:#999;font-size:13px">Пока тревог нет</em>';
    }
  } catch(e) {}
}
setInterval(refresh, 700);
refresh();

document.getElementById('form-upload').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  await fetch('/start', { method: 'POST', body: f });
  document.getElementById('video-img').src = '/stream?ts=' + Date.now();
  document.getElementById('pause-btn').textContent = '⏸ Пауза';
});

async function stop() {
  await fetch('/stop', { method: 'POST' });
  document.getElementById('video-img').src = '/stream?ts=' + Date.now();
}

async function togglePause() {
  const r = await fetch('/pause', { method: 'POST' });
  const j = await r.json();
  document.getElementById('pause-btn').textContent = j.paused ? '▶ Продолжить' : '⏸ Пауза';
}

function onSeekInput(el) {
  seekDragging = true;
  document.getElementById('time-label').textContent =
    `${(+el.value).toFixed(1)} / ${(+el.max).toFixed(1)} сек`;
}

async function onSeekChange(el) {
  const fd = new FormData();
  fd.append('position', el.value);
  await fetch('/seek', { method: 'POST', body: fd });
  setTimeout(() => { seekDragging = false; }, 800);
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/stats")
async def stats():
    return JSONResponse({**state.stats, "alerts": state.alerts})


@app.post("/start")
async def start(
    file: UploadFile | None = File(None),
    path: str | None = Form(None),
    radius: int = Form(200),
    threshold: float = Form(5.0),
):
    # остановить текущую обработку
    state.stop_flag = True
    if state.worker_thread and state.worker_thread.is_alive():
        state.worker_thread.join(timeout=2)
    state.stop_flag = False

    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        tmp_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        state.video_path = tmp_path
    elif path:
        candidate = path
        if not os.path.isabs(candidate):
            candidate = os.path.join(os.path.dirname(__file__), candidate)
        if not os.path.exists(candidate):
            return JSONResponse({"error": f"file not found: {path}"}, status_code=400)
        state.video_path = candidate
    else:
        return JSONResponse({"error": "no file/path"}, status_code=400)

    state.proximity_radius = radius
    state.alert_threshold = threshold
    state.last_frame_jpeg = None
    state.worker_thread = threading.Thread(target=_processing_loop, daemon=True)
    state.worker_thread.start()
    return {"status": "started", "path": state.video_path}


@app.post("/stop")
async def stop_endpoint():
    state.stop_flag = True
    state.paused = False
    if state.worker_thread and state.worker_thread.is_alive():
        state.worker_thread.join(timeout=2)
    state.stats["status"] = "idle"
    state.stats["fps"] = 0.0
    with state.lock:
        state.last_frame_jpeg = None
    return {"status": "stopped"}


@app.post("/pause")
async def pause_endpoint():
    state.paused = not state.paused
    return {"paused": state.paused}


@app.post("/seek")
async def seek_endpoint(position: float = Form(...)):
    state.seek_to_sec = max(0.0, position)
    return {"seek": state.seek_to_sec}


def _mjpeg_frames():
    """Генератор MJPEG-кадров для multipart/x-mixed-replace."""
    placeholder = None
    while True:
        with state.lock:
            jpg = state.last_frame_jpeg
        if jpg is None:
            # отдадим серый кадр-плейсхолдер
            if placeholder is None:
                blank = np.full((360, 640, 3), 24, dtype=np.uint8)
                cv2.putText(blank, "Waiting for video...", (140, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                ok, p = cv2.imencode('.jpg', blank)
                placeholder = p.tobytes() if ok else b''
            jpg = placeholder
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.07)  # ~14 fps дисплея — комфортно для глаза


@app.get("/stream")
async def stream():
    return StreamingResponse(_mjpeg_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
