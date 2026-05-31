# Система детекции детей без сопровождения взрослых

Прототип системы компьютерного зрения для автоматизированного выявления детей, находящихся без сопровождения взрослых, на видео с камер наблюдения.

## Быстрый старт (без Docker)

### 1. Создать виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить веб-интерфейс

```bash
streamlit run web_app.py
```

Откроется браузер по адресу **http://localhost:8501**.  
В интерфейсе выберите источник видео в боковой панели и нажмите **▶ Запустить**.

---

## Запуск через консоль (без браузера)

```bash
# Видео из config.py
python main.py

# Указать файл
python main.py --source data/1.mp4

# Веб-камера
python main.py --source 0

# Настройка параметров
python main.py --source data/1.mp4 --threshold 3.0 --radius 150

# Без окна (только запись в файл)
python main.py --source data/1.mp4 --no-display --output output/result.mp4
```

Для выхода — нажмите `q` в окне видео или `Ctrl+C` в терминале.

---

## Запуск через Docker

```bash
docker compose up --build
```

Приложение будет доступно на **http://localhost:8501**.  
Видеофайлы кладите в папку `data/` — она монтируется в контейнер автоматически.

---

## Структура проекта

```
├── web_app.py           # Streamlit веб-интерфейс
├── main.py              # Консольный запуск
├── config.py            # Все параметры системы
├── detection.py         # Детекция + трекинг (YOLO + BoT-SORT)
├── age_classifier.py    # Классификация возраста (ансамбль методов)
├── alert_logic.py       # Логика сопровождения и алертов
├── heatmap.py           # Тепловая карта зон риска
├── visualization.py     # Отрисовка bbox, меток, баннеров
├── utils.py             # FPS-счётчик, webhook
├── models/              # Веса моделей (.pt, .onnx)
├── data/                # Видеофайлы для тестирования
├── requirements.txt     # Python-зависимости
└── tests/               # Юнит-тесты
```

## Тесты

```bash
pytest tests/ -v
```

## Основные параметры (config.py)

| Параметр | Описание | По умолчанию |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | Порог уверенности детекции | `0.25` |
| `PROXIMITY_RADIUS_PX` | Радиус сопровождения (px) | `200` |
| `ALERT_THRESHOLD_SEC` | Время до алерта (сек) | `5.0` |
| `AGE_CLASSIFIER` | Метод классификации: `ensemble` / `pose` / `heuristic` | `ensemble` |
| `YOLO_IMGSZ` | Разрешение inference (меньше = быстрее) | `640` |
| `POSE_SKIP_FRAMES` | Запускать pose-модель раз в N кадров | `3` |
