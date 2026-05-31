"""
Обучение YOLO26 для прямой детекции adult/child.

Датасет: Roboflow (adult-child, version 3) в формате YOLO.
Скачать: https://app.roboflow.com/projects-20qd2/adult-child/3 → YOLOv8 format

Структура после скачивания:
  data/adult_child_dataset/
    train/images/
    train/labels/
    valid/images/
    valid/labels/
    test/images/
    test/labels/
    data.yaml

Запуск:
  python train_yolo_child.py                          # YOLO26s (рекомендуется)
  python train_yolo_child.py --model yolo26n.pt       # YOLO26n (быстрее, менее точная)
  python train_yolo_child.py --model yolo26m.pt       # YOLO26m (точнее, медленнее)
  python train_yolo_child.py --resume                 # Продолжить обучение
"""

import argparse
import sys
from pathlib import Path

# --- Fix SSL для macOS ---
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 adult/child detector")
    parser.add_argument("--model", default="yolo26s.pt",
                        help="Base model: yolo26n.pt, yolo26s.pt, yolo26m.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--dataset", default="data/adult_child_dataset/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--device", default=None, help="Device: cpu, mps, 0 (CUDA)")
    args = parser.parse_args()

    # Проверяем наличие датасета
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Датасет не найден: {dataset_path}")
        print()
        print("Скачай датасет с Roboflow:")
        print("  1. https://app.roboflow.com/projects-20qd2/adult-child/3")
        print("  2. Download Dataset → YOLOv8 format (совместим с YOLO26)")
        print(f"  3. Распакуй в {dataset_path.parent}/")
        print()
        print("Или через API:")
        print('  from roboflow import Roboflow')
        print('  rf = Roboflow(api_key="YOUR_KEY")')
        print('  project = rf.workspace("projects-20qd2").project("adult-child")')
        print(f'  project.version(3).download("yolov8", location="{dataset_path.parent}")')
        sys.exit(1)

    # Определяем устройство
    if args.device is None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[INIT] Модель: {args.model}")
    print(f"[INIT] Датасет: {args.dataset}")
    print(f"[INIT] Device: {device}")
    print(f"[INIT] Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print("-" * 60)

    from ultralytics import YOLO

    if args.resume:
        model = YOLO("runs/detect/adult_child/weights/last.pt")
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=args.dataset,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            # Аугментации для surveillance-видео
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,          # Масштаб (дети = маленькие объекты)
            shear=2.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            # Оптимизация
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            # Сохранение
            project="runs/detect",
            name="adult_child",
            exist_ok=True,
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
        )

    print("-" * 60)
    print("[DONE] Обучение завершено!")
    print(f"[DONE] Лучшая модель: runs/detect/adult_child/weights/best.pt")

    # Валидация
    print("\n[EVAL] Валидация на тестовом наборе...")
    best_model = YOLO("runs/detect/adult_child/weights/best.pt")
    metrics = best_model.val(data=args.dataset, split="test")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    # Копируем лучшую модель в models/
    import shutil
    dest = Path("models/yolo_child_detector.pt")
    dest.parent.mkdir(exist_ok=True)
    shutil.copy("runs/detect/adult_child/weights/best.pt", dest)
    print(f"[DONE] Скопировано в: {dest}")


if __name__ == "__main__":
    main()
