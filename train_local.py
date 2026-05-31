# Для локального запуска зависимости уже установлены через pip3
# Если чего-то не хватает, запустите в терминале:
# pip3 install numpy matplotlib pillow torch torchvision scikit-learn albumentations onnx onnxruntime
print('OK')

import os
import glob
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Apple Silicon MPS >> CPU, CUDA для nvidia
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# Fix SSL certificates on macOS
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Kaggle не нужен — данные уже скачаны
print("OK")

AGE_THRESHOLD = 12  # Ключевое изменение: было 16

# === UTKFace ===
DATA_DIR = None
for candidate in ["data/UTKFace", "data/utkface_aligned_cropped/UTKFace", "data/crop_part1"]:
    if os.path.isdir(candidate):
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise FileNotFoundError("UTKFace не найден!")

print(f"UTKFace: {DATA_DIR}")

image_paths = []
labels = []
ages = []

for fpath in glob.glob(os.path.join(DATA_DIR, "*.jpg")):
    fname = os.path.basename(fpath)
    parts = fname.split("_")
    if len(parts) < 3:
        continue
    try:
        age = int(parts[0])
    except ValueError:
        continue
    if age < 0 or age > 120:
        continue
    
    image_paths.append(fpath)
    ages.append(age)
    labels.append(0 if age < AGE_THRESHOLD else 1)

print(f"UTKFace: {len(image_paths)} изображений")
print(f"  Детей (age < {AGE_THRESHOLD}): {labels.count(0)}")
print(f"  Взрослых (age >= {AGE_THRESHOLD}): {labels.count(1)}")

# === FG-NET (если скачался) ===
fgnet_paths = []
fgnet_labels = []
fgnet_ages = []

# FG-NET: имена файлов вида 001A02.JPG (ID + A + возраст)
fgnet_dirs = glob.glob("data/FGNET/**/*.JPG", recursive=True) + \
             glob.glob("data/FGNET/**/*.jpg", recursive=True) + \
             glob.glob("data/FGNET/**/*.png", recursive=True)

for fpath in fgnet_dirs:
    fname = os.path.basename(fpath)
    # Пробуем парсить формат FG-NET: 001A02.JPG
    import re
    match = re.search(r'(\d+)[Aa](\d+)', fname)
    if match:
        age = int(match.group(2))
        fgnet_paths.append(fpath)
        fgnet_ages.append(age)
        fgnet_labels.append(0 if age < AGE_THRESHOLD else 1)

if fgnet_paths:
    print(f"FG-NET: {len(fgnet_paths)} изображений")
    print(f"  Детей: {fgnet_labels.count(0)}, Взрослых: {fgnet_labels.count(1)}")
    image_paths.extend(fgnet_paths)
    labels.extend(fgnet_labels)
    ages.extend(fgnet_ages)
else:
    print("FG-NET: не найден, продолжаем только с UTKFace")

# === Children vs Adults (если скачался) ===
for child_dir in glob.glob("data/children_adults/**/children", recursive=True) + \
                 glob.glob("data/children_adults/**/child", recursive=True) + \
                 glob.glob("data/children_adults/**/kids", recursive=True):
    child_imgs = glob.glob(os.path.join(child_dir, "*.jpg")) + \
                 glob.glob(os.path.join(child_dir, "*.png")) + \
                 glob.glob(os.path.join(child_dir, "*.jpeg"))
    for fpath in child_imgs:
        image_paths.append(fpath)
        labels.append(0)  # child
        ages.append(5)  # приблизительно
    if child_imgs:
        print(f"Children vs Adults (children): +{len(child_imgs)} изображений")

for adult_dir in glob.glob("data/children_adults/**/adults", recursive=True) + \
                 glob.glob("data/children_adults/**/adult", recursive=True):
    adult_imgs = glob.glob(os.path.join(adult_dir, "*.jpg")) + \
                 glob.glob(os.path.join(adult_dir, "*.png")) + \
                 glob.glob(os.path.join(adult_dir, "*.jpeg"))
    for fpath in adult_imgs:
        image_paths.append(fpath)
        labels.append(1)  # adult
        ages.append(30)  # приблизительно
    if adult_imgs:
        print(f"Children vs Adults (adults): +{len(adult_imgs)} изображений")

print(f"\nИТОГО: {len(image_paths)} изображений")
print(f"  Детей: {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
print(f"  Взрослых: {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")

# Распределение возрастов
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(ages, bins=range(0, 100), edgecolor='black', alpha=0.7)
plt.axvline(x=AGE_THRESHOLD, color='red', linestyle='--', linewidth=2,
            label=f'Порог = {AGE_THRESHOLD} лет')
plt.xlabel('Возраст')
plt.ylabel('Количество')
plt.title('Распределение возрастов')
plt.legend()

plt.subplot(1, 2, 2)
counts = Counter(labels)
plt.bar(['Child (<12)', 'Adult (>=12)'], [counts[0], counts[1]],
        color=['#4CAF50', '#2196F3'])
plt.ylabel('Количество')
plt.title('Баланс классов')
for i, v in enumerate([counts[0], counts[1]]):
    plt.text(i, v + 100, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('dataset_distribution_v2.png', dpi=100)
plt.close()
print("Сохранено: dataset_distribution_v2.png")

# Аугментации через albumentations (мощнее чем torchvision)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_augment = A.Compose([
    A.Resize(224, 224),
    
    # --- Геометрические ---
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
    
    # --- Имитация видеокамеры ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),      # Расфокус
        A.MotionBlur(blur_limit=7, p=1.0),              # Смаз от движения
    ], p=0.4),
    
    A.ImageCompression(quality_range=(30, 80), p=0.4),  # JPEG-артефакты
    A.GaussNoise(std_range=(0.02, 0.1), p=0.3),          # Шум камеры

    A.OneOf([
        A.Downscale(scale_range=(0.3, 0.7), p=1.0),  # Имитация далёкой камеры
        A.Sharpen(alpha=(0.2, 0.5), p=1.0),
    ], p=0.3),
    
    # --- Освещение ---
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
    
    # --- Нормализация ---
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

val_augment = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])

print("Аугментации настроены (с имитацией видеокамеры)")

# Визуализация аугментаций
sample_idx = random.choice([i for i, l in enumerate(labels) if l == 0])  # ребёнок
sample_img = np.array(Image.open(image_paths[sample_idx]).convert('RGB'))

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes[0][0].imshow(sample_img)
axes[0][0].set_title('Оригинал')
axes[0][0].axis('off')

for i in range(1, 10):
    row, col = divmod(i, 5)
    aug = train_augment(image=sample_img)
    # Денормализуем для отображения
    img_show = aug['image'].numpy().transpose(1, 2, 0)
    img_show = img_show * np.array(STD) + np.array(MEAN)
    img_show = np.clip(img_show, 0, 1)
    axes[row][col].imshow(img_show)
    axes[row][col].set_title(f'Аугментация {i}')
    axes[row][col].axis('off')

plt.suptitle('Примеры аугментаций (имитация видеокамеры)', fontsize=14)
plt.tight_layout()
plt.savefig('augmentations_v2.png', dpi=100)
plt.close()

class AgeDataset(Dataset):
    """Dataset с albumentations аугментациями."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        except Exception:
            # Битый файл — возвращаем чёрное изображение
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, label

# Разделение на train / val (80% / 20%)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

# Оверсэмплинг детей в train (у нас их меньше)
child_indices = [i for i, l in enumerate(train_labels) if l == 0]
adult_indices = [i for i, l in enumerate(train_labels) if l == 1]

# Дублируем детей чтобы классы были ~равны
oversample_factor = max(1, len(adult_indices) // len(child_indices))
extra_child_paths = [train_paths[i] for i in child_indices] * (oversample_factor - 1)
extra_child_labels = [0] * len(extra_child_paths)

train_paths_balanced = train_paths + extra_child_paths
train_labels_balanced = train_labels + extra_child_labels

print(f"Train (до балансировки): {len(train_paths)}")
print(f"  child: {train_labels.count(0)}, adult: {train_labels.count(1)}")
print(f"Train (после оверсэмплинга): {len(train_paths_balanced)}")
print(f"  child: {train_labels_balanced.count(0)}, adult: {train_labels_balanced.count(1)}")
print(f"Val: {len(val_paths)}")
print(f"  child: {val_labels.count(0)}, adult: {val_labels.count(1)}")

BATCH_SIZE = 32

train_dataset = AgeDataset(train_paths_balanced, train_labels_balanced, transform=train_augment)
val_dataset = AgeDataset(val_paths, val_labels, transform=val_augment)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Батчей в train: {len(train_loader)}")
print(f"Батчей в val: {len(val_loader)}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Замораживаем всё
for param in model.parameters():
    param.requires_grad = False

# Размораживаем последние 5 блоков (больше чем раньше)
for param in model.features[-5:].parameters():
    param.requires_grad = True

# Улучшенный классификатор с BatchNorm
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)  # 1 выход: sigmoid -> вероятность adult
)

model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Всего параметров: {total_params:,}")
print(f"Обучаемых: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss с label smoothing.
    
    Вместо 0/1 используем smoothing/1-smoothing.
    Это уменьшает overconfidence модели.
    """
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        targets_smooth = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_smooth, pos_weight=self.pos_weight
        )
        return loss


NUM_EPOCHS = 10  # 10 эпох достаточно с MPS
LEARNING_RATE = 5e-4  # Чуть ниже для стабильности

# Веса классов: ошибка на ребёнке стоит дороже
n_child = train_labels_balanced.count(0)
n_adult = train_labels_balanced.count(1)
# Даже после оверсэмплинга даём больший вес child-ошибкам
pos_weight = torch.tensor([n_child / n_adult * 0.7]).to(device)  # adult весит меньше
print(f"pos_weight (adult): {pos_weight.item():.3f}")

criterion = LabelSmoothingBCELoss(smoothing=0.1, pos_weight=pos_weight)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4  # L2 регуляризация
)

# Cosine annealing — плавное снижение LR
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

print(f"Эпох: {NUM_EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
print(f"Label smoothing: 0.1, Weight decay: 1e-4")

import time

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    n_batches = len(loader)

    for batch_idx, (images, labels_batch) in enumerate(loader):
        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx}/{n_batches}", end="\r")
        images = images.to(device)
        labels_batch = labels_batch.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.int().cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Для анализа калибровки
    
    with torch.no_grad():
        for images, labels_batch in loader:
            images = images.to(device)
            labels_batch = labels_batch.float().to(device)
            
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels_batch)
            
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.int().cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels, all_probs

# Основной цикл обучения
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}
best_val_f1_child = 0  # Оптимизируем F1 для класса child!
best_epoch = 0

print(f"{'Epoch':>5} | {'T.Loss':>7} | {'T.Acc':>6} | {'V.Loss':>7} | {'V.Acc':>6} | {'Child F1':>8} | {'LR':>8}")
print("-" * 65)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_preds, val_true, val_probs = validate(model, val_loader, criterion, device)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # F1 для класса child (pos_label=0)
    child_f1 = f1_score(val_true, val_preds, pos_label=0)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Сохраняем лучшую модель по F1 РЕБЁНКА (не accuracy!)
    if child_f1 > best_val_f1_child:
        best_val_f1_child = child_f1
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model_v2.pth')
    
    elapsed = time.time() - t0
    print(f"{epoch:>5} | {train_loss:>7.4f} | {train_acc:>5.3f} | {val_loss:>7.4f} | {val_acc:>5.3f} | {child_f1:>7.4f} | {current_lr:.6f} | {elapsed:.0f}s")

print("-" * 65)
print(f"Лучшая модель: эпоха {best_epoch}, child F1 = {best_val_f1_child:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, NUM_EPOCHS + 1)

ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss', markersize=4)
ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss', markersize=4)
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Loss')
ax1.set_title('Функция потерь')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy', markersize=4)
ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Val Accuracy', markersize=4)
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Accuracy')
ax2.set_title('Точность классификации')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_v2.png', dpi=150)
plt.close()

# Загружаем лучшую модель
model.load_state_dict(torch.load('best_model_v2.pth', map_location=device))
val_loss, val_acc, val_preds, val_true, val_probs = validate(model, val_loader, criterion, device)

print("=" * 50)
print("МЕТРИКИ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ")
print("=" * 50)
print(f"\nAccuracy:  {accuracy_score(val_true, val_preds):.4f}")
print(f"\nКласс CHILD (главный для нас):")
print(f"  Precision: {precision_score(val_true, val_preds, pos_label=0):.4f}")
print(f"  Recall:    {recall_score(val_true, val_preds, pos_label=0):.4f}")
print(f"  F1-score:  {f1_score(val_true, val_preds, pos_label=0):.4f}")
print(f"\nПолный отчёт:")
print(classification_report(val_true, val_preds, target_names=['Child', 'Adult']))

# Confusion Matrix
cm = confusion_matrix(val_true, val_preds)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')

classes = ['Child (<12)', 'Adult (>=12)']
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)
ax.set_xlabel('Предсказание', fontsize=13)
ax.set_ylabel('Истина', fontsize=13)
ax.set_title(f'Confusion Matrix (порог {AGE_THRESHOLD} лет)', fontsize=14)

for i in range(2):
    for j in range(2):
        color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, f'{cm[i, j]}\n({cm[i, j]/cm.sum()*100:.1f}%)',
                ha='center', va='center', fontsize=14, color=color)

plt.colorbar(im)
plt.tight_layout()
plt.savefig('confusion_matrix_v2.png', dpi=150)
plt.close()

# Анализ калибровки: распределение вероятностей
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

child_probs = [p for p, t in zip(val_probs, val_true) if t == 0]
adult_probs = [p for p, t in zip(val_probs, val_true) if t == 1]

ax1.hist(child_probs, bins=50, alpha=0.7, color='green', label='Реальные дети')
ax1.axvline(x=0.5, color='red', linestyle='--', label='Порог 0.5')
ax1.set_xlabel('P(adult)')
ax1.set_ylabel('Количество')
ax1.set_title('Распределение P(adult) для ДЕТЕЙ\n(хорошо = скопилось слева)')
ax1.legend()

ax2.hist(adult_probs, bins=50, alpha=0.7, color='blue', label='Реальные взрослые')
ax2.axvline(x=0.5, color='red', linestyle='--', label='Порог 0.5')
ax2.set_xlabel('P(adult)')
ax2.set_ylabel('Количество')
ax2.set_title('Распределение P(adult) для ВЗРОСЛЫХ\n(хорошо = скопилось справа)')
ax2.legend()

plt.tight_layout()
plt.savefig('calibration_v2.png', dpi=150)
plt.close()

# Ошибки на детях — самое важное
child_errors = sum(1 for p, t in zip(val_probs, val_true) if t == 0 and p > 0.5)
print(f"\nДетей неправильно классифицированных как adult: {child_errors}/{len(child_probs)} ({child_errors/len(child_probs)*100:.1f}%)")
print(f"Средняя P(adult) для детей: {np.mean(child_probs):.3f} (идеал: <0.2)")
print(f"Средняя P(adult) для взрослых: {np.mean(adult_probs):.3f} (идеал: >0.8)")

import onnx

model.eval()
model_cpu = model.to('cpu')

dummy_input = torch.randn(1, 3, 224, 224)

ONNX_PATH = 'age_classifier_v2.onnx'

torch.onnx.export(
    model_cpu,
    dummy_input,
    ONNX_PATH,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=11,
    dynamo=False
)

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

file_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"Модель: {ONNX_PATH}")
print(f"Размер: {file_size:.1f} MB")

if file_size < 1.0:
    print("\u26a0\ufe0f  ВНИМАНИЕ: файл слишком маленький!")
else:
    print("\u2713 Размер в норме")

# Проверяем ONNX inference
import onnxruntime as ort

session = ort.InferenceSession(ONNX_PATH)

print("Проверка ONNX inference:")
print(f"{'Файл':<40} {'Возраст':>7} {'Истина':>8} {'ONNX':>8} {'P(adult)':>8}")
print("-" * 75)

# Специально берём детей 5-12 лет (наша целевая группа)
target_children = [i for i, (l, a) in enumerate(zip(val_labels, 
    [ages[image_paths.index(p)] if p in image_paths else 0 for p in val_paths]))
    if l == 0]
test_indices = random.sample(target_children, min(5, len(target_children)))
test_indices += random.sample([i for i, l in enumerate(val_labels) if l == 1], 5)

for i in test_indices:
    img = np.array(Image.open(val_paths[i]).convert('RGB'))
    aug = val_augment(image=img)
    img_t = aug['image'].unsqueeze(0).numpy()
    
    output = session.run(None, {'input': img_t})[0]
    prob = 1 / (1 + np.exp(-output[0, 0]))
    pred = 'adult' if prob > 0.5 else 'child'
    true_label = 'adult' if val_labels[i] == 1 else 'child'
    
    fname = os.path.basename(val_paths[i])
    age_str = fname.split('_')[0] if '_' in fname else '?'
    match = '\u2713' if pred == true_label else '\u2717'
    
    print(f"{fname:<40} {age_str:>7} {true_label:>8} {pred:>8} {prob:>7.3f} {match}")

