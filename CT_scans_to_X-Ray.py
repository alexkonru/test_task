"""
Генерация синтетических рентгеновских снимков из датасета VerSe
"""

import os
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import zipfile
from pathlib import Path
import scipy.ndimage as ndimage
import time
from datetime import datetime
import urllib3
import gc
import psutil
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Конфигурация
WORK_DIR = "./verse" 
DATASET_VERSION = "20"
DATA_TYPE = "training"

# Параметры генерации DRR
DRR_CONFIG = {
    "hu_min": -500,
    "hu_max": 1500,
    "attenuation_coeff": 0.03,
}

# Метки по спецификации VerSe
VERTEBRA_LABELS = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
    20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6'
}
VALID_LABELS = set(VERTEBRA_LABELS.keys())

# Создание директорий
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(f"{WORK_DIR}/synthetic_xrays/images", exist_ok=True)
os.makedirs(f"{WORK_DIR}/synthetic_xrays/labels", exist_ok=True)
os.makedirs(f"{WORK_DIR}/verse_data", exist_ok=True)

# ================== ИСПРАВЛЕННЫЕ ФУНКЦИИ ==================

def force_memory_cleanup(): # Очистка памяти
    gc.collect()
    gc.collect()
    time.sleep(1)

def get_orientation(affine): # Определяет анатомическую ориентацию тома
    # Получаем направляющие векторы осей
    x_axis = affine[:3, 0]
    y_axis = affine[:3, 1]
    z_axis = affine[:3, 2]
    
    # Определяем ось позвоночника (Z)
    spine_axis = np.argmax([np.linalg.norm(x_axis), np.linalg.norm(y_axis), np.linalg.norm(z_axis)])
    
    # Определяем ось X и Y
    if spine_axis == 0:
        x_axis_idx = 1
        y_axis_idx = 2
    elif spine_axis == 1:
        x_axis_idx = 0
        y_axis_idx = 2
    else:  
        x_axis_idx = 0
        y_axis_idx = 1
    
    return spine_axis, x_axis_idx, y_axis_idx

def scale_to_physical(drr, seg_2d, voxel_sizes, projection="lateral"): #Масштабирует изображение, voxel_sizes = (dx, dy, dz) — размеры вокселя в мм
    # Определение коэффициента масштабирования
    if projection == "axial":
        scale = voxel_sizes[0] / voxel_sizes[1]  # X / Y
    else:
        # Для латерального и AP
        scale = voxel_sizes[2] / voxel_sizes[0]  # Z / X
    
    # Масштабируем
    h, w = drr.shape
    new_w = int(w * scale)
    drr = cv2.resize(drr, (new_w, h), interpolation=cv2.INTER_LINEAR)
    seg_2d = cv2.resize(seg_2d, (new_w, h), interpolation=cv2.INTER_NEAREST)
    
    return drr, seg_2d

def download_verse_dataset(): # Загрузка датасета VerSe
    print(f"\n{'='*50}")
    print(f"ЗАГРУЗКА VerSe'{DATASET_VERSION} ({DATA_TYPE})")
    print(f"{'='*50}")

    base_url = f"https://s3.bonescreen.de/public/VerSe-complete/dataset-verse{DATASET_VERSION}{DATA_TYPE}.zip"
    archive_path = f"{WORK_DIR}/verse_data/dataset-verse{DATASET_VERSION}{DATA_TYPE}.zip"
    extract_dir = f"{WORK_DIR}/verse_data/verse{DATASET_VERSION}_{DATA_TYPE}/dataset-01training"

    if not os.path.exists(archive_path) and not os.path.exists(extract_dir):
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        try:
            response = requests.get(base_url, stream=True, timeout=120, verify=False)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            with open(archive_path, 'wb') as f, tqdm(
                desc=f"Скачивание {Path(archive_path).name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    pbar.update(size)
        except Exception as e:
            print(f"Ошибка загрузки: {str(e)}")
            return [], None

    if not os.path.exists(extract_dir) and os.path.exists(archive_path):
        print(f"Распаковка: {archive_path}")
        extract_temp_dir = f"{WORK_DIR}/verse_data/verse{DATASET_VERSION}_{DATA_TYPE}"
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp_dir)
    
    data_root = Path(extract_dir)
    print(f"Корень данных: {data_root}")

    rawdata_dir = data_root / "rawdata"
    if not rawdata_dir.exists():
        print(f"Не найдена папка rawdata в: {data_root}")
        return [], None

    patient_dirs = [
        d for d in rawdata_dir.iterdir()
        if d.is_dir() and d.name not in {"__MACOSX"} and not d.name.startswith("._")
    ]

    print(f"Найдено пациентов: {len(patient_dirs)}")
    return patient_dirs, data_root

def get_file_paths(patient_dir, base_dir): # Поиск файлов КТ и сегментации
    patient_name = patient_dir.name

    ct_files = [f for f in patient_dir.glob("*ct.nii.gz") if not f.name.startswith("._")]
    if not ct_files:
        ct_files = [f for f in patient_dir.glob("*.nii.gz") if not f.name.startswith("._")]
    if not ct_files:
        raise FileNotFoundError(f"CT не найден в {patient_dir}")

    ct_path = ct_files[0]

    derivatives_dir = base_dir / "derivatives" / patient_name
    if not derivatives_dir.exists():
        derivatives_dir = base_dir.parent / "derivatives" / patient_name
    if not derivatives_dir.exists():
        raise FileNotFoundError(f"derivatives не найден для {patient_name}")

    seg_files = [f for f in derivatives_dir.glob("*seg*.nii.gz") if not f.name.startswith("._")]
    if not seg_files:
        seg_files = [f for f in derivatives_dir.glob("*.nii.gz") if not f.name.startswith("._")]
    if not seg_files:
        raise FileNotFoundError(f"Сегментация не найдена в {derivatives_dir}")

    for f in seg_files:
        if "seg-vert" in f.name or "seg_vert" in f.name:
            return ct_path, f
    return ct_path, seg_files[0]

def generate_drr(ct_volume, config, projection="lateral", spine_axis=2, x_axis_idx=0, y_axis_idx=1): # DRR-генерация
    vol = ct_volume.astype(np.float32)
    vol = np.clip(vol, config["hu_min"], config["hu_max"])
    vol = (vol - config["hu_min"]) / (config["hu_max"] - config["hu_min"] + 1e-6)

    attenuation = config["attenuation_coeff"]
    transmission = np.exp(-attenuation * vol)
    
    # Определение оси для суммирования
    if projection == "lateral":  # боковая проекция
        sum_axis = x_axis_idx  # суммируем по X
    elif projection == "ap":    # передне-задняя проекция
        sum_axis = y_axis_idx  # суммируем по Y
    elif projection == "axial": # аксиальная проекция
        sum_axis = spine_axis  # суммируем по Z
    else:
        raise ValueError("Неподдерживаемая проекция")
    
    # Суммируем по правильной оси
    integral = np.sum(transmission, axis=sum_axis)
    
    # Формируем 2D-изображение (удаляем ось суммирования)
    drr = -np.log(integral + 1e-8)
    drr = (drr - drr.min()) / (drr.max() - drr.min() + 1e-8) * 255
    
    return drr.astype(np.uint8)

def project_segmentation(seg_volume, projection="lateral", spine_axis=2, x_axis_idx=0, y_axis_idx=1): # Проекция сегментации с учётом ориентации тома
    seg_volume = seg_volume.astype(np.uint16)
    
    # Определение осей
    if projection == "lateral":
        axis1, axis2 = y_axis_idx, spine_axis  # Y, Z
    elif projection == "ap":
        axis1, axis2 = x_axis_idx, spine_axis  # X, Z
    elif projection == "axial":
        axis1, axis2 = x_axis_idx, y_axis_idx  # X, Y
    
    # Создаем 2D массив для проекции
    seg_2d = np.zeros((seg_volume.shape[axis1], seg_volume.shape[axis2]), dtype=np.uint8)
    
    for i in range(seg_volume.shape[axis1]):
        for j in range(seg_volume.shape[axis2]):
            # Определяем индексы для проекции
            indices = [slice(None)] * 3
            indices[axis1] = i
            indices[axis2] = j
            column = seg_volume[tuple(indices)]
            
            non_zero = column[column > 0]
            if len(non_zero) > 0:
                valid_labels = non_zero[np.isin(non_zero, list(VALID_LABELS))]
                if len(valid_labels) > 0:
                    valid_labels_int = valid_labels.astype(np.int64)
                    seg_2d[i, j] = np.bincount(valid_labels_int).argmax()
    return seg_2d

def create_yolo_labels(seg_2d, image_name, output_dir): # Создание YOLO разметки
    label_path = Path(output_dir) / f"{image_name}.txt"
    
    h, w = seg_2d.shape
    
    vertebra_ids = np.unique(seg_2d)
    vertebra_ids = [vid for vid in vertebra_ids if vid in VALID_LABELS]
    count = 0

    with open(label_path, "w") as f:
        for vert_id in vertebra_ids:
            # Создаем маску для текущего позвонка
            mask = (seg_2d == vert_id).astype(np.uint8)
            
            if mask.sum() < 25:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) < 3:
                continue

            # Нормализация координат
            polygon = []
            for point in approx:
                x_norm = point[0][0] / w
                y_norm = point[0][1] / h
                polygon.append(f"{x_norm:.6f} {y_norm:.6f}")

            if len(polygon) >= 3:
                class_id = vert_id - 1
                f.write(f"{class_id} " + " ".join(polygon) + "\n")
                count += 1
    
    return count

def process_single_patient_all_projections(patient_dir, base_dir, idx): # Обработка одного пациента с генерацией всех проекций
    patient_name = patient_dir.name
    
    try:
        force_memory_cleanup()
        
        print(f"[{idx}] Обработка: {patient_name}")
        
        # Получение путей к файлам
        ct_path, seg_path = get_file_paths(patient_dir, base_dir)
        
        # Загрузка данных
        ct_img = nib.load(str(ct_path))
        ct_data = ct_img.get_fdata()
        
        seg_img = nib.load(str(seg_path))
        seg_data = seg_img.get_fdata()
        
        # Определение ориентации тома
        spine_axis, x_axis_idx, y_axis_idx = get_orientation(ct_img.affine)
        voxel_sizes = ct_img.header.get_zooms()
        
        # Приведение к одинаковому размеру
        if ct_data.shape != seg_data.shape:
            print(f"  Ресемплинг сегментации: {seg_data.shape} -> {ct_data.shape}")
            scale_factors = [ct_dim / seg_dim for ct_dim, seg_dim in zip(ct_data.shape, seg_data.shape)]
            seg_data = ndimage.zoom(seg_data, scale_factors, order=0)
        
        total_vert_count = 0
        projections_processed = 0
        
        # Генерация для всех проекций
        projections = [
            ('ap', 'ap'),
            ('lateral', 'lateral'), 
            ('axial', 'axial')
        ]
        
        for projection_name, projection_type in projections:
            try:
                print(f"  Генерация {projection_name} проекции...")
                
                # Генерация DRR
                drr = generate_drr(
                    ct_data, 
                    DRR_CONFIG,
                    projection=projection_type,
                    spine_axis=spine_axis,
                    x_axis_idx=x_axis_idx,
                    y_axis_idx=y_axis_idx
                )
                
                # Проекция сегментации
                seg_2d = project_segmentation(
                    seg_data,
                    projection=projection_type,
                    spine_axis=spine_axis,
                    x_axis_idx=x_axis_idx,
                    y_axis_idx=y_axis_idx
                )
                
                # Масштабирование по физическим размерам
                drr, seg_2d = scale_to_physical(drr, seg_2d, voxel_sizes, projection=projection_type)
                
                # Проверка совпадения размеров
                if drr.shape != seg_2d.shape:
                    print(f"    Размеры не совпадают: DRR {drr.shape} != Seg {seg_2d.shape}")
                    seg_2d = cv2.resize(seg_2d, (drr.shape[1], drr.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Создание YOLO разметки
                vert_count = create_yolo_labels(seg_2d, f"{patient_name}_{projection_name}", 
                                              f"{WORK_DIR}/synthetic_xrays/labels")
                
                if vert_count > 0:
                    # Сохранение изображения
                    cv2.imwrite(f"{WORK_DIR}/synthetic_xrays/images/{patient_name}_{projection_name}.png", drr)
                    total_vert_count += vert_count
                    projections_processed += 1
                    print(f"    {projection_name}: {vert_count} позвонков, размер {drr.shape}")
                else:
                    print(f"    {projection_name}: нет позвонков для разметки")
                
                # Очистка памяти
                del drr, seg_2d
                force_memory_cleanup()
                
            except Exception as e:
                print(f"    Ошибка в {projection_name} проекции: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Очистка основных данных
        del ct_data, seg_data
        force_memory_cleanup()
        
        if projections_processed > 0:
            print(f"  Успешно: {projections_processed}/3 проекций, {total_vert_count} позвонков")
            return True, total_vert_count
        else:
            print(f"  Ни одна проекция не обработана успешно")
            return False, 0
        
    except Exception as e:
        print(f"[{idx}] Ошибка {patient_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        force_memory_cleanup()
        return False, 0

def visualize_annotations(image_path, label_path, output_dir, class_names): # Визуализация разметки на изображениях
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Создание копии для рисования
    vis_image = image.copy()
    
    # Чтение разметки
    label_file = Path(label_path)
    if not label_file.exists():
        print(f"Файл разметки не найден: {label_path}")
        return
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Отрисовка разметки
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        class_id = int(parts[0])
        polygon_points = []
        
        # Преобразование нормализованных координат в пиксельные
        for i in range(1, len(parts), 2):
            x_norm = float(parts[i])
            y_norm = float(parts[i+1])
            x_pixel = int(x_norm * w)
            y_pixel = int(y_norm * h)
            polygon_points.append([x_pixel, y_pixel])
        
        if len(polygon_points) >= 3:
            # Конвертируем в numpy array
            polygon = np.array(polygon_points, np.int32)
            
            # Случайный цвет для каждого класса
            color = np.random.randint(0, 255, 3).tolist()
            
            # Рисуем полигон
            cv2.polylines(vis_image, [polygon], True, color, 2)
            
            # Подписываем класс
            centroid = polygon.mean(axis=0).astype(int)
            class_name = class_names.get(class_id + 1, str(class_id))
            cv2.putText(vis_image, class_name, 
                       (centroid[0], centroid[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Сохранение результата
    output_path = Path(output_dir) / f"vis_{image_path.name}"
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image

# Проверка качества разметки на случайных примерах
def check_annotation_quality(work_dir, num_samples=10):
    images_dir = Path(work_dir) / "synthetic_xrays" / "images"
    labels_dir = Path(work_dir) / "synthetic_xrays" / "labels"
    output_dir = Path(work_dir) / "validation_vis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список всех изображений
    image_files = list(images_dir.glob("*.png"))
    
    if not image_files:
        print("Нет изображений для проверки")
        return
    
    # Выбираем случайные примеры
    sample_files = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    for image_path in sample_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Нет разметки для {image_path.name}")
            continue
            
        # Визуализируем
        visualize_annotations(image_path, label_path, output_dir, VERTEBRA_LABELS)
        print(f"Проверено: {image_path.name}")
    
    print(f"\nРезультаты проверки сохранены в: {output_dir}")

def main():
    try:
        patient_dirs, base_dir = download_verse_dataset()
        
        if not patient_dirs:
            print("Не найдены пациенты для обработки")
            return
            
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return

    # Обработка пациентов
    print(f"\nОбработка {len(patient_dirs)} пациентов...")
    
    processed, skipped, total_verts = 0, 0, 0
    
    for i, patient_dir in enumerate(tqdm(patient_dirs, desc="Обработка пациентов"), 1):
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            time.sleep(5)
            force_memory_cleanup()
        
        ok, vert_count = process_single_patient_all_projections(patient_dir, base_dir, i)
        
        if ok:
            processed += 1
            total_verts += vert_count
        else:
            skipped += 1
        
        if i < len(patient_dirs):
            time.sleep(2)
            force_memory_cleanup()
    
    if processed == 0:
        print("Ни один пациент не обработан успешно")
        return

    # Создание data.yaml
    names_list = [VERTEBRA_LABELS[i] for i in sorted(VERTEBRA_LABELS.keys())]
    data_yaml_content = f"""# YOLOv8 config
path: /content/verse_dataset
train: train/images
val: val/images
nc: 25
names: 
  0: 'C1'
  1: 'C2'
  2: 'C3'
  3: 'C4'
  4: 'C5'
  5: 'C6'
  6: 'C7'
  7: 'T1'
  8: 'T2'
  9: 'T3'
  10: 'T4'
  11: 'T5'
  12: 'T6'
  13: 'T7'
  14: 'T8'
  15: 'T9'
  16: 'T10'
  17: 'T11'
  18: 'T12'
  19: 'L1'
  20: 'L2'
  21: 'L3'
  22: 'L4'
  23: 'L5'
  24: 'L6'
"""

    with open(f"{WORK_DIR}/synthetic_xrays/data.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    # Создание ZIP архива
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"{WORK_DIR}/synthetic_spine_xrays_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        image_files = list(Path(f"{WORK_DIR}/synthetic_xrays/images").glob("*.png"))
        for f in tqdm(image_files, desc="Архивация изображений"):
            zf.write(f, f"images/{f.name}")
        
        label_files = list(Path(f"{WORK_DIR}/synthetic_xrays/labels").glob("*.txt"))
        for f in tqdm(label_files, desc="Архивация разметок"):
            zf.write(f, f"labels/{f.name}")
        
        zf.write(f"{WORK_DIR}/synthetic_xrays/data.yaml", "data.yaml")

    # Визуальная проверка
    check_annotation_quality(WORK_DIR, num_samples=5)

    print(f"Обработано пациентов: {processed}")
    print(f"Пропущено пациентов: {skipped}")
    print(f"Всего размечено позвонков: {total_verts}")
    print(f"Финальный архив: {zip_path} ({os.path.getsize(zip_path)/1024**2:.1f} MB)")

if __name__ == "__main__":
    main()