import os
from ultralytics import YOLO
from pathlib import Path
import zipfile
import shutil
from sklearn.model_selection import train_test_split

def check_existing_dataset():
    target_dir = "/content/verse_dataset"

    if os.path.exists(target_dir):
        images_dir = os.path.join(target_dir, 'images')
        labels_dir = os.path.join(target_dir, 'labels')
        yaml_file = os.path.join(target_dir, 'data.yaml')

        print("Датасет уже распакован и готов к использованию")
        return target_dir

    return None

def setup_dataset(uploaded_zip_path):
    target_dir = "/content/verse_dataset"
    os.makedirs(target_dir, exist_ok=True)

    temp_dir = "/content/temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)

    # Распаковка
    with zipfile.ZipFile(uploaded_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Поиск данных
    for root, dirs, files in os.walk(temp_dir):
        if 'images' in dirs and 'labels' in dirs:
            images_source = os.path.join(root, 'images')
            labels_source = os.path.join(root, 'labels')
            break
    else:
        raise Exception("Не найдены папки images/labels")

    # Копируем в целевую директорию
    shutil.copytree(images_source, os.path.join(target_dir, 'images'), dirs_exist_ok=True)
    shutil.copytree(labels_source, os.path.join(target_dir, 'labels'), dirs_exist_ok=True)

    # Разделяем данные на train/val
    images_list = os.listdir(os.path.join(target_dir, 'images'))
    train_images, val_images = train_test_split(images_list, test_size=0.1, random_state=42)

    # Создаем папки для train и val
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

    # Распределяем train изображения
    for img in train_images:
        shutil.move(
            os.path.join(target_dir, 'images', img),
            os.path.join(train_dir, 'images', img)
        )
        label_file = f"{Path(img).stem}.txt"
        if os.path.exists(os.path.join(target_dir, 'labels', label_file)):
            shutil.move(
                os.path.join(target_dir, 'labels', label_file),
                os.path.join(train_dir, 'labels', label_file)
            )

    # Распределяем val изображения
    for img in val_images:
        shutil.move(
            os.path.join(target_dir, 'images', img),
            os.path.join(val_dir, 'images', img)
        )
        label_file = f"{Path(img).stem}.txt"
        if os.path.exists(os.path.join(target_dir, 'labels', label_file)):
            shutil.move(
                os.path.join(target_dir, 'labels', label_file),
                os.path.join(val_dir, 'labels', label_file)
            )

    # Удаляем пустые исходные папки
    shutil.rmtree(os.path.join(target_dir, 'images'))
    shutil.rmtree(os.path.join(target_dir, 'labels'))

    print(f"Разделение завершено: {len(train_images)} train, {len(val_images)} val")

    return target_dir

def train_model(dataset_path):
    """Обучение модели"""


    data_yaml = f"{dataset_path}/data.yaml"

    model = YOLO('yolov8l-seg.pt')

    train_args = {
        'data': data_yaml,           # Путь к data.yaml
        'epochs': 200,               # Количество эпох
        'imgsz': (512, 1024),        # Размер изображения
        'batch': 4,                  # Размер батча
        'patience': 30,              # Ранняя остановка
        'device': 0,                 # Использовать GPU

        # === ОПТИМИЗАЦИЯ ===
        'lr0': 0.0001,               # Скорость обучения
        'weight_decay': 0.01,        # L2 регуляризация
        'momentum': 0.9,             # Момент
        'optimizer': 'AdamW',        # Оптимизатор
        'cos_lr': True,              # Косинусный планировщик

        # === АУГМЕНТАЦИЯ ===
        'hsv_h': 0.015,              # Изменение оттенка
        'hsv_s': 0.7,                # Изменение насыщенности
        'hsv_v': 0.4,                # Изменение яркости
        'degrees': 10.0,             # Повороты ±10°
        'translate': 0.1,            # Сдвиги
        'scale': 0.3,                # Масштабирование
        'shear': 1.0,                # Наклоны
        'perspective': 0.0003,       # Перспективные искажения
        'fliplr': 0.5,               # Горизонтальное отражение
        'mosaic': 0.8,               # Мозаика
        'mixup': 0.2,                # Смешивание изображений

        # === РЕГУЛЯРИЗАЦИЯ ===
        'dropout': 0.2,              # Dropout
        'label_smoothing': 0.1,      # Сглаживание меток

        'save': True,
        'project': '/content/training_results',
        'name': 'spine_detector',
        'exist_ok': True,
        'plots': True,
    }

    results = model.train(**train_args)
    metrics = model.val()

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return model

def main():
    from google.colab import files

    # Проверяем существующий датасет
    dataset_path = check_existing_dataset()

    if not dataset_path:
        from google.colab import files
        uploaded = files.upload()

        if not uploaded:
            raise Exception("Файл не загружен")

        uploaded_zip_path = list(uploaded.keys())[0]
        dataset_path = setup_dataset(uploaded_zip_path)

    model = train_model(dataset_path)

if __name__ == "__main__":
    main()