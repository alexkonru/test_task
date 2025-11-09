from ultralytics import YOLO

# Загрузка модели
model = YOLO('best.pt')

results = model.predict(
    'test_images/',  # папка с тестовыми снимками
    conf=0.5,
    imgsz=1024,
    save=True
)

print(f"Обработано {len(results)} изображений")