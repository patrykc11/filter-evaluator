# https://github.com/ultralytics/ultralytics?tab=readme-ov-file
# wytrenowany model yolo v8 na danych z coco

from ultralytics import YOLO
import os
import shutil
from PIL import Image

model = YOLO("yolov8n.pt")

input_folder = 'images/original_day'
output_folder = 'images/yolo_output_images'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(input_folder, filename)
        results = model(image_path)
        print(results)
        for img in results.imgs:
            img_with_boxes = results.render(img)
            output_image_path = os.path.join(output_folder, filename)
            Image.fromarray(img_with_boxes).save(output_image_path)

        print(f"Zdjęcie {filename} zostało przetworzone i zapisane w {output_image_path}")