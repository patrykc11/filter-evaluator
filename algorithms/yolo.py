# https://github.com/ultralytics/ultralytics?tab=readme-ov-file
# wytrenowany model yolo v8 na danych z coco

from ultralytics import YOLO
import os
import shutil
import cv2

model = YOLO("ready-models/yolov8x.pt")

input_folder = 'images/original_night'
output_folder = 'images/custom_filter_yolo_8n'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        results = model(image_path, classes=[0])
        annotated_frame = results[0].plot()
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, annotated_frame)