from fastapi import HTTPException
from PIL import Image
from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8x.pt")

def detect_human(image: Image):
    try:
        results = model(image, classes=[0])
        annotated_frame = Image.fromarray(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB))
        
        return annotated_frame
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image")