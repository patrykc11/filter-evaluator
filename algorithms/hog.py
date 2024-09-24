# lots of false positive 
# https://github.com/ThuraAung1601/human-detection-hog
# https://debuggercafe.com/opencv-hog-for-accurate-and-fast-person-detection/
# https://thedatafrog.com/en/articles/human-detection-video/

import cv2
import glob as glob
import os
import shutil
import sys
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import time

class HogDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_and_save(self, image_path):
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=max(2840, image.shape[1])) #Resizing our image also improves the overall accuracy of our pedestrian detection (i.e., less false-positives).
        rects, weights = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            index = np.where((rects[:, 0] == xA) & (rects[:, 1] == yA) & (rects[:, 2] == xB) & (rects[:, 3] == yB))[0][0]

            weight = weights[index]

            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            text = f"{weight * 100:.2f}%"
            cv2.putText(image, text, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image

input_folder = 'images/' + sys.argv[1]
output_folder = input_folder + '_hog'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        detector = HogDetector()

        start_time = time.time()
        annotated_frame = detector.detect_and_save(image_path)
        inference_time = time.time() - start_time

        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, annotated_frame)

        print(f"Inference time for {filename}: {inference_time:.3f} seconds")

