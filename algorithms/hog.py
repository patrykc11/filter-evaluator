# https://github.com/ThuraAung1601/human-detection-hog
# https://debuggercafe.com/opencv-hog-for-accurate-and-fast-person-detection/

import cv2
import glob as glob
import os
import shutil

red = 0
green = 0

class HogDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_and_save(self, image_path, output_dir):
        global red, green
        image = cv2.imread(image_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects, weights = self.hog.detectMultiScale(img_gray)
        
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                color = (0, 0, 255)  # Red if confidence < 70%
                red += 1
            else:
                color = (0, 255, 0)  # Green if confidence >= 70%
                green += 1
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)


def test():
  detector = HogDetector()
  image_paths = glob.glob('./images/original_day/*.jpg') + glob.glob('./images/original_day/*.png')
  output_dir = './images/hog_output_images'
  shutil.rmtree(output_dir)
  os.makedirs(output_dir, exist_ok=True)

  for image_path in image_paths:
      detector.detect_and_save(image_path, output_dir)
  print(red)
  print(green)

test()