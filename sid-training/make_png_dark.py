import os
import glob
import cv2
import numpy as np

well_lit_dir = './dataset/png/long'
low_light_dir = './dataset/png/low_light'

os.makedirs(low_light_dir, exist_ok=True)

image_files = glob.glob(os.path.join(well_lit_dir, '*.png'))

brightness_factor = 0.2

contrast_factor = 0.5
noise_mean = 0 
noise_stddev = 10  

def add_noise(image, mean=0, stddev=10):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

for image_path in image_files:
    image = cv2.imread(image_path)

    darkened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    darkened_image = cv2.convertScaleAbs(darkened_image, alpha=contrast_factor, beta=0)

    file_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(low_light_dir, file_name), darkened_image)

print('Przekształcanie zakończone!')
