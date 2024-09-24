import glob
import numpy as np
import rawpy
from PIL import Image
import os

gt_dir = './dataset/Sony/short/'
output_dir = './dataset/png/long'

folder_path = output_dir

for filename in os.listdir(folder_path):
    name, ext = os.path.splitext(filename)
    if len(name) > 4:
        new_name = name[:-4] + ext
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_name}')
    else:
        print(f'Skipped: {filename} (name too short)')

print('Renaming complete!')