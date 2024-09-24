import os
import shutil
import random

def create_subset(source_dir, target_dir, num_images):
    os.makedirs(target_dir, exist_ok=True)
    image_files = os.listdir(source_dir)
    random.shuffle(image_files)
    for file_name in image_files[:num_images]:
        src_path = os.path.join(source_dir, file_name)
        dst_path = os.path.join(target_dir, file_name)
        shutil.copyfile(src_path, dst_path)

source_dir_train_night = 'images/train/night'
source_dir_train_day = 'images/train/day'
source_dir_val_night = 'images/val/night'
source_dir_val_day = 'images/val/day'

target_dir_train_night = 'images_subs/train/night'
target_dir_train_day = 'images_subs/train/day'
target_dir_val_night = 'images_subs/val/night'
target_dir_val_day = 'images_subs/val/day'

num_images_train = 10000
num_images_val = 4000

os.makedirs(target_dir_train_night, exist_ok=True)
os.makedirs(target_dir_train_day, exist_ok=True)
os.makedirs(target_dir_val_night, exist_ok=True)
os.makedirs(target_dir_val_day, exist_ok=True)

create_subset(source_dir_train_night, target_dir_train_night, num_images_train)
create_subset(source_dir_train_day, target_dir_train_day, num_images_train)
create_subset(source_dir_val_night, target_dir_val_night, num_images_val)
create_subset(source_dir_val_day, target_dir_val_day, num_images_val)

print('Subset creation completed.')