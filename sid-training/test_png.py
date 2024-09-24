import time
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate

test_input_dir = './dataset/png/images_brugia/'
test_result_dir = './results_png/images_brugia/'
checkpoint_dir = './checkpoints/'

os.makedirs(test_result_dir, exist_ok=True)

def adjust_brightness(image, threshold=0.9, factor=0.5):
    bright_pixels = image > threshold
    image[bright_pixels] = image[bright_pixels] * factor
    return image

def load_image(image_path):
    target_size = (4256, 2848)
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image, dtype=np.float32) / 255.0
    return image

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

def network(input_tensor):
    conv1 = Conv2D(32, (3, 3), activation=lrelu, padding='same')(input_tensor)
    conv1 = Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(64, (3, 3), activation=lrelu, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (3, 3), activation=lrelu, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = Conv2D(256, (3, 3), activation=lrelu, padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv4)
    pool4 = MaxPooling2D((2, 2), padding='same')(conv4)

    conv5 = Conv2D(512, (3, 3), activation=lrelu, padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation=lrelu, padding='same')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(256, (3, 3), activation=lrelu, padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation=lrelu, padding='same')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(128, (3, 3), activation=lrelu, padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation=lrelu, padding='same')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(64, (3, 3), activation=lrelu, padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation=lrelu, padding='same')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    conv9 = Conv2D(32, (3, 3), activation=lrelu, padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation=lrelu, padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation=None, padding='same')(conv9)
    return conv10

input_tensor = Input(shape=(None, None, 3))
output_tensor = network(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

def save_image(image, path):
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)

test_files = glob.glob(test_input_dir + '*.png')
for test_file in test_files:
    in_image = load_image(test_file)
    in_image = np.expand_dims(in_image, axis=0)

    start_time = time.time()
    output = model(in_image, training=False)
    output = output[0].numpy()
    elapsed_time = time.time() - start_time

    output = np.minimum(np.maximum(output, 0), 1)

    base_name = os.path.basename(test_file)
    result_path = os.path.join(test_result_dir, base_name)
    save_image(output, result_path)

    print(f'Processed {base_name} in {elapsed_time:.3f} seconds')

print('Testing complete!')
