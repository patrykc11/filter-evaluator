import time
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Input, concatenate, Lambda


input_dir = './dataset/png/low_light/'
gt_dir = './dataset/png/long/'
result_dir = './result_png/'

train_fns = glob.glob(gt_dir + '0*.png')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

ps = 512
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
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

loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
if status:
    print('loaded ' + str(status))

class ModelTraining:
    def __init__(self):
        self.g_loss = []
        self.learning_rate = tf.Variable(1e-4, dtype=tf.float32)
        self.lastepoch = tf.Variable(0, dtype=tf.int32)

    def update_lastepoch(self, new_epoch):
        self.lastepoch.assign(tf.maximum(self.lastepoch, new_epoch))

model_training = ModelTraining()

allfolders = glob.glob(result_dir + '*0')
for folder in allfolders:
    epoch_num = int(folder[-4:])
    model_training.update_lastepoch(epoch_num)

print(f'Ostatnia epoka: {model_training.lastepoch.numpy()}')

@tf.function
def train_step(in_image, gt_image, learning_rate):
    with tf.GradientTape() as tape:
        out_image = model(in_image, training=True)
        loss = loss_fn(gt_image, out_image)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.learning_rate.assign(learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def preprocess_image(image, ps):
    H, W = image.shape[0], image.shape[1]
    xx = np.random.randint(0, W - ps)
    yy = np.random.randint(0, H - ps)
    return image[yy:yy + ps, xx:xx + ps, :]

for epoch in range(model_training.lastepoch.numpy(), 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        model_training.learning_rate.assign(1e-5)

    for ind in np.random.permutation(len(train_ids)):
        train_id = train_ids[ind]
        in_path = f'{input_dir}{train_id:05d}_00.png'
        gt_path = f'{gt_dir}{train_id:05d}_00.png'

        in_image = load_image(in_path)
        gt_image = load_image(gt_path)

        in_image = preprocess_image(in_image, ps)
        gt_image = preprocess_image(gt_image, ps)

        in_image = np.expand_dims(in_image, axis=0)
        gt_image = np.expand_dims(gt_image, axis=0)

        if np.random.rand() < 0.5:
            in_image = np.flip(in_image, axis=1)
            gt_image = np.flip(gt_image, axis=1)
        if np.random.rand() < 0.5:
            in_image = np.flip(in_image, axis=2)
            gt_image = np.flip(gt_image, axis=2)
        if np.random.rand() < 0.5:
            in_image = np.transpose(in_image, (0, 2, 1, 3))
            gt_image = np.transpose(gt_image, (0, 2, 1, 3))

        in_image = np.minimum(in_image, 1.0)

        st = time.time()
        loss = train_step(in_image, gt_image, model_training.learning_rate)
        model_training.g_loss.append(loss.numpy())

        print(f"Epoch: {epoch}, Step: {cnt}, Loss: {loss.numpy()}, Time: {time.time() - st}")
        cnt += 1

    if epoch % save_freq == 0:
        if not os.path.isdir(result_dir + '%04d' % epoch):
            os.makedirs(result_dir + '%04d' % epoch)
        checkpoint.save(file_prefix=checkpoint_prefix)

print('Training complete!')