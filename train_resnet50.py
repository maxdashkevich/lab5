from google.colab import drive
drive.mount('/content/drive')

!ls ./xray/

!mkdir xray

!cp -r /content/drive/'My Drive'/xray/* ./xray/

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard

import argparse
import gc
import random
import math
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from datetime import datetime

LOG_DIR = './logs'
SHUFFLE_BUFFER = 10
BATCH_SIZE = 100
NUM_CLASSES = 50
PARALLEL_CALLS=4
RESIZE_TO = 224
TRAINSET_SIZE = 5216
VALSET_SIZE=624

def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }
    example = tf.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize_images(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image'], example['image/class/label']


def normalize(image, label):
    return tf.image.per_image_standardization(image), label

def resize(image, label):
    return tf.image.resize_images(image, tf.constant([RESIZE_TO, RESIZE_TO])), label

def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .map(resize)\
        .map(normalize)\
        .shuffle(buffer_size=5 * batch_size)\
        .repeat()\
        .batch(batch_size)\
        .prefetch(2 * batch_size)

def augmented_train(image, label):
    image = tf.image.random_flip_left_right(image)
    if random.random() > 0.5:
        image = tf.contrib.image.rotate(image, 90 * math.pi / 180, interpolation='BILINEAR')
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    # image = tf.image.random_crop(image, size=[50, 50, 3], seed=None, name=None)
    tf.image.central_crop(image, central_fraction=0.5)
    return image,label

class Validation(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, validation_files, batch_size):
        self.log_dir = log_dir
        self.batch_size = batch_size
        validation_dataset = create_dataset(validation_files, batch_size)
        self.validation_images, validation_labels = validation_dataset.make_one_shot_iterator().get_next()
        self.validation_labels = tf.one_hot(validation_labels, NUM_CLASSES)

    def on_epoch_end(self, epoch, logs=None):

        result = self.model.evaluate(
            self.validation_images,
            self.validation_labels,
            steps=int(np.ceil(VALSET_SIZE / float(BATCH_SIZE)))
        )
        print('The average valid loss for epoch {} is {:7.2f} '.format(
            epoch, result[0]
        ))
        callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, update_freq='epoch', batch_size=self.batch_size)

        callback.set_model(self.model)
        callback.on_epoch_end(epoch, {
            'val_' + self.model.metrics_names[i]: v for i, v in enumerate(result)
        })
        gc.collect()

def build_model():
    model = tf.keras.applications.ResNet50(input_shape=(224,224,3),
                                           include_top=False,
                                           weights='imagenet')
    return model


base_model = build_model()
model = tf.keras.models.Sequential([
  base_model,
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)
])
model.summary()

train_path = './xray/train*'
test_path = './xray/test*'
def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.3
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate
def step_decay_warmup(epoch):
   initial_lrate = 0.001
   drop = 0.3
   epochs_drop = 10.0
   warmup_batches = 10.0
   if epoch <= warmup_batches:
     lrate = epoch * initial_lrate/warmup_batches
   else:
     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate
def exp_decay(epoch):
   initial_lrate = 0.001
   k = 0.4
   lrate = initial_lrate * math.exp(-k*epoch)
   return lrate
def exp_decay_warmup(epoch):
   initial_lrate = 0.001
   k = 0.25
   warmup_batches = 10.0
   if epoch <= warmup_batches:
     lrate = epoch * initial_lrate/warmup_batches
   else:
     lrate = initial_lrate * math.exp(-k*epoch)
   return lrate
lrate = tf.keras.callbacks.LearningRateScheduler(exp_decay_warmup)

train_dataset = create_dataset(glob.glob(train_path), BATCH_SIZE)
train_images, train_labels = train_dataset.make_one_shot_iterator().get_next()
train_labels = tf.one_hot(train_labels, NUM_CLASSES)


model.compile(
    optimizer=keras.optimizers.sgd(momentum=0.9),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
    target_tensors=[train_labels]
)

log_dir='{}/xray-{}'.format(LOG_DIR, datetime.now())
model.fit(
    (train_images, train_labels),
    epochs=80,
    steps_per_epoch=int(np.ceil(TRAINSET_SIZE / float(BATCH_SIZE))),
    callbacks=[
        lrate,
        tf.keras.callbacks.TensorBoard(log_dir),
        Validation(log_dir, validation_files=glob.glob(test_path), batch_size=BATCH_SIZE)
    ]
)

!rm -Rf logs/*

!ls logs

!mkdir logs

!nvidia-smi

! wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
! unzip -o ngrok-stable-linux-amd64.zip

# kill all running ngrok instances
!pkill -f ngrok
# !pkill -f tensorboard

LOG_DIR = './logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

# execute ngrok
get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

!cp -r ./logs/* /content/drive/'My Drive'/xray_logs

!kill 1189

!top
