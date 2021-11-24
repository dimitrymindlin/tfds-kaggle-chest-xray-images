from pathlib import Path
import tensorflow as tf

PRE_WEIGHTS = '../chexnet_checkpoint/checkpoint'

import tensorflow_datasets as tfds
from src.kaggleChestXrayImages import Kagglechestxrayimages

(train, val, test), ds_info = tfds.load(
            'Kagglechestxrayimages',
            split=['train[:88%]', 'train[88%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

"""## NEW Model Definition of DenseNet with new layers for Kaggle chest-xray dataset"""
from chexnet.model.chexnet import CheXNet
from chexnet.configs.config import chexnet_config
from datetime import datetime

# Load model
train_base = chexnet_config['train']['train_base']
old_model = CheXNet(chexnet_config, train_base=train_base).model()
old_model.load_weights("../chexnet_checkpoint/2021-11-22--00.20")

import tensorflow.keras as K

# Delete classification layer
old_model.layers.pop() 

# Freeze some layers
for layer in old_model.layers[:149]:
    layer.trainable = False
for layer in old_model.layers[149:]:
    layer.trainable = True

# Create new Model with new output layers
model = K.Sequential()

kernel_init = "glorot_uniform"

model.add(old_model)
model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(256, activation='relu',
kernel_initializer=kernel_init))
model.add(K.layers.Dropout(0.7))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu',
kernel_initializer=kernel_init))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu',
kernel_initializer=kernel_init))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(10, activation='relu',
kernel_initializer=kernel_init))
model.add(K.layers.Dropout(0.1))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(2, activation='softmax',
kernel_initializer=kernel_init))

optimizer = K.optimizers.Adam(chexnet_config["train"]["learn_rate"])
loss = K.losses.BinaryCrossentropy(from_logits=False)
metric_auc = K.metrics.AUC(curve='ROC',multi_label=True, num_labels=len(chexnet_config["data"]["class_names"]), from_logits=False)
metric_accuracy = K.metrics.Accuracy()

"""## Train Parameters and Callbacks"""

# callbacks to avoid overfitting
checkpoint_filepath = 'checkpoint/chexnet_transfer_learning/'
checkpoint_callback = K.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=True)
early_stopping = K.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=chexnet_config['train']['early_stopping_patience'])

# Tensorboard Callback and config logging
log_dir = 'logs/chexnet_checkpoint/'+ datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = K.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(optimizer=optimizer,loss=loss, metrics=[metric_auc, metric_accuracy])

def normalize_img(image, label):
  return tf.image.resize_with_pad(
        image,
        chexnet_config['data']['image_height'], 
        chexnet_config['data']['image_width'], 
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False), label


def preprocess(ds):
  ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
  # For true randomness, we set the shuffle buffer to the full dataset size.
  ds = ds.shuffle(ds_info.splits['train'].num_examples)
  # Batch after shuffling to get unique batches at each epoch.
  ds = ds.batch(chexnet_config['train']['batch_size'])
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

"""## Model Training (fine tuning)"""

model.fit(preprocess(train),
          epochs=5,
          callbacks=[checkpoint_callback, early_stopping, tensorboard_callback],
          validation_data=preprocess(val))
#model.save('kerasChestXray.h5') TODO: Is that needed?