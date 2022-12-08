
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import os
import io
import imageio
from tqdm.auto import tqdm
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

cur_folder = "/home/cs230/"
subset_data = "/home/cs230/rgb_sub/"
fullset_data = "/home/cs230/rgb/"
save_path_sub = cur_folder + "generated_dataset/video_sub"
#save_path_full = cur_folder + "generated_dataset/image_full"
genres = sorted(os.listdir(fullset_data))

#img_height = 720
#img_width = 1080
img_height = 60
img_width = 90
# Split training/validation 
videoPath = save_path_sub + '/video.npy'
dataset = np.load(videoPath)

indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)

trainIndex = indexes[:int(0.9 * dataset.shape[0])]
valIndex = indexes[int(0.9 * dataset.shape[0]):]

trainSet = dataset[trainIndex]
valSet = dataset[valIndex]

print("training set shape: " + str(trainSet.shape))
print("val set shape: " + str(valSet.shape))

# Normalize
trainSet = trainSet / 255
valSet = valSet / 255

# Helper to shift frames so X is 0 to n -1, Y is 1 to n
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

xTrain, yTrain = create_shifted_frames(trainSet)
xVal, yVal = create_shifted_frames(valSet)

print("xTrain, yTrain shape: " + str(xTrain.shape) + str(yTrain.shape))
print("xVal, yVal shape: " + str(xVal.shape) + str(yVal.shape))

# Model construction
print(xTrain.shape[2:])

inp = layers.Input(shape=(None, *xTrain.shape[2:]))

# 3 ConvLSTMM2D layers with batch norm,
# followed by 'Conv3D`

x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5,5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3,3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1,1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
output = layers.Conv3D(
    filters=1,
    kernel_size=(3,3,3),
    activation="sigmoid",
    padding="same"
)(x)

model = keras.models.Model(inp, output)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
             )

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

epochs=10
batch_size=5

model.summary()
print(xTrain.shape, yTrain.shape)


model.fit(
    x=xTrain,
    y=xTrain,
    batch_size = batch_size,
    epochs=epochs,
    validation_data=(xVal, yVal),
    callbacks=[early_stopping, reduce_lr],
    verbose=True
)

model.save(cur_folder + 'v1_greyscale_model')
