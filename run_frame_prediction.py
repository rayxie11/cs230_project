
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

model= keras.models.load_model(cur_folder + 'v1_greyscale_model')

# Select a few random examples from the dataset.
choices = np.random.choice(range(len(valSet)), size=5)
print("random selected: " + str(choices))
examples = valSet[np.random.choice(range(len(valSet)), size=5)]

# Iterate over the examples and predict the frames.
predicted_videos = []
idx = 1
for example in examples:
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))

    # Predict a new set of 10 frames.
    for i in range(10):
        # Extract the model's prediction and post-process it.
        frames = example[: 10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        new_predictions[i] = predicted_frame
        print("orignial video: " + str(len(original_frames)) + " predict video: " + str(len(new_predictions)))
        
    # Create and save GIFs for each of the ground truth/prediction images.
    frame_set = new_predictions
    # Construct a GIF from the selected video frames.
    current_frames = np.squeeze(frame_set)
    current_frames = current_frames[..., np.newaxis] * np.ones(3)
    current_frames = (current_frames * 255).astype(np.uint8)
    # https://theailearner.com/2021/05/29/creating-gif-from-video-using-opencv-and-imageio/
    #current_frames = cv2.cvtColor(current_frames, cv2.COLOR_BGR2RGB)
    current_frames = list(current_frames)
    
    # Construct a GIF from the frames.
    #with io.BytesIO() as gif:
    #    imageio.mimsave(gif, current_frames, "GIF", fps=5)
    #    predicted_videos.append(gif.getvalue())
    imageio.mimsave(cur_folder + str(idx) +  ".gif", current_frames, "GIF", fps=5)
    
    # Create and save GIFs for each of the ground truth/prediction images.
    frame_set = original_frames
    # Construct a GIF from the selected video frames.
    current_frames = np.squeeze(frame_set)
    current_frames = current_frames[..., np.newaxis] * np.ones(3)
    current_frames = (current_frames * 255).astype(np.uint8)
    # https://theailearner.com/2021/05/29/creating-gif-from-video-using-opencv-and-imageio/
    #current_frames = cv2.cvtColor(current_frames, cv2.COLOR_BGR2RGB)
    current_frames = list(current_frames)
    
    # Construct a GIF from the frames.
    #with io.BytesIO() as gif:
    #    imageio.mimsave(gif, current_frames, "GIF", fps=5)
    #    predicted_videos.append(gif.getvalue())
    imageio.mimsave(cur_folder + str(idx) + "orig.gif", current_frames, "GIF", fps=5)
    idx += 1
