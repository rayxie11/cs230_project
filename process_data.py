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

dataset = []

def video_data_gen_sub(img_height = 256, img_width = 256, max_frame = 20):
    '''
    This function generates a subsection of dataset for videos and save them as npy files
    Args:
        feature_extractor: Feature extractor
        img_height: Dataset image height
        img_width: Dataset image width
        frames: Number of frames in each sample
    Returns:
        None
    '''

    dataset = []
    # Retrieve data from dataset
    for i in range(len(genres)):
        # Loop through all images within one genre
        cur_path = subset_data + genres[i]
        print(cur_path)
        image_files = os.listdir(cur_path)
        videoFramesMap = buildPerVideoFramesMap(image_files, max_frame)

        for k, (video, frames) in tqdm(enumerate(videoFramesMap.items()), position=0, desc="Current Genre: " + str(genres[i])):
            j = 0
            # Single datapoint storage list
            singleVideo = np.zeros((len(frames), img_height, img_width, 3))

            for frame in frames:
                # print(frame)
                framePath = cur_path + "/" + frame
                # print(framePath)
                frame = cv2.imread(cur_path + "/" + frame)
                # print(frame)
                # print("frame size before resize: " )
                # print(frame.shape)
                frame = cv2.resize(frame, (img_width, img_height))
                # print("frame size after resize: " )
                # print(frame.shape)

                singleVideo[j] = frame
                j += 1

            dataset.append(singleVideo)

        # cur genre end


    if not os.path.exists(save_path_sub):
        os.makedirs(save_path_sub)

    dataset = np.array(dataset)
    print("dataset shape" + str(dataset.shape))
    np.save(save_path_sub + '/video', dataset)

# Build map from video to all frames according to max_frame limit
def buildPerVideoFramesMap(images, max_frame):
    videoFramesMap = {}
    #rgb/tap/2TU7dE_9Vi4_341_0195.jpg
    for image in images:
        # Skip hidden files
        if image[0] == '.':
            continue
        videoName = image[:image.rfind('_')]
        indexStr = image[image.rfind('_') + 1 : image.rfind('.')]
        index = int(image[image.rfind('_') + 1 : image.rfind('.')])
        # print(videoName, indexStr, index)
        if videoName not in videoFramesMap:
            videoFramesMap[videoName] = []

        # print(type(index), type(max_frame))
        if index <= max_frame:
            videoFramesMap[videoName].append(image)
    print(len(videoFramesMap))
    return videoFramesMap

video_data_gen_sub(img_height=img_height, img_width=img_width)
