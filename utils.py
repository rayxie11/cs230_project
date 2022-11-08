'''
This file contains all the util functions that are needed to extract information from the given dataset
'''
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import tensorflow as tf
from numba import jit
from skimage.io import imread_collection


# Choreography genre name paths
path = "C:/Users/ray_s/Desktop/cs230_project/dataset"
genres = os.listdir(path+"/video")


def get_min_segment_val():
    '''
    This function returns the minimum length of one sample in the given dataset
    Args:
        None
    Returns:
        min_val: min length of one sample
        idx_dict: dictionary of indices of the last image in sample
    '''
    # Set initial parameters
    vid_min_size = 10000
    idx_dict = {}

    for genre in genres:
        idx_dict[genre] = [0]
        cur_path = path + "/video/" + genre
        dir_names = os.listdir(cur_path)

        # Loop through each video
        prev_header = dir_names[0].split('_')[0]

        # Keep track of current video length
        cur_vid_len = 0

        for i in range(len(dir_names)):
            cur_header = dir_names[i].split('_')[0]
            if prev_header == cur_header:
                # Increase current video length
                cur_vid_len += 1
            else:
                # Update minimum video length
                vid_min_size = min(vid_min_size, cur_vid_len)

                # Add index to idx_dict
                idx_dict[genre].append(i)

                # Reset count and prev_header
                cur_vid_len = 1
                prev_header = dir_names[i].split('_')[0]
        
        # Add the last index to idx_dict
        idx_dict[genre].append(len(dir_names)-1)
    
    return vid_min_size, idx_dict


def video_data_gen(img_height = 720, img_width = 1080):
    '''
    This function generates the full dataset for videos
    Args:
        img_height: Dataset image height
        img_width: Dataset image width
    Returns:
        X: Video data (num_samples,img_dim_x,img_dim_y,3)
        Y: One-hot label (num_samples,label_dim)
    '''
    '''
    dataset_full = tf.keras.preprocessing.image_dataset_from_directory(
        path + "/video",
        labels = 'inferred',
        label_mode = 'int'
    )
    '''
    # Return variables
    X = []
    Y = []

    
    vid_min_size, idx_dict = get_min_segment_val()

    cur_path = path + "/video/" + genres[0]
    dataset_full = tf.keras.utils.image_dataset_from_directory(
        cur_path,
        labels = None,
        image_size = (img_height, img_width),
        color_mode = "rgb",
        crop_to_aspect_ratio = True
    )
    


    for images in dataset_full.take(-1):
        x = images.numpy()
        for img in x:
            print(img)
            disp = Image.fromarray(img,'RGB')
            disp.show()
            break
    #print(x[0])
    
    

    print(dataset_full)
    '''
    cur_path = path + "/video/" + genres[0]
    c = imread_collection(cur_path+"/*.jpg")
    c = c.concatenate()
    disp = Image.fromarray(c[0])
    disp.show()
    '''




    '''
    # Get smallest video segment value
    vid_min_size = 10000
    for i in range(len(genres)):
        cur_path = path + "/video/" + genres[i]
        dir_names = os.listdir(cur_path)

        # Loop through each video
        prev_header = dir_names[0].split('_')[0]

        # Keep track of current video length
        cur_vid_len = 0

        for pic_name in dir_names:
            cur_header = pic_name.split('_')[0]
            if prev_header == cur_header:
                # Get current image
                #cur_img = cv2.imread(path + "/video/" + genres[i] + "/" + pic_name)
                cur_img = tf.keras.preprocessing.image.load_img(path + "/video/" + genres[i] + "/" + pic_name, target_size=(256,256,3))
                cur_img_arr = tf.keras.preprocessing.image.img_to_array(cur_img)

                # Add image to current video
                single_video.append(cur_img_arr)

                # Increase current video length
                cur_vid_len += 1
            else:
                # Add current one-hot label
                cur_label = np.zeros(len(genres))
                cur_label[i] = 1
                Y_full.append(cur_label)

                # Add current video to X_full
                X_full.append(single_video)

                # Update minimum video length
                vid_min_size = min(vid_min_size, cur_vid_len)

                # Start recording a new video
                #cur_img = cv2.imread(path + "/video/" + genres[i] + pic_name)
                #single_video = [cur_img]
                cur_img = tf.keras.preprocessing.image.load_img(path + "/video/" + genres[i] + "/" + pic_name, target_size=(256,256,3))
                cur_img_arr = tf.keras.preprocessing.image.img_to_array(cur_img)
                single_video = [cur_img_arr]
                cur_vid_len = 1
                prev_header = pic_name.split('_')[0]

    
    # Full video data
    X_full = []

    # Y labels
    Y_full = []

    # Crop all videos to minimum size
    vid_min_size = 10000

    # Loop through each genre
    for i in tqdm(range(len(genres))):
    #for i in range(len(genres)):
        cur_path = path + "/video/" + genres[i]
        dir_names = os.listdir(cur_path)

        # Loop through each video
        prev_header = dir_names[0].split('_')[0]
        single_video = []

        # Keep track of current video length
        cur_vid_len = 0

        for pic_name in tqdm(dir_names, position=0, leave=True):
        #for pic_name in dir_names:
            cur_header = pic_name.split('_')[0]
            if prev_header == cur_header:
                # Get current image
                #cur_img = cv2.imread(path + "/video/" + genres[i] + "/" + pic_name)
                cur_img = tf.keras.preprocessing.image.load_img(path + "/video/" + genres[i] + "/" + pic_name, target_size=(256,256,3))
                cur_img_arr = tf.keras.preprocessing.image.img_to_array(cur_img)

                # Add image to current video
                single_video.append(cur_img_arr)

                # Increase current video length
                cur_vid_len += 1
            else:
                # Add current one-hot label
                cur_label = np.zeros(len(genres))
                cur_label[i] = 1
                Y_full.append(cur_label)

                # Add current video to X_full
                X_full.append(single_video)

                # Update minimum video length
                vid_min_size = min(vid_min_size, cur_vid_len)

                # Start recording a new video
                #cur_img = cv2.imread(path + "/video/" + genres[i] + pic_name)
                #single_video = [cur_img]
                cur_img = tf.keras.preprocessing.image.load_img(path + "/video/" + genres[i] + "/" + pic_name, target_size=(256,256,3))
                cur_img_arr = tf.keras.preprocessing.image.img_to_array(cur_img)
                single_video = [cur_img_arr]
                cur_vid_len = 1
                prev_header = pic_name.split('_')[0]

    print(len(X_full))
    print(len(Y_full))
    print(vid_min_size)

    # Return variables
    X = []
    Y = []

    # Crop each sample into same length
    for x, y in zip(X_full, Y_full):
        #print(x)
        x = np.array(x)

        # See how many segments can be extracted w.r.t vid_min_size
        reps = len(x) // vid_min_size

        # If remainder is less than 1/4 of vid_min_size, fill with 0
        fill_0 = False
        remainder = len(x) % vid_min_size
        if (remainder >= (3 * (vid_min_size // 4))):
            fill_0 = True

        print(remainder,len(x),reps,fill_0)

        # Add segments into X and Y
        for j in range(reps):
            X.append(x[j*vid_min_size:(j+1)*vid_min_size])
            Y.append(y)

        # Fill with 0 if need to
        if fill_0:
            single_zeros = np.zeros_like(x[0])
            end_x = x[(j+1)*vid_min_size:]
            num_to_fill = vid_min_size - len(end_x)
            all_zeros = np.tile(single_zeros, (num_to_fill,1))
            print(end_x.shape)
            print(all_zeros.shape)
            full_x = np.concatenate((end_x, all_zeros))
            X.append(full_x)
            Y.append(y)


    print(len(X))
    print(len(Y))
    '''


if __name__ == '__main__':
    video_data_gen()