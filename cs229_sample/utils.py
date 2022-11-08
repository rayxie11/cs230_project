'''
This file contains all the util functions that are needed to extract information from dataset
'''
from copyreg import pickle
import os
import json
import numpy as np
import cv2
import sys
import math
import pandas as pd


'''
This function converts labels {0,10} into onehot labels
Args: 
    n: An int label
Returns: 
    A numpy array of onehot labels
Example: 
    6 -> [0,0,0,0,0,0,1,0,0,0,0]
'''
def convert_onehot(n):
    zeros = np.zeros(11)
    zeros[n] = 1

    return zeros

'''
This function converts all labels in dataset to onehot labels
Args:
    file_location: A string for the file location of dataset labels
Returns:
    file: A list of file names (n,)
    label: A 2d numpy array of file labels (n,11)
'''
def onehot_labels(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    f.close()

    label = []
    file = []
    for key in data:
        file.append(key)
        label.append(convert_onehot(data[key]))
    label = np.vstack(label)

    np.save("label.npy",label)

    return file, label

'''
This function matches labels with actual actions
Args:
    file_location: A string for the file location matching labels to actions
Returns:
    A dictionary with onehot labels (tuple) as keys and actions (string) as values
'''
def matching(file_location):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    f = open(parent_dir+'/dataset/'+file_location)
    data = json.load(f)
    f.close()

    match_dict = dict()
    for key in data:
        label = tuple(convert_onehot(int(key)))
        match_dict[label] = data[key]
    
    return match_dict

'''
This function extracts the first frame of video
Args:
    video_location: A string for the location of video to be captured
    conv: True or False if using a Conv2D layer in NN
Returns:
    A 1d greyscale vector reshaped from 3d matrix (22528,)
    OR
    A 3d matrix of pixels
Exception:
    First frame capture fails
'''
def extract_first_frame(video_location,conv):
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    if success:
        if conv:
            return image
        else:
            grey_scale = np.mean(image,axis=-1)
            return grey_scale.reshape(-1)
    else:
        raise Exception("First frame not captured")


def extract_first_frame_cnn(video_location):
    vidcap = cv2.VideoCapture(video_location)
    success, image = vidcap.read()
    if success:
        grey_scale = np.mean(image,axis=-1)
        #img = np.repeat(np.expand_dims(grey_scale,-1),3,axis=-1)
        #cv2.imwrite("test.jpg",img)
        return grey_scale
    else:
        raise Exception("First frame not captured")


'''
This function extracts the first frame of all video data and saves as an npy file
Args:
    file: A list of file names returned from onehot_labels
    conv: True or False if using a Conv2D layer in NN
Returns:
    A 2d numpy array of greyscale vectors (n,22528)
    OR
    A 4d numpy array of pixels (n,pixels)
'''
def frame_to_data(file,conv):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    dir = parent_dir + '/dataset/examples/'
    vid_end = '.mp4'

    img = []
    for name in file:
        img.append(extract_first_frame(dir+name+vid_end,conv))
    
    if conv:
        np.save('img_conv.npy',img)
    else:
        np.save('img.npy', np.vstack(img))

    return img


def frame_to_data_cnn(file):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    dir = parent_dir + '/dataset/examples/'
    vid_end = '.mp4'

    img = []
    for index, name in enumerate(file):
        if index == 15000:
            break
        img.append(extract_first_frame_cnn(dir+name+vid_end))


    imgs = np.stack(img, axis=0)
    imgs = np.expand_dims(imgs, axis=3)
    np.save('images_cnn.npy', imgs)

    return imgs

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


'''
This function reads the motion data from dataset
'''
def read_motion(file_loc):
    data = np.load(file_loc,allow_pickle=True)
    flattened = np.zeros((16,2*18))
    for i in range(len(data)):
        for j in range(18):
            if j in data[i]:
                flattened[i][j*2] = data[i][j][0]
                flattened[i][j*2+1] = data[i][j][1]
    return flattened

def read_motion_fill_mean(file_loc):
    data = np.load(file_loc,allow_pickle=True)
    data_x = []
    data_y = []
    for elem in data:
        x_dict = dict()
        y_dict = dict()
        for key in elem.keys():
            x_dict[key] = elem[key][0]
            y_dict[key] = elem[key][1]
        data_x.append(x_dict)
        data_y.append(y_dict)
    col = list(range(0,18))
    df_x = pd.DataFrame(data_x,columns=col)
    df_y = pd.DataFrame(data_y,columns=col)
    for i in col:
        if df_x[i].isnull().all():
            df_x[i].fillna(value=0,inplace=True)
        else:
            x_mean = df_x[i].mean()
            df_x[i].fillna(value=x_mean,inplace=True)
        if df_y[i].isnull().all():
            df_y[i].fillna(value=0,inplace=True)
        else:
            y_mean = df_y[i].mean()
            df_y[i].fillna(value=y_mean,inplace=True)
    x_ls = df_x.values.tolist()
    y_ls = df_y.values.tolist()
    flattened = np.zeros((16,2*18))
    flattened[:,0::2] = x_ls
    flattened[:,1::2] = y_ls
    return flattened

def read_motion_fill_interpolate(file_loc):
    data = np.load(file_loc,allow_pickle=True)
    data_x = []
    data_y = []
    for elem in data:
        x_dict = dict()
        y_dict = dict()
        for key in elem.keys():
            x_dict[key] = elem[key][0]
            y_dict[key] = elem[key][1]
        data_x.append(x_dict)
        data_y.append(y_dict)
    col = list(range(0,18))
    df_x = pd.DataFrame(data_x,columns=col)
    df_y = pd.DataFrame(data_y,columns=col)
    for i in col:
        if df_x[i].isnull().all():
            df_x[i].fillna(value=0,inplace=True)
        if df_y[i].isnull().all():
            df_y[i].fillna(value=0,inplace=True)
    df_x = df_x.interpolate(limit_direction='both')
    df_y = df_y.interpolate(limit_direction='both')
    x_ls = df_x.values.tolist()
    y_ls = df_y.values.tolist()
    flattened = np.zeros((16,2*18))
    flattened[:,0::2] = x_ls
    flattened[:,1::2] = y_ls
    return flattened

def read_motion_fill_pad(file_loc):
    data = np.load(file_loc,allow_pickle=True)
    data_x = []
    data_y = []
    for elem in data:
        x_dict = dict()
        y_dict = dict()
        for key in elem.keys():
            x_dict[key] = elem[key][0]
            y_dict[key] = elem[key][1]
        data_x.append(x_dict)
        data_y.append(y_dict)
    col = list(range(0,18))
    df_x = pd.DataFrame(data_x,columns=col)
    df_y = pd.DataFrame(data_y,columns=col)
    df_x = df_x.pad(limit=4)
    df_y = df_y.pad(limit=4)
    df_x = df_x.bfill(limit=4)
    df_y = df_y.bfill(limit=4)
    # df_x = df_x.pad()
    # df_y = df_y.pad()
    df_x.fillna(value=0, inplace=True)
    df_y.fillna(value=0, inplace=True)
    x_ls = df_x.values.tolist()
    y_ls = df_y.values.tolist()
    flattened = np.zeros((16,2*18))
    flattened[:,0::2] = x_ls
    flattened[:,1::2] = y_ls
    return flattened

def motion_pts_all(file):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.split(cur_dir)[0]
    dir = parent_dir + '/dataset/examples/'
    npy_end = '.npy'
    result = []
    for name in file:
        #result.append(read_motion(dir+name+npy_end))
        #result.append(read_motion_fill_mean(dir+name+npy_end))
        #result.append(read_motion_fill_interpolate(dir+name+npy_end))
        result.append(read_motion_fill_pad(dir + name + npy_end))
    np.save('motion_fill_pad.npy',result)

    return np.shape(result)


def main(cnn=False):

    file, label = onehot_labels('annotation_dict.json')
    print(len(file))
    #motion_shape = motion_pts_all(file)
    #print(motion_shape)

    # check nan in file
    #data = np.load('motion_fill_interpolate.npy',allow_pickle=True)
    #print(np.isnan(data).any())
    '''
    if cnn:
        img = frame_to_data_cnn(file)
        print(np.shape(img))
    else:
        img = frame_to_data(file,True)
        print(np.shape(img))
    '''



if __name__ == '__main__':
    '''
    if len(sys.argv) > 1:
        cnn = True
    else:
        cnn = False
    '''
    main()
