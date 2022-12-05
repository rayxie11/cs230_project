'''
This file contains all the util functions that are needed to extract information from the given dataset
'''
import os
import numpy as np
import cv2
from tqdm.auto import tqdm
from tensorflow import keras


# Get dataset path and genres list
#path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/dataset"
cur_folder = "/home/cs230"
subset_data = "/home/cs230/rgb_sub/"
fullset_data = "/home/cs230/rgb/"
save_path = cur_folder + "/generated_dataset/image_sub"
genres = os.listdir(subset_data)


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
    genre_idx_dict = {}
    sample_tot = 0

    for genre in genres:
        genre_idx_dict[genre] = [0]
        cur_path =  fullset_data + genre
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
                sample_tot += 1

                # Update minimum video length
                vid_min_size = min(vid_min_size, cur_vid_len)

                # Add index to idx_dict
                genre_idx_dict[genre].append(i)

                # Reset count and prev_header
                cur_vid_len = 1
                prev_header = dir_names[i].split('_')[0]
        
        # Add the last index to idx_dict
        genre_idx_dict[genre].append(len(dir_names)-1)
    
    return vid_min_size, genre_idx_dict, sample_tot


def build_feature_extractor(img_height = 256, img_width = 256):
    '''
    This function builds a feature extractor using features produced by InceptionV3 
    before the last layer
    Args:
        img_height: Input image height
        img_width: Input image width
    Returns:
        keras.Model object
    '''
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_height, img_width, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((img_height, img_width, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def video_data_gen_sub(feature_extractor, img_height = 256, img_width = 256, frames = 20):
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
    # Storage variables
    X = []
    Y = []

    # Retrieve data from dataset
    for i in range(len(genres)):
        # Set one-hot label
        label = np.zeros(len(genres))
        label[i] = 1

        # Single datapoint storage list
        X_single = []

        # Loop through all images within one genre
        cur_path = subset_data + genres[i]
        dir_names = os.listdir(cur_path)
        for j in tqdm(range(len(dir_names)), position=0, desc="Curent genre: "+str(genres[i])):
            # Load single image
            img = cv2.imread(cur_path + "/" + dir_names[j])
            if img is None:
                continue
            img = cv2.resize(img, (img_width,img_height))

            # Extract image features
            extraction = feature_extractor.predict(np.expand_dims(img,axis=0))
            extraction = extraction.flatten()
            
            # Add to X_single
            X_single.append(extraction)

            # When reached desired frame rate, store current video
            if j != 0 and (j+1)%frames == 0:
                X_single = np.array(X_single)
                X.append(X_single)
                X_single = []
                Y.append(label)

    # Set dataset and label to np.arrays
    X = np.array(X)
    Y = np.array(Y)

    # Shuffle them for training and validation purposes
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Store them into .npy files for extraction later
    with open(save_path + '/data', 'wb') as f:
        np.save(f, X)
    with open(save_path + '/label', 'wb') as f:
        np.save(f, Y)

def video_data_gen_full(feature_extractor, img_height = 720, img_width = 1080, frames = 30):
    '''
    This function generates the full dataset for videos and save them as npy files
    Args:
        feature_extractor: Feature extractor
        img_height: Dataset image height
        img_width: Dataset image width
        frames: Number of frames in each sample
    Returns:
        None
    '''
    # Storage variables
    X = []
    Y = []

    # Get minimum video size and video slice indices
    vid_min_size, genre_idx_dict = get_min_segment_val()

    # Check if given frame is larger than minimum video length
    if frames > vid_min_size:
        print("Too many frames to collect.")
        return 0

    # Construct dataset
    for i in range(len(genres)):
        # Set current path, file names, video splits and label
        cur_path = path + "/images/" + genres[i]
        dir_names = os.listdir(cur_path)
        genre_idx = genre_idx_dict[genres[i]]
        label = np.zeros(len(genres))
        label[i] = 1

        for j in tqdm(range(len(genre_idx)-1) ,position=0, desc="Current genre: "+str(genres[i])):
            # Get beginning and ending indices for a single video
            idx1 = genre_idx[j]
            idx2 = genre_idx[j+1] + 1
            sub_samples = list(range(idx1, idx2, frames-1))

            for k in range(len(sub_samples)-1):
                # Get start and end indices for a single sample
                i1 = sub_samples[k]
                i2 = sub_samples[k+1] + 1
                X_single = []

                # Loop through each image in single sample
                for m in range(i1,i2):
                    # Read image
                    img = cv2.imread(cur_path + "/" + dir_names[m])
                    img = cv2.resize(img, (img_width,img_height))

                    # Extract image features
                    extraction = feature_extractor.predict(np.expand_dims(img,axis=0))
                    extraction = extraction.flatten()
            
                    # Add to X_single
                    X_single.append(extraction)
                
                # Add data to X, Y
                X_single = np.array(X_single)
                X.append(X_single)
                Y.append(label)
                X_single = []

    # Set dataset and label to np.arrays
    X = np.array(X)
    Y = np.array(Y)

    # Shuffle them for training and validation purposes
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    # Store them into .npy files for extraction later
    save_path = path + "/generated_dataset/image"
    np.save(save_path + '/data', X)
    np.save(save_path + '/label', Y)


if __name__ == '__main__':
    # Set image height and width
    img_height = 720
    img_width = 1080

    _,_,n = get_min_segment_val()
    print(n)

    # Build feature extractor
    feature_extractor = build_feature_extractor(img_height=img_height, img_width=img_width)

    # Generate dataset
    video_data_gen_sub(feature_extractor, img_height=img_height, img_width=img_width)
    #video_data_gen_full(feature_extractor, img_height=img_height, img_width=img_width)
