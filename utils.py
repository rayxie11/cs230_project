'''
This file contains all the util functions that are needed to extract information from the given dataset
'''
import os
import numpy as np
import cv2
from tqdm.auto import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow_datasets as tfds
from tensorflow import keras

# Dataset path
path = "C:/Users/ray_s/Desktop/cs230_project/dataset"
genres = os.listdir(path + "/images")


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

    for genre in genres:
        genre_idx_dict[genre] = [0]
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
                genre_idx_dict[genre].append(i)

                # Reset count and prev_header
                cur_vid_len = 1
                prev_header = dir_names[i].split('_')[0]
        
        # Add the last index to idx_dict
        genre_idx_dict[genre].append(len(dir_names)-1)
    
    return vid_min_size, genre_idx_dict


def remap_dir():
    '''
    This function remaps the directories and files for the video dataset
    Details:
        Genre -> One video -> Video segments
    Args:
        None
    Returns:
        None
    '''
    vid_min_size, genre_idx_dict = get_min_segment_val()

    for genre in genres:
        cur_path = path + "/video/" + genre
        dir_names = os.listdir(cur_path)
        genre_idx = genre_idx_dict[genre]

        parent_folder_idx = 0

        for i in range(len(genre_idx)-1):
            idx1 = genre_idx[i]
            idx2 = genre_idx[i+1]
            dir_slice = dir_names[idx1:idx2+1]


def video_data_gen_full(img_height = 720, img_width = 1080):
    '''
    This function generates the full dataset for videos
    Args:
        img_height: Dataset image height
        img_width: Dataset image width
    Returns:
        X: Video data (num_samples,img_dim_x,img_dim_y,3)
        Y: One-hot label (num_samples,label_dim)
    '''
    # Return variables
    dataset = None

    # Get minimum video size and video slice indices
    vid_min_size, genre_idx_dict = get_min_segment_val()

    # Construct dataset
    for i in range(len(genres)):
        # Set current path, video splits and label
        cur_path = path + "/video/" + genres[i]
        genre_idx = genre_idx_dict[genres[i]]
        label = np.zeros(len(genres))
        label[i] = 1
        label = np.expand_dims(label, axis=0)

        # Extract all images in current genre
        dataset_cur_genre = tf.keras.utils.image_dataset_from_directory(
            cur_path,
            labels = None,
            image_size = (img_height, img_width),
            color_mode = "rgb",
            crop_to_aspect_ratio = True,
            shuffle = False,
            batch_size = 1
        )
        dataset_cur_genre = dataset_cur_genre.unbatch()

        # Set tf.data.Dataset manip constants
        skip_count = 0

        

        # Get video slices w.r.t to genre_idx and vid_min_size
        for j in tqdm(range(len(genre_idx)-1),position=0,leave=True):
        #for j in range(len(genre_idx)-1):
            idx_1 = genre_idx[j]
            idx_2 = genre_idx[j+1]

            # See how many samples can a single video generate
            n_samples = (idx_2 - idx_1) // vid_min_size
            remainder = (idx_2 - idx_1) % vid_min_size

            # Collect data from a single video segment
            #X_sub = []
            #Y_sub = np.tile(label, (n_samples,1))

            # Generate sub dataset
            for _ in range(n_samples):
                sub_x = dataset_cur_genre.skip(skip_count).take(vid_min_size)
                #sub_x = np.array(list(tfds.as_numpy(sub_x)))
                #sub_x = np.array([list(tfds.as_numpy(sub_x))])
                #print(sub_x.shape)
                #X_sub.append(sub_x)
                
                if dataset == None:
                    dataset = tf.data.Dataset.from_tensor_slices((sub_x,label))
                else:
                    sub_dataset = tf.data.Dataset.from_tensor_slices((sub_x,label))
                    dataset = dataset.concatenate(sub_dataset)
                

                # Update skip_count
                skip_count += vid_min_size
            
            #X_sub = np.array(X_sub)
            #assert X_sub.shape[0] == Y_sub.shape[0], "Data and label have wrong shapes"

            # Add generated data into dataset
            '''
            if dataset == None:
                dataset = tf.data.Dataset.from_tensor_slices((X_sub,Y_sub))
            else:
                sub_dataset = tf.data.Dataset.from_tensor_slices((X_sub,Y_sub))
                dataset = dataset.concatenate(sub_dataset)
            '''
            # Update skip_count
            skip_count += remainder
        break





    '''
    cur_path = path + "/video/" + genres[0]
    dataset_full = tf.keras.utils.image_dataset_from_directory(
        cur_path,
        labels = None,
        image_size = (img_height, img_width),
        color_mode = "rgb",
        crop_to_aspect_ratio = True,
        shuffle = False,
        batch_size = 1
    )
    

    #idx_0 = idx_dict[genres[0]]
    #for i in range(len(idx_0)-1):
        #new = dataset_full[idx_0[i]:idx_0[i+1]]
    #print(len(dataset_full[:100]))
    #ls = np.array(list(dataset_full.as_numpy_iterator()))
    #print(ls.shape)

    sub = dataset_full.unbatch().skip(500).take(101)
    sub = np.array(list(tfds.as_numpy(sub)))
    print(sub.shape)
    '''
    '''
    for i in sub:
        x = i.numpy().astype('uint8')
        disp = Image.fromarray(x[0],'RGB')
        disp.show()
    '''
    '''
    X_0 = []
    cur_path = path + "/video/" + genres[0]
    c = imread_collection(cur_path+"/*.jpg")
    for img in tqdm(c,position=0,leave=True):
        height, width, _ = img.shape
        if height == img_height and width == img_width:
            X_0.append(img)
        else:
            res = cv2.resize(img, dsize=(img_height,img_width))
            X_0.append(res)
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

def build_feature_extractor(img_height = 256, img_width = 256):
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
    This function generates a subsection of dataset for videos
    Args:
        img_height: Dataset image height
        img_width: Dataset image width
        frames: number of frames per video
    Returns:
        X: Video data (num_samples,img_dim_x,img_dim_y,3)
        Y: One-hot label (num_samples,label_dim)
    '''
    # Genres
    genres = os.listdir(path + "/images_sub")

    # Return variables
    X = []
    Y = []

    # Retrieve data from dataset
    for i in tqdm(range(len(genres)), position=0):
        # Set one-hot label
        label = np.zeros(len(genres))
        label[i] = 1

        # Keep count of images and single datapoint
        j = 0
        X_single = []

        # Loop through all images within one genre
        cur_path = path + "/images_sub/" + genres[i]
        dir_names = os.listdir(cur_path)
        #while j < len(dir_names):
        for j in tqdm(range(len(dir_names)),position=0):
            img = cv2.imread(cur_path + "/" + dir_names[j])
            img = cv2.resize(img, (img_width,img_height))
            #print(j)
            extraction = feature_extractor.predict(np.expand_dims(img,axis=0))
            X_single.append(extraction)
            if j != 0 and (j+1)%frames == 0:
                X_single = np.array(X_single)
                X.append(X_single)
                X_single = []
                Y.append(label)
            #j += 1

    # Set dataset and label to np.arrays
    X = np.array(X)
    Y = np.array(Y)

    # Shuffle them for training and validation purposes
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    # Store them into .npy files for extraction later
    save_path = path + "/generated_dataset/image_sub"
    np.save(save_path + '/data', X)
    np.save(save_path + '/label', Y)

    return X, Y    
        





if __name__ == '__main__':
    feature_extractor = build_feature_extractor()
    video_data_gen_sub(feature_extractor)