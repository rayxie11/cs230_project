# CS230 Project: AI Choreographer

## Introduction
This repository contains all the code CS230 Deep Learning Final Project which uses a recurrent neural network (bi-directional LSTM) to classify dance genres among 16 different styles.

## File Explanantion

`utils.py`: contains utility function for dataset generation

`frameToVideo.py`: converts series of frames (images) into videos

`subset.py`: generates a subset of training and test data (each video 20 frames each)

`hyperparameter_tuning.py`: contains the neural network structure, plots the accuracy and loss, and tunes hyperparameters of the model

`process_data.py`: dataset generation for prediction model

`prediction.py`: prediction model

`run_frame_prediction`: test prediction model

`base_line.py` (not used anymore): contains neural network for classifying dance genres

## Installation
Clone this repository and install dependecies if you are missing any.

## Dataset
All the datafiles should be constructed in the same directory as this repository. The Georgia Tech dataset can be accessed through here: https://www.cc.gatech.edu/cpl/projects/dance/. Only "Let' Dance: Original Frames" are used in this project.

## Test
1. Dataset generation: In `utils.py`, set the image data directory, the number of frames you want to generate for each example, size of the image and whether to generate a full or small dataset. Run this file.

2. Model construction and tuning: In `hyperparameter_tuning.py`, set extracted dataset directory and the values of hyperparameters. Run the file.
