# import OS module
import os
import cv2
from pathlib import Path
import numpy as np
import glob

FRAMES_PER_SEC = 30
# this program assumes that all frames end with XXXX.jpg 

# Get the list of all files and directories
path = "/Users/theodorekanell/Downloads/School/Fall 2022/cs230/rgb/"
outPath = path + "/../rgbVids/" 

#root path should be rgb. 
#out path should be where you want new images. 
def traverseDirs(rootPath, outPath):
    count = 0
    dir_list = os.listdir(rootPath)
    for folder in dir_list:
        if (not os.path.exists(outPath + folder)):
            os.mkdir(outPath + folder)
        if (not folder.startswith(".")):
            addImgs(rootPath + folder + "/", outPath + folder + "/")


def addImgs(path, outputPath):
    dir_list = os.listdir(path)
    dir_list.sort() # sorts in ascending order.
    curr = dir_list[0][:-8]
    img_array = []
    count = 0
    for filename in dir_list:
        if (not filename.startswith(curr)): # reset for next video. 
            print("new image, " + filename[:-8])
            outputImage(outputPath + curr, img_array, size)
            img_array = []
            curr = filename[:-8]
            count += 1
            if count > 0:
                return
        #print(path, filename)
        img = cv2.imread(path + filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

def outputImage(outPath, img_array, size):
    out = cv2.VideoWriter(outPath + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), FRAMES_PER_SEC, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

traverseDirs(path, outPath)