import builtins
import os
import sys
import tarfile

from collections import deque

END_FRAME_INDEX = 20
START_FRAME_INDEX = 1 

filePath = '/Users/xfeng/cs230/rgb.tar.gz'
outPath = '/Users/xfeng/cs230/rgb_new'

def files_from_tar_url(tar_path, target_dir, start_frame_index, end_frame_index):
    labelMap = {}
    with tarfile.open(tar_path) as archive:
        for file in archive:
            if not file.isdir() and file.name.count('/') >=2:
                videoName = file.name[:file.name.rfind('_')]
                index = int(file.name[file.name.rfind('_') + 1 : file.name.rfind('.')])
                if index >= start_frame_index and index <= end_frame_index:
                    print("extracting " + file.name)
                    archive.extract(file, target_dir) 


files_from_tar_url(filePath, outPath, START_FRAME_INDEX, END_FRAME_INDEX)

#files = [f for f in files if f.endswith('.jpg')]
#files[:10]
#print(files[:10])
