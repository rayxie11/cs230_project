import builtins
import os
import sys
import tarfile

from collections import deque


MAX_FRAME = 20
#deprecate
MAX_VIDEO_COUNT = 2
filePath = '/Users/xfeng/cs230/rgb.tar.gz'
dirPath = '/Users/xfeng/cs230/rgb'

def files_from_tar_url(tar_path):
    labelMap = {}
    with tarfile.open(tar_path) as archive:
        for file in archive:
            if not file.isdir() and file.name.count('/') >=2:
                print(file.name)
                #rgb/tap/2TU7dE_9Vi4_341_0195.jpg
                label = file.name[file.name.find('/') + 1 : file.name.rfind('/')]
                print(label)
                if label not in labelMap:
                   labelMap[label] = {}

                d = labelMap[label]

                videoName = file.name[:file.name.rfind('_')]
                index = int(file.name[file.name.rfind('_') + 1 : file.name.rfind('.')])
                print(videoName, index)
                if videoName not in d:
                    d[videoName] = []
                if index <= MAX_FRAME:
                    d[videoName].append(file.name[file.name.rfind('/') + 1 :]) 

    retentionFiles = []

    for labelVideos in labelMap.values():
        for (x, files) in labelVideos.items():
            files = sorted(files)
            labelVideos[x] = files
            for file in files:
                retentionFiles.append(file)
    return labelMap, retentionFiles

labels,retentionFiles = files_from_tar_url(filePath)
#print(labels)
print(len(retentionFiles))
print(retentionFiles)

for root, dirs, files in os.walk(dirPath):
    for name in files:
        if name not in retentionFiles:
            print("remove " + name)
            os.remove(os.path.join(root, name))

#files = [f for f in files if f.endswith('.jpg')]
#files[:10]
#print(files[:10])
