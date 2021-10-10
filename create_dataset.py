import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import sys

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    except:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

import pdb
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k in cache:
            v = cache[k]
            txn.put(k.encode(),str(v).encode())


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        # print(imageKey)
        cache[imageKey] = imageBin
        cache[labelKey] = label
        # pdb.set_trace()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'Written {cnt} / {nSamples}')
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples')


if __name__ == '__main__':
    
    input_file,out_file = sys.argv[1:]
    f = open(input_file,'r')
    data = f.readlines()
    img_paths = []
    labels = []
    for line in data:

        line_splits = line.strip().split(' ', 1)  # split on first occurrence of space
        img_paths.append(line_splits[0])
        labels.append(line_splits[1])
        assert len(img_paths) == len(labels)
    createDataset(out_file,img_paths,labels)
        