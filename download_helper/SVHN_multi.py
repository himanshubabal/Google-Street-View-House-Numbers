from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from PIL import Image
import PIL
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import scipy.ndimage
import scipy.misc
import h5py
import gc
import time

dataset_location = 'datasets/svhn_raw/'

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['label']  = pictDat[i]['label'][j]
                figures.append(figure)
            
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
    
def process_image(data, folder):
    images = np.ndarray([len(data), 32, 96, 1], dtype='int32')
    labels = np.ndarray([len(data), 6], dtype='int32')
    folder_name = folder
    l = len(data)

    for i in range(l) :
        image = data[i]
        image_name = dataset_location + folder_name + '/' + image['filename']

        img = Image.open(image_name)
        img = img.resize((96,32), PIL.Image.ANTIALIAS)
        img = np.asarray(img, dtype="int32")
        img = np.dot(img, [[0.2989],[0.5870],[0.1140]])

        no_of_digits = len(image['boxes'])
        if no_of_digits > 5:
            continue

        dig = np.array([])
        dig = np.append(dig, no_of_digits)

        for j in range(no_of_digits) :
            digit = image['boxes'][j]['label']
            dig = np.append(dig, digit)

        zeros = 5 - no_of_digits
        for z in range(zeros) :
            dig = np.append(dig, 0)

        images[i] = img
        labels[i] = dig
        
    return(images, labels)

def prep_svhn_multi():    
    pickle_file = 'datasets/pickles/SVHN_new_data_struct.pickle'

    if os.path.exists(pickle_file) :
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            test_data = save['test_data']
            train_data = save['train_data']
            extra_data = save['extra_data']
        del save

    else :
        fin = os.path.join(dataset_location + 'test', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        test_data = dsf.getAllDigitStructure_ByDigit()

        fin = os.path.join(dataset_location + 'train', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        train_data = dsf.getAllDigitStructure_ByDigit()

        fin = os.path.join(dataset_location + 'extra', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        extra_data = dsf.getAllDigitStructure_ByDigit()

    test_images, test_labels = process_image(test_data, 'test')
    train_images, train_labels = process_image(train_data, 'train')
    extra_images, extra_labels = process_image(extra_data, 'extra')

    del test_data, train_data, extra_data

    comb_images = np.concatenate((train_images,extra_images),axis=0)
    comb_labels = np.concatenate((train_labels,extra_labels),axis=0)

    del train_images, train_labels, extra_images, extra_labels 

    length_all = comb_labels.shape[0]
    shuffle_all = np.arange(length_all)
    np.random.shuffle(shuffle_all)

    valid_set = shuffle_all[0:6666]
    train_set = shuffle_all[6666:]

    valid_images = comb_images[valid_set,:,:,:]
    valid_labels = comb_labels[valid_set,:]

    train_images = comb_images[train_set,:,:,:]
    train_labels = comb_labels[train_set,:]

    hdf_file = 'datasets/pickles/SVHN_multi.hdf5'

    hdf = h5py.File(hdf_file, 'w')

    with hdf as hf:
        hf.create_dataset("train_images",  data=train_images)
        hf.create_dataset("train_labels",  data=train_labels)
        hf.create_dataset("valid_images",  data=valid_images)
        hf.create_dataset("valid_labels",  data=valid_labels)
        hf.create_dataset("test_images",  data=test_images)
        hf.create_dataset("test_labels",  data=test_labels)

    print('SVHN Datasets ready in SVHN_multi.hdf5')


if __name__ == "__main__":
    prep_svhn_multi()