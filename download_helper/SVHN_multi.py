from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from PIL import Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import scipy.ndimage
import scipy.misc
#get_ipython().magic('matplotlib inline')
import h5py
import gc
import time

dataset_location = 'datasets/svhn_raw/'

class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result
    
def plot_img(image):
    plt.imshow(image)
    plt.show()
    
    
def prep_svhn_multi():    
    pickle_file = 'datasets/pickles/SVHN_new_data_struct.pickle'

    if os.path.exists(pickle_file) :
        with open(pickle_file, 'rb') as f:
            save = pickle.load(f)
            test_data = save['test_data']
            train_data = save['train_data']
            extra_data = save['extra_data']
        del save
        #print(len(train_data))
        #print(len(test_data))
        #print(len(extra_data))

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

    test_images = np.ndarray([len(test_data), 32, 96, 1], dtype='int32')
    test_labels = np.ndarray([len(test_data), 6], dtype='int32')
    folder_name = 'test'
    l = len(test_data)

    for i in range(l) :
        image = test_data[i]
        image_name = dataset_location + folder_name + '/' + image['filename']

        img = Image.open(image_name)
        img.load()
        img = np.asarray(img, dtype="int32")

        # Resizing Image to be of 32x96x3 dimensions
        img = scipy.misc.imresize(img, (32, 96))
        # Converting A x B x 3 -> A x B x 1
        img = np.dot(img, [[0.2989],[0.5870],[0.1140]])

        # Mean and Std Deviation of Image
        mean = np.mean(img, dtype='float32')
        std = np.std(img, dtype='float32', ddof=1)
        if std < 1e-4:
            std = 1.0
        # Normalizing the image
        im = (img - mean)/std

        # Finding No of Digits in given image
        no_of_digits = len(image['boxes'])
        # If no of digits is > 5
        # Leave this example out, as we don't need it
        # Continue to next Iteration
        if no_of_digits > 5:
            #print('more then 5 digits', image['boxes'])
            continue

        # Numpy Array 'dig' will contain digits in the format :
        # [No_of_digits, _, _, _, _, _]
        dig = np.array([])
        dig = np.append(dig, no_of_digits)

        for j in range(no_of_digits) :
            digit = image['boxes'][j]['label']
            dig = np.append(dig, digit)

        # Appending '0' to represent Empty Space
        zeros = 5 - no_of_digits
        for z in range(zeros) :
            dig = np.append(dig, 0)

        test_images[i,:,:,:] = im[:,:,:]
        test_labels[i,:] = dig

        #if i % 1500 == 0:
            #print((i/l)*100)


    l = len(train_data)
    more_then_5_counter = 0
    train_images = np.ndarray([l, 32, 96, 1], dtype='int32')
    train_labels = np.ndarray([l, 6], dtype='int32')
    folder_name = 'train'


    for i in range(l) :
        image = train_data[i]
        image_name = dataset_location + folder_name + '/' + image['filename']

        img = Image.open(image_name)
        img.load()
        img = np.asarray(img, dtype="int32")

        img = scipy.misc.imresize(img, (32, 96))
        img = np.dot(img, [[0.2989],[0.5870],[0.1140]])

        mean = np.mean(img, dtype='float32')
        std = np.std(img, dtype='float32', ddof=1)
        if std < 1e-4:
            std = 1.0
        im = (img - mean)/std

        no_of_digits = len(image['boxes'])
        if no_of_digits > 5:
            more_then_5_counter += 1
            continue

        dig = np.array([])
        dig = np.append(dig, no_of_digits)

        for j in range(no_of_digits) :
            digit = image['boxes'][j]['label']
            dig = np.append(dig, digit)

        zeros = 5 - no_of_digits
        for z in range(zeros) :
            dig = np.append(dig, 0)

        train_images[i,:,:,:] = im[:,:,:]
        train_labels[i,:] = dig


        #if i % 5000 == 0:
            #print('progress : ', (i/l)*100, '%')

    #print('Cases Containing More then 5 Digits : ', more_then_5_counter)


    l = len(extra_data)
    more_then_5_counter = 0
    extra_images = np.ndarray([l, 32, 96, 1], dtype='int32')
    extra_labels = np.ndarray([l, 6], dtype='int32')
    folder_name = 'extra'

    for i in range(l) :
        image = extra_data[i]
        image_name = dataset_location + folder_name + '/' + image['filename']

        img = Image.open(image_name)
        img.load()
        img = np.asarray(img, dtype="int32")

        img = scipy.misc.imresize(img, (32, 96))
        img = np.dot(img, [[0.2989],[0.5870],[0.1140]])

        mean = np.mean(img, dtype='float32')
        std = np.std(img, dtype='float32', ddof=1)
        if std < 1e-4:
            std = 1.0
        im = (img - mean)/std

        no_of_digits = len(image['boxes'])
        if no_of_digits > 5:
            more_then_5_counter += 1
            continue

        dig = np.array([])
        dig = np.append(dig, no_of_digits)

        for j in range(no_of_digits) :
            digit = image['boxes'][j]['label']
            dig = np.append(dig, digit)

        zeros = 5 - no_of_digits
        for z in range(zeros) :
            dig = np.append(dig, 0)


        extra_images[i,:,:,:] = im[:,:,:]
        extra_labels[i,:] = dig

        #if i % 25000 == 0:
            #print('progress : ', (i/l)*100, '%')

    #print('Cases Containing More then 5 Digits : ', more_then_5_counter)

    del test_data, train_data, extra_data

    comb_images = np.concatenate((train_images,extra_images),axis=0)
    comb_labels = np.concatenate((train_labels,extra_labels),axis=0)
    #print(comb_images.shape, comb_labels.shape)

    del train_images
    del train_labels 
    del extra_images
    del extra_labels

    length_all = comb_labels.shape[0]
    shuffle_all = np.arange(length_all)
    np.random.shuffle(shuffle_all)

    valid_set = shuffle_all[0:6666]
    train_set = shuffle_all[6666:]

    valid_images = comb_images[valid_set,:,:,:]
    valid_labels = comb_labels[valid_set,:]

    train_images = comb_images[train_set,:,:,:]
    train_labels = comb_labels[train_set,:]

    #print("Train : ", train_images.shape, train_labels.shape)
    #print("Test : ", test_images.shape, test_labels.shape)
    #print("Validation : ", valid_images.shape, valid_labels.shape)

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