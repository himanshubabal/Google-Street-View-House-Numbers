#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
import h5py
from PIL import Image

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

data_location = '/Users/himanshubabal/Documents/External_Disk_Link_WD_HDD/Study/SVHN/SVHN-Full_Dataset/'


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
....slow internet connections. Reports every 1% change in download progress.
...."""

    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('%s%%' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, force=False):
    """Download a file if not present, and make sure it's the right size."""

    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        (filename, _) = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
        statinfo = os.stat(filename)
    return filename


train_filename = maybe_download(data_location + 'train.tar.gz')
test_filename = maybe_download(data_location + 'test.tar.gz')
extra_filename = maybe_download(data_location + 'extra.tar.gz')

np.random.seed(133)


def maybe_extract(filename, force=False):
    root = filename[:-7]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = root
    print(data_folders)
    return data_folders

print(train_filename)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
extra_folders = maybe_extract(extra_filename)


# The DigitStructFile is just a wrapper around the h5py data.  It basically references
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data

class DigitStructFile:

    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct.

    def getName(self, n):
        return ''.join([chr(c[0]) for c in
                       self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox.

    def bboxHelper(self, attr):
        if len(attr) > 1:
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in
                    range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox.

    def getBbox(self, n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]['height'])
        bbox['label'] = self.bboxHelper(self.inf[bb]['label'])
        bbox['left'] = self.bboxHelper(self.inf[bb]['left'])
        bbox['top'] = self.bboxHelper(self.inf[bb]['top'])
        bbox['width'] = self.bboxHelper(self.inf[bb]['width'])
        return bbox

    def getDigitStructure(self, n):
        s = self.getBbox(n)
        s['name'] = self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.

    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in
                range(len(self.digitStructName))]

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
            item = {'filename': pictDat[i]['name']}
            figures = []
            for j in range(len(pictDat[i]['height'])):
                figure = {}
                figure['height'] = pictDat[i]['height'][j]
                figure['label'] = pictDat[i]['label'][j]
                figure['left'] = pictDat[i]['left'][j]
                figure['top'] = pictDat[i]['top'][j]
                figure['width'] = pictDat[i]['width'][j]
                figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result


train_folders = data_location + 'train'
test_folders = data_location + 'test'
extra_folders = data_location + 'extra'

struct_pickle_file = data_location + 'SVHN_new_data_struct.pickle'

if not os.path.exists(struct_pickle_file):
    print('$$')
    fin = os.path.join(train_folders, 'digitStruct.mat')
    dsf = DigitStructFile(fin)
    train_data = dsf.getAllDigitStructure_ByDigit()

    fin = os.path.join(test_folders, 'digitStruct.mat')
    dsf = DigitStructFile(fin)
    test_data = dsf.getAllDigitStructure_ByDigit()

    fin = os.path.join(extra_folders, 'digitStruct.mat')
    dsf = DigitStructFile(fin)
    extra_data = dsf.getAllDigitStructure_ByDigit()

    print('@@@@@@@@@@@@')
    test_struct_hdf5 = h5py.File(data_location + "SVHN_test_struct.hdf5", "w")
    with test_struct_hdf5 as hf:
        hf.create_dataset("test_data",  data=test_data)

    print('@@@@@@@@@@@@')
    train_struct_hdf5 = h5py.File(data_location + "SVHN_train_struct.hdf5", "w")
    with train_struct_hdf5 as hf:
        hf.create_dataset("train_data",  data=train_data)

    print('@@@@@@@@@@@@')
    extra_struct_hdf5 = h5py.File(data_location + "SVHN_extra_struct.hdf5", "w")
    with extra_struct_hdf5 as hf:
        hf.create_dataset("extra_data",  data=extra_data)

else:
    with open(struct_pickle_file, 'rb') as f:
        save = pickle.load(f, encoding='latin1')
        test_data = save['test_data']
        extra_data = save['extra_data']
        train_data = save['train_data']
        del save

train_imsize = np.ndarray([len(train_data), 2])
for i in np.arange(len(train_data)):
    filename = train_data[i]['filename']
    fullname = os.path.join(train_folders, filename)
    im = Image.open(fullname)
    train_imsize[i, :] = im.size[:]

test_imsize = np.ndarray([len(test_data), 2])
for i in np.arange(len(test_data)):
    filename = test_data[i]['filename']
    fullname = os.path.join(test_folders, filename)
    im = Image.open(fullname)
    test_imsize[i, :] = im.size[:]

extra_imsize = np.ndarray([len(extra_data), 2])
for i in np.arange(len(extra_data)):
    filename = extra_data[i]['filename']
    fullname = os.path.join(extra_folders, filename)
    im = Image.open(fullname)
    extra_imsize[i, :] = im.size[:]


def generate_dataset(data, folder):

    dataset = np.ndarray([len(data), 32, 96, 1], dtype='float32')
    labels = np.ones([len(data), 6], dtype=int) * 10
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        labels[i, 0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5:
                labels[i, j + 1] = boxes[j]['label']
                if boxes[j]['label'] == 10:
                    labels[i, j + 1] = 0

            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']

        im_top = np.amin(top)
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left

        im_top = np.floor(im_top - 0.1 * im_height)
        im_left = np.floor(im_left - 0.1 * im_width)
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])

        im = im.crop((im_left, im_top, im_right, im_bottom)).resize([96, 32], Image.ANTIALIAS)
        im = np.dot(np.array(im, dtype='float32'), [[0.2989], [0.5870], [0.1140]])
        mean = np.mean(im, dtype='float32')
        std = np.std(im, dtype='float32', ddof=1)
        if std < 1e-4:
            std = 1.
        im = (im - mean) / std
        dataset[i, :, :, :] = im[:, :, :]

    return (dataset, labels)


(train_dataset, train_labels) = generate_dataset(train_data, train_folders)
(test_dataset, test_labels) = generate_dataset(test_data, test_folders)
(extra_dataset, extra_labels) = generate_dataset(extra_data, extra_folders)

train_dataset = np.delete(train_dataset, 29929, axis=0)
train_labels = np.delete(train_labels, 29929, axis=0)

random.seed()

n_labels = 10
valid_index = []
valid_index2 = []
train_index = []
train_index2 = []
for i in np.arange(n_labels):
    valid_index.extend((np.where(train_labels[:, 1] == i)[0])[:400].tolist())
    train_index.extend((np.where(train_labels[:, 1] == i)[0])[400:].tolist())
    valid_index2.extend((np.where(extra_labels[:, 1] == i)[0])[:200].tolist())
    train_index2.extend((np.where(extra_labels[:, 1] == i)[0])[200:].tolist())

random.shuffle(valid_index)
random.shuffle(train_index)
random.shuffle(valid_index2)
random.shuffle(train_index2)

valid_dataset = np.concatenate((extra_dataset[valid_index2, :, :, :], train_dataset[valid_index, :, :, :]), axis=0)
valid_labels = np.concatenate((extra_labels[valid_index2, :], train_labels[valid_index, :]), axis=0)
train_dataset_t = np.concatenate((extra_dataset[train_index2, :, :, :], train_dataset[train_index, :, :, :]), axis=0)
train_labels_t = np.concatenate((extra_labels[train_index2, :], train_labels[train_index, :]), axis=0)

print('train dataset shapes : ', train_dataset_t.shape, train_labels_t.shape)
print('test dataset shapes : ', test_dataset.shape, test_labels.shape)
print('validation dataset shapes : ', valid_dataset.shape, valid_labels.shape)

print('-----------')
test_data_hdf5 = h5py.File(data_location + "SVHN_multi_box_test_data.hdf5", "w")
with test_data_hdf5 as hf:
    hf.create_dataset("test_dataset",  data=test_dataset)
print('-----------')
test_label_hdf5 = h5py.File(data_location + "SVHN_multi_box_test_labels.hdf5", "w")
with test_label_hdf5 as hf:
    hf.create_dataset("test_labels",  data=test_labels)
print('-----------')
valid_data_hdf5 = h5py.File(data_location + "SVHN_multi_box_valid_data.hdf5", "w")
with valid_data_hdf5 as hf:
    hf.create_dataset("valid_dataset",  data=valid_dataset)
print('-----------')
valid_labels_hdf5 = h5py.File(data_location + "SVHN_multi_box_valid_labels.hdf5", "w")
with valid_labels_hdf5 as hf:
    hf.create_dataset("valid_labels",  data=valid_labels)
print('-----------')
train_data_hdf5 = h5py.File(data_location + "SVHN_multi_box_train_data.hdf5", "w")
with train_data_hdf5 as hf:
    hf.create_dataset("train_dataset",  data=train_dataset_t)
print('-----------')
train_labels_hdf5 = h5py.File(data_location + "SVHN_multi_box_train_labels.hdf5", "w")
with train_labels_hdf5 as hf:
    hf.create_dataset("train_labels_t",  data=train_labels_t)
print('-----------')





pickle_file = 'SVHN_multi_box_test.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
print('***********')
pickle_file = 'SVHN_multi_box_valid.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
print('***********')
pickle_file = 'SVHN_multi_box_train.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_t,
        'train_labels': train_labels_t,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
print('***********')
# pickle_file = data_location + 'SVHN_multi_box.pickle'
pickle_file = 'SVHN_multi_box.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset_t,
        'train_labels': train_labels_t,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
print('SVHN Datasets ready in SVHN_multi.pickle')


			