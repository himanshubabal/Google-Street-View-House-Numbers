from __future__ import print_function
from __future__ import division

import os
import math
import h5py
import tarfile
import idx2numpy
import scipy.misc
import numpy as np
import cPickle as pickle
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt

from downloader_extracter import maybe_download, maybe_extract
from path import data_dir

# data_dir = '/Volumes/700_GB/Study/SVHN/SVHN-Full_Dataset/'
# data_dir = '/home/himanshubabal/Google-Street-View-House-Numbers/datasets/svhn_raw/'
svhn_dataset_location = data_dir + 'svhn_raw/'

MAX_DIGITS = 10
label_width = MAX_DIGITS + 1

OUT_SHAPE = (64, 64)

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


# Plot image with it's bounding boxes
def get_plot_full_img_with_bb(image, bbox_list):
    fig,ax = plt.subplots(1)
    ax.imshow(image)

    draw_rectangles(ax, bbox_list)
    plt.show()

# Draw bounding on top of image
def draw_rectangles(ax, bbox_list):
    blank_box = [0, 0, 15, 15]

    for i in range(MAX_DIGITS):
        bbox = list()
        bbox.append(bbox_list[4*i+0])
        bbox.append(bbox_list[4*i+1])
        bbox.append(bbox_list[4*i+2])
        bbox.append(bbox_list[4*i+3])

        # Do not draw bounding box for blank image
        if bbox < blank_box:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            rect = patches.Rectangle((x1, y1),x2-x1,y2-y1,linewidth=1,edgecolor='y',facecolor='none')
            ax.add_patch(rect)


# Normalise the image. Helps in image recognition
def normalize_image(img_resize):
    # Mean and Std Deviation of Image
    mean = np.mean(img_resize, dtype='float32')
    std = np.std(img_resize, dtype='float32', ddof=1)

    if std < 1e-4:
        std = 1.0
    # Normalizing the image
    img_resize = (img_resize - mean)/std

    return(img_resize)


# Resize, greyscale, normalise image
def process_image(img):
    # Resizing Image to be of 64x64x3 dimensions
    img_resize = scipy.misc.imresize(img, OUT_SHAPE)
    # Converting A x B x 3 -> A x B x 1
    img_resize = np.dot(img_resize, [[0.2989],[0.5870],[0.1140]])
    # Normalise
    img_resize = normalize_image(img_resize)
    # 64 x 64 x 1 -> 64 x 64
    img_resize = img_resize.reshape(img_resize.shape[:2])

    return img_resize

# Get label as np array
def get_label(img_data):
    num_images = len(img_data['boxes'])
    new_label = np.empty((MAX_DIGITS+1), dtype='int')
    new_label[0] = num_images
    for i in range(num_images+1, MAX_DIGITS+1):
        new_label[i] = 10

    for i in range(num_images):
        new_label[i+1] = img_data['boxes'][i]['label']

    return new_label

# Get bounding box as np array for original image
def get_bbox(img_data):
    num_images = len(img_data['boxes'])
    new_box = np.empty((MAX_DIGITS*4), dtype='int')

    for i in range(num_images):
        new_box[4*i  ] = img_data['boxes'][i]['left']
        new_box[4*i+1] = img_data['boxes'][i]['top']
        new_box[4*i+2] = img_data['boxes'][i]['width'] + new_box[4*i]
        new_box[4*i+3] = img_data['boxes'][i]['height'] + new_box[4*i+1]

    # Assining empty spaces bbox (0,0), (5,5)
    for i in range(num_images, MAX_DIGITS):
        new_box[4*i  ] = 0
        new_box[4*i+1] = 0
        new_box[4*i+2] = 5
        new_box[4*i+3] = 5

    return new_box

# Get bounding boxes for reshaped image as np array
def get_new_bbox(bbox, old_shape, new_shape):
    Ry = (new_shape[0] * 1.00)/old_shape[0]
    Rx = (new_shape[1] * 1.00)/old_shape[1]

    new_box = np.empty((MAX_DIGITS*4), dtype='int')
    for i in range(MAX_DIGITS):
        new_box[4*i  ] = int(Rx * bbox[4*i  ])
        new_box[4*i+1] = int(Ry * bbox[4*i+1])
        new_box[4*i+2] = int(Rx * bbox[4*i+2])
        new_box[4*i+3] = int(Ry * bbox[4*i+3])

    return(new_box)

# Primary pipeline
def pipeline(index, folder_name, data):
    image_data = data[index]
    image_name = svhn_dataset_location + folder_name + '/' + image_data['filename']

    # Open and load image as np array
    img = Image.open(image_name)
    img = np.asarray(img, dtype="float32")
    original_shape = img.shape[:2]

    # Process image, get label and bounding box
    img_resize = process_image(img)
    bounding_box = get_new_bbox(get_bbox(image_data), original_shape, img_resize.shape)
    label = get_label(image_data)

    return(img_resize, bounding_box, label)

# Simple enough to understand
def pipeline_primary(data, folder_name):
    images = np.ndarray([len(data), OUT_SHAPE[0], OUT_SHAPE[1]], dtype='float32')
    labels = np.ndarray([len(data), MAX_DIGITS+1], dtype='int32')
    bboxes = np.ndarray([len(data), MAX_DIGITS*4], dtype='int32')

    for i in range(len(data)) :
        img_resize, bounding_box, label = pipeline(i, folder_name, data)
        images[i] = img_resize
        labels[i] = label
        bboxes[i] = bounding_box

        if i % (len(data)//10) == 0:
            print((i/len(data))*100)

    return(images, labels, bboxes)


# saves images, labels, bboxes
def np_save(save_location, name, images, labels, bboxes):
    np.save(save_location + name + '_images', images)
    np.save(save_location + name + '_labels', labels)
    np.save(save_location + name + '_bboxes', bboxes)

# load images, labels, bboxes
def np_load(save_location, name):
    loc = str(save_location + name)
    images = np.load(loc + '_images.npy')
    labels = np.load(loc + '_labels.npy')
    bboxes = np.load(loc + '_bboxes.npy')

    return(images, labels, bboxes)

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    assert len(b) == len(c)
    p = np.random.permutation(len(a))
    return (a[p], b[p], c[p])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #  P R O C E S S I N G  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Download and Extract Data
# See 'downloader_extractor.py' for details
svhn_url = 'http://ufldl.stanford.edu/housenumbers/'
file_list = ['test.tar.gz', 'train.tar.gz', 'extra.tar.gz']

for svhn_file in file_list:
    svhn_zip = maybe_download(svhn_dataset_location, svhn_file, svhn_url + svhn_file)
    svhn_folder = maybe_extract(svhn_zip)

print('All files downloaded and extracted')


hdf_file = svhn_dataset_location + 'SVHN.hdf5'
if not os.path.exists(hdf_file):
    # Proceed further
    struct_file = svhn_dataset_location + 'SVHN_data_struct.pkl'

    if os.path.exists(struct_file) :
        print('Loading saved SVHN_data_struct.pkl')
        with open(struct_file, 'rb') as f:
            save = pickle.load(f)
            test_data = save['test_data']
            train_data = save['train_data']
            extra_data = save['extra_data']
        del save


    else :
        print('Creating SVHN_data_struct.pkl')
        print('SVHN_data_struct.pkl - test data')
        fin = os.path.join(svhn_dataset_location + 'test', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        test_data = dsf.getAllDigitStructure_ByDigit()

        print('SVHN_data_struct.pkl - train data')
        fin = os.path.join(svhn_dataset_location + 'train', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        train_data = dsf.getAllDigitStructure_ByDigit()

        print('SVHN_data_struct.pkl - extra data')
        fin = os.path.join(svhn_dataset_location + 'extra', 'digitStruct.mat')
        dsf = DigitStructFile(fin)
        extra_data = dsf.getAllDigitStructure_ByDigit()
        # This data point is corrupted
        del extra_data[118847]

        print('SVHN_data_struct.pkl')
        with open(struct_file, 'wb') as f:
            data = {}
            data['test_data'] = test_data
            data['train_data'] = train_data
            data['extra_data'] = extra_data
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)



    # Checking if already done till here
    npy_data = ['test_images.npy', 'test_labels.npy', 'test_bboxes.npy',
                'extra_labels.npy', 'extra_bboxes.npy', 'extra_images.npy',
                'train_labels.npy', 'train_images.npy', 'train_bboxes.npy']


    # Check whether process till here has already taken place
    # If yes, then load saved files
    if set(npy_data).issubset(os.listdir(svhn_dataset_location)):
        print('Loading saved .npy files')
        test_images,  test_labels,  test_bboxes  = np_load(svhn_dataset_location, 'test')
        train_images, train_labels, train_bboxes = np_load(svhn_dataset_location, 'train')
        extra_images, extra_labels, extra_bboxes = np_load(svhn_dataset_location, 'extra')

    else:
        print('Primary Pipeline - test data')
        test_images,  test_labels,  test_bboxes  = pipeline_primary(test_data,  'test' )

        print('Primary Pipeline - train data')
        train_images, train_labels, train_bboxes = pipeline_primary(train_data, 'train')

        print('Primary Pipeline - extra data')
        extra_images, extra_labels, extra_bboxes = pipeline_primary(extra_data, 'extra')

        np_save(svhn_dataset_location, 'test', test_images,  test_labels,  test_bboxes)
        np_save(svhn_dataset_location, 'train', train_images, train_labels, train_bboxes)
        np_save(svhn_dataset_location, 'extra', extra_images, extra_labels, extra_bboxes)

    print(train_images.shape, train_labels.shape, train_bboxes.shape)
    print(test_images.shape , test_labels.shape , test_bboxes.shape )
    print(extra_images.shape, extra_labels.shape, extra_bboxes.shape)

    print('Combining train and extra data')
    combined_images = np.vstack((train_images, extra_images))
    combined_labels = np.vstack((train_labels, extra_labels))
    combined_bboxes = np.vstack((train_bboxes, extra_bboxes))

    print(combined_images.shape, combined_labels.shape, combined_bboxes.shape)

    # Shuffling
    print('Shuffling combined data')
    combined_images, combined_labels, combined_bboxes = unison_shuffled_copies(combined_images, combined_labels, combined_bboxes)

    print('Splitting combined data in train and validation sets')
    valid_inputs = 5754
    train_images = combined_images[:len(combined_labels)-valid_inputs]
    train_labels = combined_labels[:len(combined_labels)-valid_inputs]
    train_bboxes = combined_bboxes[:len(combined_labels)-valid_inputs]

    valid_images = combined_images[len(combined_labels)-valid_inputs:]
    valid_labels = combined_labels[len(combined_labels)-valid_inputs:]
    valid_bboxes = combined_bboxes[len(combined_labels)-valid_inputs:]

    print(train_images.shape, train_labels.shape, train_bboxes.shape)
    print(test_images.shape , test_labels.shape , test_bboxes.shape )
    print(valid_images.shape, valid_labels.shape, valid_bboxes.shape)

    del combined_images, combined_labels, combined_bboxes


    print('Saving all processed data in SVHN.hdf5')
    hdf = h5py.File(hdf_file, 'w')

    with hdf as hf:
        hf.create_dataset("train_images",  data=train_images)
        hf.create_dataset("train_labels",  data=train_labels)
        hf.create_dataset("train_bboxes",  data=train_bboxes)

        hf.create_dataset("valid_images",  data=valid_images)
        hf.create_dataset("valid_labels",  data=valid_labels)
        hf.create_dataset("valid_bboxes",  data=valid_bboxes)

        hf.create_dataset("test_bboxes",  data=test_bboxes)
        hf.create_dataset("test_images",  data=test_images)
        hf.create_dataset("test_labels",  data=test_labels)

        hf.close()

    print('SVHN Datasets saved in SVHN.hdf5')

    # Creating smaller datasets so ram usage is lower
    trial_images = train_images[:100]
    trial_labels = train_labels[:100]
    trial_bboxes = train_bboxes[:100]

    print('Saving all processed data in SVHN.hdf5')
    hdf = h5py.File(svhn_dataset_location + 'SVHN_trial.hdf5', 'w')

    with hdf as hf:
        hf.create_dataset("trial_images",  data=trial_images)
        hf.create_dataset("trial_labels",  data=trial_labels)
        hf.create_dataset("trial_bboxes",  data=trial_bboxes)
        hf.close()
else :
    print('SVHN Data already processed')

