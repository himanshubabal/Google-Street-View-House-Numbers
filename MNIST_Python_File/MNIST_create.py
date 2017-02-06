from __future__ import print_function
import os
import sys
import gzip
import h5py
import pickle
import random
import scipy.io
import idx2numpy
import scipy.misc
import numpy as np
from six.moves.urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder
from six.moves import cPickle as pickle
from six.moves import range

from MNIST_Python_File.MNIST_downloader import maybe_download
from MNIST_Python_File.MNIST_downloader import maybe_extract

from MNIST_Python_File.MNIST_sequence import randomize_inputs
from MNIST_Python_File.MNIST_sequence import multiple_img_dataset_generator

url = 'http://yann.lecun.com/exdb/mnist/'
mnist_dataset_location = "datasets/MNIST/"
multi_image_width = 32*3
multi_image_height = 32

def download_and_create_data() :
    train_images_zip = maybe_download(mnist_dataset_location, 'train-images-idx3-ubyte.gz')
    train_labels_zip = maybe_download(mnist_dataset_location, 'train-labels-idx1-ubyte.gz')

    test_images_zip = maybe_download(mnist_dataset_location, 't10k-images-idx3-ubyte.gz')
    test_labels_zip = maybe_download(mnist_dataset_location, 't10k-labels-idx1-ubyte.gz')

    print('MNIST Dataset Download Complete')

    train_images_file = maybe_extract(mnist_dataset_location + 'train-images-idx3-ubyte.gz')
    train_labels_file = maybe_extract(mnist_dataset_location + 'train-labels-idx1-ubyte.gz')

    test_images_file = maybe_extract(mnist_dataset_location + 't10k-images-idx3-ubyte.gz')
    test_labels_file = maybe_extract(mnist_dataset_location + 't10k-labels-idx1-ubyte.gz')

    print('MNIST Dataset Extraction Complete')

    train_images = idx2numpy.convert_from_file(mnist_dataset_location + 'train-images-idx3-ubyte')
    train_label = idx2numpy.convert_from_file(mnist_dataset_location + 'train-labels-idx1-ubyte')

    test_images = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-images-idx3-ubyte')
    test_label = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-labels-idx1-ubyte')


    X_test_new, Y_test_new = randomize_inputs(test_images, test_label, 2500)
    X_test_multi, Y_test_multi = multiple_img_dataset_generator(X_test_new, Y_test_new, 5, 10000)

    X_train_new, Y_train_new = randomize_inputs(train_images, train_label, 20000)
    X_train_multi, Y_train_multi = multiple_img_dataset_generator(X_train_new, Y_train_new, 5, 80000)

    X_train_multi = X_train_multi[:,:,:,np.newaxis]
    X_test_multi = X_test_multi[:,:,:,np.newaxis]

    print('Final train and test datasets sizes')
    print (X_train_multi.shape, Y_train_multi.shape)
    print (X_test_multi.shape, Y_test_multi.shape)

    
    hdf_file = 'datasets/pickles/MNIST_multi.hdf5'

    hdf = h5py.File(hdf_file, 'w')

    with hdf as hf:
        hf.create_dataset("train_images",  data=X_train_multi)
        hf.create_dataset("train_labels",  data=Y_train_multi)
        hf.create_dataset("test_images",  data=X_test_multi)
        hf.create_dataset("test_labels",  data=Y_test_multi)    


    print('Final Data Saved as hdf5 File : MNIST_multi.hdf5  in the directory datasets/pickles/')
    print('----- Process Complete -----')


if __name__ == '__main__':
    download_and_create_data()