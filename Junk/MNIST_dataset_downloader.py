from __future__ import print_function
import os
import sys
import gzip
from six.moves.urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder
import pickle
import scipy.io
import idx2numpy
import numpy as np
url = 'http://yann.lecun.com/exdb/mnist/'
last_percent_reported = None

'''
This File will download MNIST Dataset
and convert it into .mat format (28 x 28)
for using in the future
'''

mnist_dataset_location = "datasets/MNIST/"

def download_progress_hook(count, blockSize, totalSize):
	"""A hook to report the progress of a download. This is mostly intended for users with
	slow internet connections. Reports every 1% change in download progress.
	"""
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)

	if last_percent_reported != percent:
		if percent % 5 == 0:
		    sys.stdout.write("%s%%" % percent)
		    sys.stdout.flush()
		else:
		    sys.stdout.write(".")
		    sys.stdout.flush()
	  
		last_percent_reported = percent
        
def maybe_download(filename, force=False):
	"""Download a file if not present, and make sure it's the right size."""
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	return filename

def maybe_extract(filename, force=False):
	outfile = filename[:-3]
	if os.path.exists(outfile) and not force:
		# You may override by setting force=True.
		print('%s already present - Skipping extraction of %s.' % (outfile, filename))
	else:
		print('Extracting data for %s.' % outfile)
		inF = gzip.open(filename, 'rb')
		outF = open(outfile, 'wb')
		outF.write(inF.read())
		inF.close()
		outF.close()
	data_folders = outfile
	print(data_folders)
	return data_folders


train_images_zip = maybe_download(mnist_dataset_location + 'train-images-idx3-ubyte.gz')
train_labels_zip = maybe_download(mnist_dataset_location + 'train-labels-idx1-ubyte.gz')

test_images_zip = maybe_download(mnist_dataset_location + 't10k-images-idx3-ubyte.gz')
test_labels_zip = maybe_download(mnist_dataset_location + 't10k-labels-idx1-ubyte.gz')

print('MNIST Dataset Download Complete')
print('-------------------------------')

train_images_file = maybe_extract(mnist_dataset_location + 'train-images-idx3-ubyte.gz')
train_labels_file = maybe_extract(mnist_dataset_location + 'train-labels-idx1-ubyte.gz')

test_images_file = maybe_extract(mnist_dataset_location + 't10k-images-idx3-ubyte.gz')
test_labels_file = maybe_extract(mnist_dataset_location + 't10k-labels-idx1-ubyte.gz')

print('MNIST Dataset Extraction Complete')
print('---------------------------------')

train_images = idx2numpy.convert_from_file(mnist_dataset_location + 'train-images-idx3-ubyte')
train_label = idx2numpy.convert_from_file(mnist_dataset_location + 'train-labels-idx1-ubyte')

test_images = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-images-idx3-ubyte')
test_label = idx2numpy.convert_from_file(mnist_dataset_location + 't10k-labels-idx1-ubyte')

enc = OneHotEncoder()
train_label = enc.fit_transform(train_label.reshape(-1, 1)).toarray()
test_label = enc.fit_transform(test_label.reshape(-1, 1)).toarray()

train_label = (train_label.astype(np.float32))
test_label = (test_label.astype(np.float32))
test_images = (test_images.astype(np.float32))
train_images = (train_images.astype(np.float32))

print('One Hot Encoding Complete')

train_images = train_images[:,:,:,np.newaxis]
test_images = test_images[:,:,:,np.newaxis]

print('Data reshaped to n x 28 x 28 x 1')
print('Converting Data to .mat format for easy access')
print('----------------------------------------------')

train = {}
test = {}

train['X'] = train_images
train['y'] = train_label

test['X'] = test_images
test['y'] = test_label

scipy.io.savemat(mnist_dataset_location + 'mnist_default_train_28x28.mat', train)
scipy.io.savemat(mnist_dataset_location + 'mnist_default_test_28x28.mat', test)

print('Dataset saved as .mat files with names \nmnist_default_train_28x28.mat \nand mnist_default_test_28x28.mat')
print('Dataset Location : datasets/MNIST/')
print('Process Complete')