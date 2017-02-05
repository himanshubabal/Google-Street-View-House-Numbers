from __future__ import print_function
import os
import sys
import gzip
from six.moves.urllib.request import urlretrieve
from MNIST_Python_File.MNIST_downloader import download_progress_hook
from MNIST_Python_File.MNIST_create import download_and_create_data

url = 'https://www.dropbox.com/s/nzplzt9dh468mrz/SVHN_new_data_struct.pickle?dl=1'
pickle_location = 'datasets/pickles/'
last_percent_reported = None

def maybe_download(filename, force=True):
	"""Download a file if not present, and make sure it's the right size."""
	if force or not os.path.exists(filename):
		print('Attempting to download:', filename) 
		filename, _ = urlretrieve(url , filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(filename)
	return filename

# Will use 2nd Method if additional command line arguments are given
if (len(sys.argv) > 1):
	maybe_download(pickle_location + 'SVHN_new_data_struct.pickle')

print()
print('You have two Options to get MNIST Data : ')
print('1. Download the Data from Yann LeCun Website and create MNIST-Multi Data Now \n   It can be very Slow.')
print()
print('2. Download pre-processed pickle file  MNIST_multi.pickle  \n   It will save you a lot of time.')
print()

user_input = raw_input("Please choose method to get Data. \ni.e. Write 1 or 2 depending on your choice    : ")
if user_input == '1' :
	download_and_create_data()
else :
	maybe_download(pickle_location + 'SVHN_new_data_struct.pickle')

