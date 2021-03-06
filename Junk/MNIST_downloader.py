from __future__ import print_function
import os
import sys
import gzip
from six.moves.urllib.request import urlretrieve

last_percent_reported = None

'''
This File will download MNIST Dataset
and convert it into .mat format (28 x 28)
for using in the future
'''
url = 'http://yann.lecun.com/exdb/mnist/'
mnist_dataset_location = "/home/himanshubabal/Google-Street-View-House-Numbers/datasets/MNIST/"

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

def maybe_download(path, filename, force=False):
	"""Download a file if not present, and make sure it's the right size."""
	if force or not os.path.exists(path + filename):
		print('Attempting to download:', filename)
		filename, _ = urlretrieve(url + filename, path + filename, reporthook=download_progress_hook)
		print('\nDownload Complete!')
	statinfo = os.stat(path + filename)
	return path + filename

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
