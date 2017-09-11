from __future__ import print_function
from six.moves.urllib.request import urlretrieve
import os
import sys
import gzip
import tarfile

last_percent_reported = None

'''
This File will download Datasets
and unzip them
'''


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


def maybe_download(path, filename, url, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(path + filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url, path + filename, reporthook=download_progress_hook)
        print(filename)
        print('\nDownload Complete!')
        statinfo = os.stat(filename)

    else:
        filename = path + filename
        statinfo = os.stat(filename)
    return (filename)

# Extract .gz files

# filename -> should include path


def maybe_extract(filename, force=False):
    # train.tar.gz -> train
    if (filename.endswith('tar.gz')):
        outfile = filename[:-7]
    # train-images-idx3-ubyte.gz -> train-images-idx3-ubyte
    elif (filename.endswith('gz')):
        outfile = filename[:-3]

    print('Outfile : ', outfile)

    # Check if extraced file already present
    if os.path.exists(outfile) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (outfile, filename))
    else:
        if (filename.endswith("tar.gz")):
            tar = tarfile.open(filename, "r:gz")
            tar.extractall(outfile)
            tar.close()

        elif (filename.endswith("tar")):
            tar = tarfile.open(filename, "r:")
            tar.extractall(outfile)
            tar.close()

        elif(filename.endswith('gz')):
            print('Extracting data for %s.' % outfile)
            inF = gzip.open(filename, 'rb')
            outF = open(outfile, 'wb')
            outF.write(inF.read())
            inF.close()
            outF.close()

    print('Extracted : ', outfile)
    return outfile


# mnist_url = 'http://yann.lecun.com/exdb/mnist/'
# mnist_file_name = 'train-images-idx3-ubyte.gz'
# mnist_dataset_location = "/Users/himanshubabal/Desktop/test/"

# mnist_zip = maybe_download(mnist_dataset_location, mnist_file_name, mnist_url + mnist_file_name)
# print(mnist_zip)
# mnist_folder = maybe_extract(mnist_zip)


# svhn_url = 'http://ufldl.stanford.edu/housenumbers/'
# svhn_file = 'test.tar.gz'
# svhn_dataset_location = '/Users/himanshubabal/Desktop/test/'

# svhn_zip = maybe_download(svhn_dataset_location, svhn_file, svhn_url + svhn_file)
# print(svhn_zip)
# svhn_folder = maybe_extract(svhn_zip)
