from __future__ import print_function
import os
import sys
import gzip
import requests
from six.moves.urllib.request import urlretrieve
from download_helper.MNIST_create import download_and_create_data
from download_helper.SVHN_multi import prep_svhn_multi
from download_helper.SVHN_multi_box import prep_svhn_multi_box

prep_svhn_multi()

print('All Downloads Completed.')
print('------------------------')