from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
import h5py
from matplotlib import pyplot as plt



data_location = '/Users/himanshubabal/Documents/External_Disk_Link_WD_HDD/Study/SVHN/SVHN-Full_Dataset/'

pickle_file = data_location + 'SVHN_test_test__.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    test_data = save['test_dataset']
    test_label = save['test_labels']
    del save
    print(len(test_data))

print(test_data.shape)
print(test_label.shape)

def plot_img(image):
    plt.imshow(image)
    plt.show()


d = np.copy(test_data[0])
print(d.shape)

# d = np.delete(d, 2)
d = d.reshape(d.shape[:2])
print(d.shape)
plot_img(d)
























# test_data = test_data[:200]
# test_label = test_label[:200]

# print(test_data.shape)
# print(test_label.shape)

# pickle_file = data_location + 'SVHN_test_test__.pickle'

# try:
#   f = open(pickle_file, 'wb')
#   save = {
#     'test_dataset': test_data,
#     'test_labels': test_label,
#     }
#   pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#   f.close()
# except Exception as e:
#   print('Unable to save data to', pickle_file, ':', e)
#   raise
    
# statinfo = os.stat(pickle_file)
# print('Compressed pickle size:', statinfo.st_size)


