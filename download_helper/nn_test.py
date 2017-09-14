import numpy as np
import tensorflow as tf
import h5py
import math
import os

from path import data_dir, proj_dir
from nn_functions import *
from nn_graph import *
from nn_train import *


def get_data(hdf_file_path):

    hdf = h5py.File(hdf_file_path, 'r')

    train_images = hdf['train_images'][:]
    train_labels = hdf['train_labels'][:]
    train_bboxes = hdf['train_bboxes'][:]

    test_images = hdf['test_images'][:]
    test_labels = hdf['test_labels'][:]
    test_bboxes = hdf['test_bboxes'][:]

    valid_images = hdf['valid_images'][:]
    valid_labels = hdf['valid_labels'][:]
    valid_bboxes = hdf['valid_bboxes'][:]

    hdf.close()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)
    valid_images = valid_images.astype(np.float32)

    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    valid_labels = valid_labels.astype(np.int32)

    train_bboxes = train_bboxes.astype(np.int32)
    test_bboxes = test_bboxes.astype(np.int32)
    valid_bboxes = valid_bboxes.astype(np.int32)

    data = {}
    data['train'] = [train_images, train_labels, train_bboxes]
    data['test'] = [test_images, test_labels, test_bboxes]
    data['valid'] = [valid_images, valid_labels, valid_bboxes]

    return(data)


def get_trial_data(hdf_file_path):
    hdf = h5py.File(hdf_file_path, 'r')

    trial_images = hdf['trial_images'][:]
    trial_labels = hdf['trial_labels'][:]
    trial_bboxes = hdf['trial_bboxes'][:]

    hdf.close()

    train_images, train_labels, train_bboxes = trial_images[:70], trial_labels[:70], trial_bboxes[:70]
    test_images, test_labels, test_bboxes = trial_images[70:90], trial_labels[70:90], trial_bboxes[70:90]
    valid_images, valid_labels, valid_bboxes = trial_images[90:], trial_labels[90:], trial_bboxes[90:]

    data = {}
    data['train'] = [train_images, train_labels, train_bboxes]
    data['test'] = [test_images, test_labels, test_bboxes]
    data['valid'] = [valid_images, valid_labels, valid_bboxes]

    return(data)


# data = get_trial_data(hdf_file_path=data_dir + 'svhn_raw/SVHN_trial.hdf5')
# data = get_data(hdf_file_path=data_dir + 'svhn_raw/SVHN.hdf5')
data = get_data(hdf_file_path=data_dir + 'MNIST/MNIST.hdf5')
graph_trial, graph_vars = TF_Graph().create_graph()
train_model = TF_Train(data=data, TF_Graph=graph_trial, Graph_vars=graph_vars,
                       to_save_full_model=False, to_save_epoch_model=False,
                       model_save_path=proj_dir + 'saved_models/' + 'trial/',
                       model_save_name='trial', to_load_model=False, load_model_path='',
                       to_log=False, log_path=proj_dir + 'tf_logs/trial/', BATCH_SIZE=128, NUM_EPOCHS=5).train()
