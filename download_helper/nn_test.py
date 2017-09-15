import numpy as np
import tensorflow as tf
import h5py
import math
import os
import cPickle as pickle

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


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# trial_data = get_trial_data(hdf_file_path=data_dir + 'svhn_raw/SVHN_trial.hdf5')
# graph_trial, graph_vars = TF_Graph().create_graph()
# train_model = TF_Train(data=trial_data, TF_Graph=graph_trial, Graph_vars=graph_vars,
#                        to_save_full_model=True, to_save_epoch_model=True,
#                        model_save_path=proj_dir + 'saved_models/' + 'trial/',
#                        model_save_name='trial', to_load_model=True,
#                        load_model_dir=proj_dir + 'saved_models/' + 'trial/Full/',
#                        load_model_name='trial', to_log=True, log_path=proj_dir + 'tf_logs/trial/',
#                        BATCH_SIZE=2, NUM_EPOCHS=2).train()
# save_obj(train_model, proj_dir + 'saved_models/trial')

# tt = load_obj(proj_dir + 'saved_models/trial')
# print(tt)


# mnist_data = get_data(hdf_file_path=data_dir + 'MNIST/MNIST.hdf5')
# graph_trial, graph_vars = TF_Graph().create_graph()
# mnist_train_model = TF_Train(data=mnist_data,
#                              TF_Graph=graph_trial,
#                              Graph_vars=graph_vars,
#                              to_save_full_model=True,
#                              to_save_epoch_model=True,
#                              model_save_path=proj_dir + 'saved_models/' + 'MNIST/',
#                              model_save_name='mnist',
#                              to_load_model=False,
#                              load_model_dir='',
#                              load_model_name='',
#                              to_log=True,
#                              log_path=proj_dir + 'tf_logs/MNIST/',
#                              BATCH_SIZE=128,
#                              NUM_EPOCHS=20).train()

# save_obj(mnist_train_model, proj_dir + 'saved_models/mnist')
# del mnist_data, graph_trial, graph_vars, mnist_train_model

svhn_data = get_data(hdf_file_path=data_dir + 'svhn_raw/SVHN.hdf5')
# graph_trial, graph_vars = TF_Graph().create_graph()
# svhn_train_model = TF_Train(data=svhn_data,
#                             TF_Graph=graph_trial,
#                             Graph_vars=graph_vars,
#                             to_save_full_model=True,
#                             to_save_epoch_model=True,
#                             model_save_path=proj_dir + 'saved_models/' + 'svhn/',
#                             model_save_name='svhn',
#                             to_load_model=False,
#                             load_model_dir='',
#                             load_model_name='',
#                             to_log=True,
#                             log_path=proj_dir + 'tf_logs/svhn/',
#                             BATCH_SIZE=128,
#                             NUM_EPOCHS=20).train()
# save_obj(svhn_train_model, proj_dir + 'saved_models/svhn')
# del graph_trial, graph_vars, svhn_train_model

graph_trial, graph_vars = TF_Graph().create_graph()
svhn_mnist_train_model = TF_Train(data=svhn_data,
                                  TF_Graph=graph_trial,
                                  Graph_vars=graph_vars,
                                  to_save_full_model=True,
                                  to_save_epoch_model=True,
                                  model_save_path=proj_dir + 'saved_models/' + 'svhn_mnist/',
                                  model_save_name='svhn_mnist',
                                  to_load_model=True,
                                  load_model_dir=proj_dir + 'saved_models/' + 'MNIST/Full/',
                                  load_model_name='mnist',
                                  to_log=True,
                                  log_path=proj_dir + 'tf_logs/svhn_mnist/',
                                  BATCH_SIZE=128,
                                  NUM_EPOCHS=20).train()
save_obj(svhn_mnist_train_model, proj_dir + 'saved_models/svhn_mnist')
