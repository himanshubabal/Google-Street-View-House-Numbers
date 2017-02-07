from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import pickle
import scipy.io
import idx2numpy
import random
from matplotlib import pyplot as plt
import scipy.misc

'''
This File will generate synthetic data by combining upto 5
random digits from MNIST dataset, so the generated data can be similar to 
SVHN Data
'''
mnist_dataset_location = "datasets/MNIST/"

multi_image_width = 32*3
multi_image_height = 32


def random_insert_seq(lst, seq):
    insert_locations = random.sample(xrange(len(lst) + len(seq)), len(seq))
    inserts = dict(zip(insert_locations, seq))
    input = iter(lst)
    lst[:] = [inserts[pos] if pos in inserts else next(input)
        for pos in xrange(len(lst) + len(seq))]

def randomize_inputs(X_training, Y_labelling, no_of_white_images):
    w = X_training.shape[1]
    n = X_training.shape[0]
    k = no_of_white_images
    # print ('w : ', w, ' n : ', n, ' k : ', k)
    
    Y_expand_label = np.zeros((n, 1, w), dtype=np.int)
    for i in range(0,n):
        Y_expand_label[i,:,:] = Y_labelling[i]
        
    data_expand = np.concatenate((X_training ,Y_expand_label), axis=1)
    
    assert (data_expand.shape[1] == w + 1)
    
    # 'white_image_mat' and 'white_image_label' prepare 'k' white images and their label
    white_image_mat = np.zeros((k, w, w))
    white_image_label = np.full((k, 1, w), None)
    
    white_images_big = np.concatenate((white_image_mat ,white_image_label), axis=1)
    
    assert (white_images_big.shape[1] == w + 1)
    
    data_big = np.concatenate((data_expand ,white_images_big))
    
    assert (data_big.shape[0] == n + k and data_big.shape[1] == w + 1 and data_big.shape[2] == w)
    
    np.random.shuffle(data_big)
    assert (data_big.shape[0] == n + k and data_big.shape[1] == w + 1 and data_big.shape[2] == w)
    
    count = 0
    list_of_outputs = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]
    for i in range(0, n + k):
        if data_big[i][28][0] not in list_of_outputs :
            count += 1

    assert (count == k)
    
    Y_out = data_big[:,w][:,0]
    X_out = np.delete(data_big, w, axis=1)
    
    return X_out, Y_out

def random_list_length_generator(start_pt, end_pt, part_length, list_length) :
    random_list = []
    for i in range(list_length):
        random_list.append(random.sample(range(start_pt, end_pt), part_length))
        
    return random_list

def stack_matrices_horizontally(list_of_matrices, width_of_image):
    w_mat = []
    non_w_mat = []
    
    white_matrix = np.zeros((width_of_image, width_of_image))
    
    for mat in list_of_matrices :
        assert white_matrix.shape == mat.shape
        if ((mat == white_matrix).all()) :
            w_mat.append(mat)
        else :
            non_w_mat.append(mat)
    
    f_list = non_w_mat + w_mat
        
    hstack = np.hstack((f_list))
    # Resizing image from 28x140 -> 32x96
    hstack = scipy.misc.imresize(hstack, (multi_image_height, multi_image_width))

    return hstack


def stack_labels_horizontally(list_of_outputs, no_of_images_to_combine):
    possible_list = [0,1,2,3,4,5,6,7,8,9]
    
    # 0 --> 10
    outs = []
    for i in list_of_outputs :
        if i in possible_list :
            if i != 0:
                outs.append(int(i))
            else :
                outs.append(10)
    
    out_l = len(outs)
    
    no_of_blanks = no_of_images_to_combine - out_l
    
    # Blank --> 0
    blank_list = []
    for i in range(no_of_blanks) :
        blank_list.append(0)
    
    f_list = [out_l] + outs + blank_list
    
    return f_list

def multiple_img_dataset_generator(X_dataset, Y_dataset, no_of_images_to_combine, length_of_new_dataset) :
    assert (X_dataset.shape[0] == Y_dataset.shape[0])
    
    n = X_dataset.shape[0]
    w = X_dataset.shape[1]        # Assuming image have same width and height
    c = no_of_images_to_combine
    k = length_of_new_dataset
    
    rand_list = random_list_length_generator(0, n, c, k)
        
    
    # X_new = np.zeros((k, w, c*w))
    X_new = np.zeros((k, multi_image_height, multi_image_width))
    Y_new = np.zeros((k, c+1), dtype=np.int)
    
    for i in range(k):
        # Get 5 random indexes
        rand_index = rand_list[i]

        images_index = []
        label_index = []
        
        for r in rand_index :
            images_index.append(X_dataset[r])
            label_index.append(Y_dataset[r])

        stacked_images = stack_matrices_horizontally(images_index, w)
        stacked_labels = stack_labels_horizontally(label_index, c)

        X_new[i] = stacked_images
        Y_new[i] = stacked_labels
    
    print('Matrix Conversion Successful')
    
    return X_new, Y_new