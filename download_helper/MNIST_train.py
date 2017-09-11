import numpy as np
import tensorflow as tf
import random
import math
import h5py
import gc
import sys
import os

import matplotlib.pyplot as plt
from PIL import Image
from download_helper.path import data_dir

hdf_file = data_dir + 'MNIST/MNIST_manufactured_data.hdf5'

hdf = h5py.File(hdf_file, 'r')

train_images = hdf['train_images'][:]
train_labels = hdf['train_labels'][:]
train_bboxes = hdf['train_bboxes'][:]

test_images = hdf['test_images'][:]
test_labels = hdf['test_labels'][:]
test_bboxes = hdf['test_bboxes'][:]

hdf.close()


valid_inputs = 5000
l = len(train_labels)

valid_images = train_images[l - valid_inputs:]
valid_labels = train_labels[l - valid_inputs:]
valid_bboxes = train_bboxes[l - valid_inputs:]

train_images = train_images[:l - valid_inputs]
train_labels = train_labels[:l - valid_inputs]
train_bboxes = train_bboxes[:l - valid_inputs]

train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)
valid_images = valid_images.astype(np.float32)

train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)
valid_labels = valid_labels.astype(np.int32)

train_bboxes = train_bboxes.astype(np.int32)
test_bboxes = test_bboxes.astype(np.int32)
valid_bboxes = valid_bboxes.astype(np.int32)

print('Training set', train_images.shape, train_labels.shape, train_bboxes.shape)
print('Test set', test_images.shape, test_labels.shape, test_bboxes.shape)
print('Valid set', valid_images.shape, valid_labels.shape, valid_bboxes.shape)

# Rectangle
# (x1, y1) & (a1, b2):
#          Top Left Corner for
#          Rectangle 1 and 2 respectively
# (x2, y2) & (a2, b2):
#          Top Left Corner for
#          Rectangle 1 and 2 respectively


def get_iou(x1, y1, x2, y2, a1, b1, a2, b2):
    x_overlap = max(0, min(x1 + x2, a1 + a2) - max(x1, a1))
    y_overlap = max(0, min(y1 + y2, b1 + b2) - max(y1, b1))
    intersection_area = x_overlap * y_overlap

    # areas of both rectangles
    area_1 = abs(x2 - x1) * abs(y2 - y1)
    area_2 = abs(a2 - a1) * abs(b2 - b1)
    # Total (Union) area
    union_area = area_1 + area_2 - intersection_area

    iou = (intersection_area * 1.0) / union_area
    return(iou)


def batchnorm(x, is_training, iteration, conv=False, offset=0.0, scale=1.0):
    """
    Credits
    -------
    This code is based on code written by Martin Gorner:
    - https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.2_batchnorm_convolutional.py
    """
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration)
    bnepsilon = 1e-5

    # calculate mean and variance for batch of logits
    if conv:
        mean, variance = tf.nn.moments(x, [0, 1, 2])
    else:
        # mean and variance along the batch
        mean, variance = tf.nn.moments(x, [0])

    update_moving_averages = exp_moving_avg.apply([mean, variance])
    tf.add_to_collection("update_moving_averages", update_moving_averages)

    # Mean and Variance (how it get it is dependent on whether it is training)
    # TODO: Change the following to use the `is_trianing` directly without logical_not()
    #       to make it more intuitive.
    m = tf.cond(tf.logical_not(is_training),
                lambda: exp_moving_avg.average(mean),
                lambda: mean)
    v = tf.cond(tf.logical_not(is_training),
                lambda: exp_moving_avg.average(variance),
                lambda: variance)

    # Offset
    param_shape = mean.get_shape().as_list()
    beta_init = tf.constant_initializer(offset)
    beta = tf.Variable(initial_value=beta_init(param_shape), name="beta")

    # Scale
    gamma_init = tf.constant_initializer(scale)
    gamma = tf.Variable(initial_value=gamma_init(param_shape), name="gamma")

    # Apply Batch Norm
    Ybn = tf.nn.batch_normalization(x, m, v, offset=beta, scale=gamma,
                                    variance_epsilon=bnepsilon, name='batchnorm')
    return Ybn


def leaky_relu(x, rate=0.01, name="leaky_relu"):
    with tf.name_scope(name) as scope:
        leak_rate = tf.multiply(x, rate, name="leak_rate")
        activation = tf.maximum(x, leak_rate, name=scope)
    return activation


def conv_pipeline(X_in, in_width, out_width, fltr_conv, stride_conv, is_train, iteration, pkeep, token=1, conv=True):
    with tf.name_scope('convolution_' + str(token)):
        W = tf.Variable(tf.truncated_normal([fltr_conv, fltr_conv, in_width, out_width], stddev=0.1))
        B = tf.Variable(tf.constant(0.1, tf.float32, [out_width]))

        Y_conv = tf.nn.conv2d(X_in, W, strides=[1, stride_conv, stride_conv, 1], padding='SAME') + B
        Y_bnorm = batchnorm(Y_conv, is_train, iteration, conv)
        Y_drop = tf.nn.dropout(Y_bnorm, pkeep)
        Y_relu = leaky_relu(Y_drop)
    return(Y_relu)


def flatten_layer(x, name="flatten_layer"):
    with tf.name_scope(name) as scope:
        num_elements = np.product(x.get_shape().as_list()[1:])
        x = tf.reshape(x, [-1, num_elements], name=scope)
    return x


def fc_pipeline(X_in, num_nodes, is_train, iteration, pkeep, token=1, conv=False):
    in_nodes = int(X_in.shape.as_list()[-1])

    with tf.name_scope('fully_connected_' + str(token)):
        W = tf.Variable(tf.truncated_normal([in_nodes, num_nodes], stddev=0.1))
        B = tf.Variable(tf.constant(0.1, tf.float32, [num_nodes]))

        Y_fc = tf.matmul(X_in, W) + B
        Y_bn = batchnorm(Y_fc, is_train, iteration, conv)
        Y_dp = tf.nn.dropout(Y_bn, keep_prob=pkeep)
        Y_lr = leaky_relu(Y_dp, rate=0.01, name="relu")
    return(Y_lr)


def multi_digit_loss(logits_list, Y_, max_digits=11, name="multi_digit_loss"):
    with tf.name_scope(name) as scope:
        # LOSSES FOR EACH DIGIT BRANCH
        losses = [None] * (max_digits)
        for i in range(max_digits):
            losses[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_list[i], labels=Y_[:,i])

        # AVERAGE LOSS
        loss = sum(losses) / float(max_digits)
        loss = tf.reduce_mean(loss, name=scope)
    return loss


log_path = 'tf_logs/MNIST/'

def accuracy_digits(predicted_digits, correct_digits):
    a = (predicted_digits == correct_digits).mean(axis=0)

    # All correct prediction accuracy
    b = (predicted_digits == correct_digits).mean(axis=None)*100

    # Individual correct prediction accuracy
    c = (predicted_digits == correct_digits).all(axis=1).mean()*100

    return (b, c)


def accuracy_bboxes(predicted_bboxes, correct_bboxes):
    # Shape of bboxes -> n x 40
    n_bboxes = correct_bboxes.shape[1] // 4
    n_samples = correct_bboxes.shape[0]

    iou = np.empty(shape=[n_samples, n_bboxes])

    for i in range(n_bboxes):
        iou[:,i] = get_batch_iou(predicted_bboxes[:, 4*i: 4+4*i], correct_bboxes[:, 4*i: 4+4*i])

    #                         - - - - - - - - -
    # return(iou.mean(axis=0), iou.mean(axis=None), iou.mean(axis=1))
    a = iou.mean(axis=None)*100
    return(a)


def get_batch_iou(a, b):
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + 1e-8)
    return(iou)


HEIGHT = 64
WIDTH = 64
no_of_digits = 10
# 11 digits possible for each place -> 0,1,2,3,4,5,6,7,8,9,10
max_possible_var = 11

graph_svhn = tf.Graph()

with graph_svhn.as_default():
    with tf.name_scope('input'):
        # Image
        X_ = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH], name='X_input')
        X_ = X_ / 255.0
        X = tf.reshape(X_, shape=[-1, HEIGHT, WIDTH, 1], name='X_input_reshaped')
        # print('X : ', X.shape.as_list())

        # Label
        Y_ = tf.placeholder(tf.int32, [None, no_of_digits + 1], name='Labels')
        # Bounding Box
        Z_ = tf.placeholder(tf.int32, [None, no_of_digits * 4], name='Bboxes')

        # Learning Rate - alpha
        alpha = tf.placeholder(tf.float32, name='Learning_Rate')
        # Dropout (or better : 1 - toDropOut) Probablity
        pkeep = tf.placeholder(tf.float32, name='Dropout-pkeep')
        # Model trainig or testing
        is_train = tf.placeholder(tf.bool, name='Is_Training')
        # Iteration
        iteration = tf.placeholder(tf.int32, name='Iteration-i')

    # Depth      # Filter   Stride   Size
    K = 6        # 3        1        64 x 64 x 6
    L = 24       # 3        1        64 x 64 x 24
    M = 96       # 5        1        64 x 64 x 96
    # MAX POOL   # 3        2        32 x 32 x 24
    N = 48       # 3        1        32 x 32 x 48
    O = 96       # 5        1        32 x 32 x 96
    P = 256      # 3        1        32 x 32 x 256
    # MAX POOL   # 3        2        16 x 16 x 256
    Q = 256      # 5        1        16 x 16 x 256
    J = 256      # 3        1        16 x 16 x 256
    # Max Pool   # 3        2         8 x  8 x 256

    # Fully Connected / Dense
    R = 4096
    S = 4096
    T = 512
    U = 64
    V = 256


    Y1 = conv_pipeline(X,  in_width=1, out_width=K, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=1)
    Y1 = conv_pipeline(Y1, in_width=K, out_width=L, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=2)
    Y1 = conv_pipeline(Y1, in_width=L, out_width=M, fltr_conv=5, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=3)

    Y1 = tf.nn.max_pool(Y1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_Pool_1')

    Y1 = conv_pipeline(Y1, in_width=M, out_width=N, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=4)
    Y1 = conv_pipeline(Y1, in_width=N, out_width=O, fltr_conv=5, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=5)
    Y1 = conv_pipeline(Y1, in_width=O, out_width=P, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=6)

    Y1 = tf.nn.max_pool(Y1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_Pool_2')

    Y1 = conv_pipeline(Y1, in_width=P, out_width=Q, fltr_conv=5, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=7)
    Y1 = conv_pipeline(Y1, in_width=Q, out_width=J, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=8)

    Y1 = tf.nn.max_pool(Y1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_Pool_3')


    Y1 = flatten_layer(Y1)

    Y1 = fc_pipeline(Y1, R, is_train, iteration, pkeep, token=1)
    Y1 = fc_pipeline(Y1, S, is_train, iteration, pkeep, token=2)
    Y1 = fc_pipeline(Y1, T, is_train, iteration, pkeep, token=3)

    Y_digits = fc_pipeline(Y1, U, is_train, iteration, pkeep=1.0, token=41)
    Y_bboxes = fc_pipeline(Y1, V, is_train, iteration, pkeep=1.0, token=42)


    d_logits = [None] * (no_of_digits + 1)
    for i in range(no_of_digits+1):
        d_logits[i] = fc_pipeline(Y_digits, max_possible_var, is_train, iteration, pkeep=1.0, token=410+i)
    digits_logits = tf.stack(d_logits, axis=0)
    # print(digits_logits.shape.as_list())

    bboxes_logits = fc_pipeline(Y_bboxes, no_of_digits * 4, is_train, iteration, pkeep=1.0, token=421)
    # print(bboxes_logits.shape.as_list())

    # print(Y_.shape.as_list())

    with tf.name_scope('loss_function'):
        loss_digits = multi_digit_loss(digits_logits, Y_, max_digits=no_of_digits+1, name="loss_digits")
        loss_bboxes = tf.sqrt(tf.reduce_mean(tf.square(1 * (bboxes_logits - tf.to_float(Z_)))), name="loss_bboxes")
        loss_total = tf.add(loss_bboxes, loss_digits, name="loss_total")

    with tf.name_scope('optimisers'):
        optimizer_digit = tf.train.AdamOptimizer(learning_rate=alpha,
                                   beta1=0.9, beta2=0.999,
                                   epsilon=1e-08,
                                   name="optimizer_digits").minimize(loss_digits)

        optimizer_box = tf.train.AdamOptimizer(learning_rate=alpha,
                                   beta1=0.9, beta2=0.999,
                                   epsilon=1e-08,
                                   name="optimizer_boxes").minimize(loss_bboxes)


    digits_preds = tf.transpose(tf.argmax(digits_logits, axis=2))
    digits_preds = tf.to_int32(digits_preds, name="digit_predictions")

    bboxes_preds = tf.to_int32(bboxes_logits, name='box_predictions')

    tf.summary.scalar("loss_digits", loss_digits)
    tf.summary.scalar("loss_bboxes", loss_bboxes)
    tf.summary.scalar("loss_total", loss_total)

    model_saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()



model_to_save = "saved_models/MNIST/"

print('Training set', train_images.shape, train_labels.shape, train_bboxes.shape)
print('Test set', test_images.shape, test_labels.shape, test_bboxes.shape)
print('Valid set', valid_images.shape, valid_labels.shape, valid_bboxes.shape)

batch_size = 128
num_steps = int(train_labels.shape[0] / batch_size)
num_epochs = 20
print('Batch Size: ', batch_size, ' num_steps: ', num_steps, ' num_epochs: ', num_epochs)

with tf.Session(graph=graph_svhn) as session:
    print('')
    print('Initalizing...')
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(log_path, graph=graph_svhn)
    print('Initialized')
    print('')

    valid_i = valid_images[:batch_size]
    valid_l = valid_labels[:batch_size]
    valid_b = valid_bboxes[:batch_size]

    for epoch in range(num_epochs - 1):
        test_i = test_images[epoch*batch_size:(epoch+1)*batch_size]
        test_l = test_labels[epoch*batch_size:(epoch+1)*batch_size]
        test_b = test_bboxes[epoch*batch_size:(epoch+1)*batch_size]

        for step in range(num_steps - 1):
            max_learning_rate = 0.0005
            min_learning_rate = 0.0001

            decay_speed = 5000.0
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-step/decay_speed)
            # learning_rate = 0.0001

            batch_data   = train_images[step*batch_size:(step + 1)*batch_size]
            batch_labels = train_labels[step*batch_size:(step + 1)*batch_size]
            batch_bboxes = train_bboxes[step*batch_size:(step + 1)*batch_size]

            feed_dict = {X_ : batch_data, Y_ : batch_labels, Z_ : batch_bboxes, pkeep : 0.90, alpha : learning_rate,
                            is_train : True, iteration : step}

            _, _, loss_digit, loss_box, digits, bboxes, summary = session.run([optimizer_digit, optimizer_box,
                                                                   loss_digits, loss_bboxes, digits_preds,
                                                                   bboxes_preds, summary_op], feed_dict=feed_dict)
            writer.add_summary(summary, step)

            if step % int(num_steps/2) == 0:
                print('Accuracy digits - Individual : %.2f%%'% accuracy_digits(digits, batch_labels)[0])
                print('Accuracy digits - All        : %.2f%%'% accuracy_digits(digits, batch_labels)[0])
                print('Accuracy bboxes - All        : %.2f%%'% accuracy_bboxes(bboxes, batch_bboxes))
                print('Loss - Digits                : %.2f%'% loss_digit)
                print('Loss - Bboxes                : %.2f%'% loss_box)
                print('')


        # Get Test accuracy
        print('------------------------------------------')
        feed_dict = {X_ : test_i, Y_ : test_l, Z_ : test_b, pkeep : 1.0, alpha : 1e-9,
                            is_train : False, iteration : epoch}
        loss_digit, loss_box, digits, bboxes = session.run([loss_digits, loss_bboxes, digits_preds, bboxes_preds], feed_dict=feed_dict)

        print('Epoch                      ==> ' + str(epoch + 1))
        print('Accuracy digits - Individual : %.2f%%'% accuracy_digits(digits, test_l)[0])
        print('Accuracy digits - All        : %.2f%%'% accuracy_digits(digits, test_l)[0])
        print('Accuracy bboxes - All        : %.2f%%'% accuracy_bboxes(bboxes, test_b))
        print('Loss - Digits                : %.2f%'% loss_digit)
        print('Loss - Bboxes                : %.2f%'% loss_box)
        print('------------------------------------------')
        print('      ')

        path = model_to_save + str(epoch+1) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        model_saver.save(session, path + 'MNIST-' + str(epoch+1))

    # Get Valid accuracy
    feed_dict = {X_ : valid_i, Y_ : valid_l, Z_ : valid_b, pkeep : 1.0, alpha : 1e-9,
                            is_train : False, iteration : epoch}
    loss_digit, loss_box, digits, bboxes = session.run([loss_digits, loss_bboxes, digits_preds, bboxes_preds], feed_dict=feed_dict)

    print('=========================================')
    print('    FINAL ACCURACY   ')
    print('Accuracy digits - Individual : %.2f%%'% accuracy_digits(digits, valid_l)[0])
    print('Accuracy digits - All        : %.2f%%'% accuracy_digits(digits, valid_l)[0])
    print('Accuracy bboxes - All        : %.2f%%'% accuracy_bboxes(bboxes, valid_b))
    print('Loss - Digits                : %.2f%'% loss_digit)
    print('Loss - Bboxes                : %.2f%'% loss_box)
    print('=========================================')
    print('      ')

    print('Training Complete on MNIST Data')
    save_path = model_saver.save(session, model_to_save + 'MNIST')
    print("Model saved in file: %s" % save_path)

