import tensorflow as tf
import numpy as np

# Leaky ReLU


def leaky_relu(x, rate=0.01, name="leaky_relu"):
    with tf.name_scope(name) as scope:
        leak_rate = tf.multiply(x, rate, name="leak_rate")
        activation = tf.maximum(x, leak_rate, name=scope)
    return activation


# Batch Normallization
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


# Intersection / Union for calculating Bounding Box
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


# Calculate accuracy of bounding boxes
def accuracy_bboxes(predicted_bboxes, correct_bboxes):
    # Shape of bboxes -> n x 40
    n_bboxes = correct_bboxes.shape[1] // 4
    n_samples = correct_bboxes.shape[0]

    iou = np.empty(shape=[n_samples, n_bboxes])

    for i in range(n_bboxes):
        iou[:, i] = get_batch_iou(predicted_bboxes[:, 4 * i: 4 + 4 * i], correct_bboxes[:, 4 * i: 4 + 4 * i])

    a = iou.mean(axis=None) * 100
    return(a)


# Calculate accuracy of digits
def accuracy_digits(predicted_digits, correct_digits):
    a = (predicted_digits == correct_digits).mean(axis=0)

    # All correct prediction accuracy
    b = (predicted_digits == correct_digits).mean(axis=None) * 100

    # Individual correct prediction accuracy
    c = (predicted_digits == correct_digits).all(axis=1).mean() * 100

    return (b, c)


# Calculate loss for digits
def multi_digit_loss(logits_list, Y_, max_digits=11, name="multi_digit_loss"):
    with tf.name_scope(name) as scope:
        # LOSSES FOR EACH DIGIT BRANCH
        losses = [None] * (max_digits)
        for i in range(max_digits):
            losses[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_list[i], labels=Y_[:, i])

        # AVERAGE LOSS
        loss = sum(losses) / float(max_digits)
        loss = tf.reduce_mean(loss, name=scope)
    return loss


# Fully Connected Layer
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


# Flatten layer, useful in Convolution layer -> Fully Connected layer
def flatten_layer(x, name="flatten_layer"):
    with tf.name_scope(name) as scope:
        num_elements = np.product(x.get_shape().as_list()[1:])
        x = tf.reshape(x, [-1, num_elements], name=scope)
    return x


# Convolution Laytr
def conv_pipeline(X_in, in_width, out_width, fltr_conv, stride_conv, is_train, iteration, pkeep, token=1, conv=True):
    with tf.name_scope('convolution_' + str(token)):
        W = tf.Variable(tf.truncated_normal([fltr_conv, fltr_conv, in_width, out_width], stddev=0.1))
        B = tf.Variable(tf.constant(0.1, tf.float32, [out_width]))

        Y_conv = tf.nn.conv2d(X_in, W, strides=[1, stride_conv, stride_conv, 1], padding='SAME') + B
        Y_bnorm = batchnorm(Y_conv, is_train, iteration, conv)
        Y_drop = tf.nn.dropout(Y_bnorm, pkeep)
        Y_relu = leaky_relu(Y_drop)
    return(Y_relu)
