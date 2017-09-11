import tensorflow as tf


class TF_Graph():
    def __init__(self, X_, Y_, Z_, BATCH_SIZE, learning_rate, HEIGHT=64, WIDTH=64,
                 no_of_digits=10, max_possible_var=11):
        self.graph = tf.graph()
        self.X_ = X_
        self.Y_ = Y_
        self.Z_ = Z_
        self.BATCH_SIZE = BATCH_SIZE
        self.learning_rate = learning_rate
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.no_of_digits = no_of_digits
        self.max_possible_var = max_possible_var

    def create_graph():
        with self.graph.as_default():
            with tf.name_scope('input'):
                # Image
                X_ = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH], name='X_input')
                X_ = X_ / 255
                X = tf.reshape(X_, shape=[-1, HEIGHT, WIDTH, 1], name='X_input_reshaped')

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
            # Max Pool   # 3        2         8 x  8 x 256 (= 16384)

            # Fully Connected / Dense
            R = 4096
            S = 4096
            T = 512
            U = 64
            V = 256

            Y1 = conv_pipeline(X, in_width=1, out_width=K, fltr_conv=3, stride_conv=1, is_train=is_train, iteration=iteration, pkeep=pkeep, token=1)
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
            for i in range(no_of_digits + 1):
                d_logits[i] = fc_pipeline(Y_digits, max_possible_var, is_train, iteration, pkeep=1.0, token=410 + i)
            digits_logits = tf.stack(d_logits, axis=0)

            bboxes_logits = fc_pipeline(Y_bboxes, no_of_digits * 4, is_train, iteration, pkeep=1.0, token=421)

            with tf.name_scope('loss_function'):
                loss_digits = multi_digit_loss(digits_logits, Y_, max_digits=no_of_digits + 1, name="loss_digits")
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

    return(self.graph, [optimizer_box, optimizer_digit, digits_preds, bboxes_preds,
                        model_saver, summary_op, loss_digits, loss_bboxes, loss_total])
