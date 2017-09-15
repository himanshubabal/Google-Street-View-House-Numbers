from nn_functions import *
import os
import math
import numpy as np
from path import proj_dir


class TF_Train():
    # data --> dictonaty
    #          data['train'] = [train_images, train_labels, train_bboxes]
    #          data['test'] = [test_images, test_labels, test_bboxes]
    #          data['valid'] = [valid_images, valid_labels, valid_bboxes]
    # model_save_path, log_path -> must end in / (i.e. in folder location)
    #                              model will be saved at model_save_path/model_name
    #                               with each epoch having its seperate folder and final
    #                               model in root folder
    #               model_load_path -> must contain saved model name
    # Graph_vars -> [X_, Y_, Z_, aplha, pkeep, is_train, iteration, optimizer_box,
    #                optimizer_digit, optimizer_all, digits_preds, bboxes_preds,
    #                model_saver, summary_op, loss_digits, loss_bboxes, loss_total]
    def __init__(self, data, TF_Graph, Graph_vars, to_save_full_model=True, to_save_epoch_model=True,
                 model_save_path=proj_dir + 'saved_models', model_save_name='tf', to_load_model=False,
                 load_model_dir=proj_dir + 'saved_models/', load_model_name='tf', to_log=False,
                 log_path=proj_dir + 'tf_logs', BATCH_SIZE=128, NUM_EPOCHS=5, verbose=True):
        self.train_images = data['train'][0]
        self.train_labels = data['train'][1]
        self.train_bboxes = data['train'][2]
        self.test_images = data['test'][0]
        self.test_labels = data['test'][1]
        self.test_bboxes = data['test'][2]
        self.valid_images = data['valid'][0]
        self.valid_labels = data['valid'][1]
        self.valid_bboxes = data['valid'][2]

        self.TF_Graph = TF_Graph
        self.Graph_vars = Graph_vars

        self.to_save_epoch_model = to_save_epoch_model
        self.to_save_full_model = to_save_full_model
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.to_load_model = to_load_model
        self.load_model_dir = load_model_dir
        self.load_model_name = load_model_name
        self.to_log = to_log
        self.log_path = log_path

        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.verbose = verbose

    # @RETURNS
    #           dict -> results_dict
    #                   results_dict['train'] -> train_results
    #                                            train_results[i] = [-,-,-,-,-]
    #                   results_dict['test']  -> test_results[epoch] -> [-,-,-]
    #                   results_dict['valid'] -> [-,-,-]
    def train(self):
        results_dict, train_results, test_results = {}, {}, {}

        [X_, Y_, Z_, alpha, pkeep, is_train, iteration, optimizer_box,
         optimizer_digit, optimizer_all, digits_preds, bboxes_preds,
         model_saver, summary_op, loss_digits, loss_bboxes, loss_total] = self.Graph_vars

        if self.verbose:
            print('Training set : ', self.train_images.shape, self.train_labels.shape,
                  self.train_bboxes.shape)
            print('Test set : ', self.test_images.shape, self.test_labels.shape,
                  self.test_bboxes.shape)
            print('Valid set : ', self.valid_images.shape, self.valid_labels.shape,
                  self.valid_bboxes.shape)

        NUM_STEPS = self.train_labels.shape[0] // self.BATCH_SIZE
        BATCH_TEST = self.test_labels.shape[0] // self.BATCH_SIZE // self.NUM_EPOCHS
        BATCH_VALID = self.valid_labels.shape[0] // self.BATCH_SIZE // self.NUM_EPOCHS

        if self.verbose:
            print('Batch Size: ', self.BATCH_SIZE, ' num_steps: ', NUM_STEPS, ' num_epochs: ',
                  self.NUM_EPOCHS, 'Batch Test : ', BATCH_TEST, 'Batch Valid : ', BATCH_VALID)

        with tf.Session(graph=self.TF_Graph) as session:
            if self.verbose:
                print('')
                print('Initalizing...')

            # Load saved model or initalize a new one
            if os.path.isfile(self.load_model_dir + self.load_model_name + '.meta'):
                if self.to_load_model:
                    print('Saved Model found')
                    model_saver = tf.train.import_meta_graph(self.load_model_dir +
                                                             self.load_model_name + '.meta')
                    model_saver.restore(session, tf.train.latest_checkpoint(self.load_model_dir + './'))
                    print('Loaded Saved Model')
            else:
                tf.global_variables_initializer().run()

            # TF Summary writer
            if self.to_log:
                writer = tf.summary.FileWriter(self.log_path, graph=self.TF_Graph)
            summary_token = 0

            if self.verbose:
                print('Initialized')
                print('')

            for epoch in range(self.NUM_EPOCHS):
                for step in range(NUM_STEPS):
                    # Decaying Learning rate
                    max_learning_rate = 0.001
                    min_learning_rate = 0.0001

                    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * \
                        math.exp(-1 * step / NUM_STEPS)

                    # Selecting Batch at random, I think it will be better in training
                    idx = np.random.choice(np.arange(len(self.train_labels)), self.BATCH_SIZE, replace=False)
                    batch_images = self.train_images[idx]
                    batch_labels = self.train_labels[idx]
                    batch_bboxes = self.train_bboxes[idx]

                    feed_dict = {X_: batch_images, Y_: batch_labels, Z_: batch_bboxes, pkeep: 0.90,
                                 alpha: learning_rate, is_train: True, iteration: step}

                    _, loss_digit, loss_box, digits, bboxes, summary = session.run([optimizer_all,
                                                                                    loss_digits,
                                                                                    loss_bboxes,
                                                                                    digits_preds,
                                                                                    bboxes_preds,
                                                                                    summary_op],
                                                                                   feed_dict=feed_dict)

                    if self.to_log:
                        # Add summary for tensorbaord
                        writer.add_summary(summary, summary_token)

                    acc_single = accuracy_digits(digits, batch_labels)[0]
                    acc_all = accuracy_digits(digits, batch_labels)[1]
                    acc_box = accuracy_bboxes(bboxes, batch_bboxes)

                    train_results[summary_token] = [acc_single, acc_all, acc_box, loss_digit, loss_box]

                    # For each epoch, display result 2 times
                    if self.verbose:
                        if step % int(NUM_STEPS / 2) == 0:
                            print('Accuracy digits - Individual : %.2f%%' % acc_single)
                            print('Accuracy digits - All        : %.2f%%' % acc_all)
                            print('Accuracy bboxes - All        : %.2f%%' % acc_box)
                            print('Loss - Digits                : %.2f' % loss_digit)
                            print('Loss - Bboxes                : %.2f' % loss_box)
                            print('')
                    summary_token += 1

                # Get Test accuracy
                test_results[epoch] = self.get_results(0, BATCH_TEST, self.test_images, self.test_labels,
                                                       self.test_bboxes, session, epoch)

                if self.to_save_epoch_model:
                    if not os.path.exists(self.model_save_path + str(epoch + 1) + '/'):
                        os.makedirs(self.model_save_path + str(epoch + 1) + '/')
                    model_saver.save(session, self.model_save_path + str(epoch + 1) + '/' +
                                     self.model_save_name + '-' + str(epoch + 1))

            results_dict['train'] = train_results
            results_dict['test'] = test_results
            # Get Valid accuracy
            results_dict['valid'] = self.get_results(1, BATCH_VALID, self.valid_images, self.valid_labels,
                                                     self.valid_bboxes, session, 0)

            print('Training Complete on %s Data' % self.model_save_name)
            if self.to_save_full_model:
                if not os.path.exists(self.model_save_path + 'Full/'):
                    os.makedirs(self.model_save_path + 'Full/')
                save_path = model_saver.save(session, self.model_save_path + 'Full/' + self.model_save_name)
                print('Model saved in file: %s' % self.model_save_path + self.model_save_name)

            return(results_dict)

    # TOKEN : 0 -> Test
    #         1 -> Valid
    # SIZE : BATCH_TEST or BATCH_VALID
    def get_results(self, TOKEN, SIZE, images, labels, boxes, session, epoch):
        [X_, Y_, Z_, alpha, pkeep, is_train, iteration, optimizer_box,
         optimizer_digit, optimizer_all, digits_preds, bboxes_preds,
         model_saver, summary_op, loss_digits, loss_bboxes, loss_total] = self.Graph_vars

        acc_digit_single, acc_digit_all, acc_bboxes = list(), list(), list()
        loss_digits, loss_bboxes = list(), list()

        for i in range(SIZE):
            idx = np.random.choice(np.arange(len(labels)), self.BATCH_SIZE, replace=False)
            img_i = images[idx]
            lbl_l = labels[idx]
            box_b = boxes[idx]

            feed_dict = {X_: img_i, Y_: lbl_l, Z_: box_b, pkeep: 1.0, alpha: 1e-9,
                         is_train: False, iteration: i}

            digits, bboxes = session.run([digits_preds, bboxes_preds], feed_dict=feed_dict)

            acc_digit_single.append(accuracy_digits(digits, lbl_l)[0])
            acc_digit_all.append(accuracy_digits(digits, lbl_l)[1])
            acc_bboxes.append(accuracy_bboxes(bboxes, box_b))

        acc_single = sum(acc_digit_single) / len(acc_digit_single)
        acc_all = sum(acc_digit_all) / len(acc_digit_all)
        acc_box = sum(acc_bboxes) / len(acc_bboxes)

        if self.verbose:
            if TOKEN == 0:
                print('------------------------------------------')
                print('Epoch                      ==> ' + str(epoch + 1))
            else:
                print('========================================')
                print('      F I N A L      A C C U R A C Y    ')

            print('Accuracy digits - Individual : %.2f%%' % acc_single)
            print('Accuracy digits - All        : %.2f%%' % acc_all)
            print('Accuracy bboxes - All        : %.2f%%' % acc_box)
            if TOKEN == 0:
                print('------------------------------------------')
            else:
                print('========================================')
            print('      ')

        return([acc_single, acc_all, acc_box])
