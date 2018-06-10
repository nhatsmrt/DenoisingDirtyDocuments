import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class LinearRegressor:

    def __init__(self, inp_w = 32, inp_h = 32, keep_prob = 0.8, threshold = 0.5, use_gpu = False):
        self._h = inp_h
        self._w = inp_w
        self._threshold = threshold

        self._X = tf.placeholder(dtype = tf.float32, shape = [None, inp_w, inp_h, 1])

        self._keep_prob = keep_prob
        self._use_gpu = use_gpu

        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.create_network(inp_w, inp_h)
        else:
            with tf.device('/device:CPU:0'):
                self.create_network(inp_w, inp_h)



    def create_network(self, inp_w, inp_h):
        self._is_training = tf.placeholder(tf.bool)
        self._keep_prob_tensor = tf.placeholder(tf.float32)

        # Create network:

        self._X_flat = tf.reshape(self._X,  shape = [-1, inp_w * inp_h])
        self._W = tf.get_variable(name = "W", shape = [inp_w * inp_h, inp_w * inp_h],
                                 initializer=tf.keras.initializers.he_normal())
        self._b = tf.get_variable(name = "b", shape = [inp_w * inp_h])
        self._op = tf.matmul(self._X_flat, self._W) + self._b


    # Adapt from Stanford's CS231n Assignment3
    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path=None, patience=None):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.to_int32(self._op > self._threshold), tf.to_int32(self._y > self._threshold))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define saver:
        saver = tf.train.Saver()

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                if i < int(math.ceil(Xd.shape[0] / batch_size)) - 1:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: training_now,
                                 self._keep_prob_tensor: self._keep_prob_passed}
                    # have tensorflow compute loss and correct predictions
                    # and (if given) perform a training step
                    loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        # print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                        #       .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                        print("Iteration {0}: with minibatch training loss = {1:.3g}" \
                              .format(iter_cnt, loss))


                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: False,
                                 self._keep_prob_tensor: 1.0}
                    val_loss = session.run(self._mean_loss, feed_dict=feed_dict)
                    print("Validation loss: " + str(val_loss))
                    val_losses.append(val_loss)
                    # if training_now and weight_save_path is not None:
                    if training_now and val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = saver.save(session, save_path=weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {1}, Overall loss = {0:.3g}" \
                  .format(total_loss, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


    # Predict:
    def predict(self, X):
        ans = self._sess.run(self._op,
                             feed_dict={self._X: X, self._is_training: False, self._keep_prob_tensor: 1.0})
        return ans


    # Train:
    def fit(self, X, y, num_epoch = 1, batch_size = 16, weight_save_path=None, weight_load_path=None,
            plot_losses=False):
        self._y = tf.placeholder(tf.float32, shape=[None, self._w * self._h])
        self._mean_loss = tf.reduce_mean(tf.square(self._y - self._op))
        # self._mean_loss = -tf.reduce_mean(self._y * tf.log(self._op) + (1 - self._y) * tf.log(1 - self._op))
        # self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self.__X_reconstructed_dropout))
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess=self._sess, save_path=weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
        if num_epoch > 0:
            print('Training Linear Regressor for ' + str(num_epoch) + ' epochs')
            self.run_model(self._sess, self._op, self._mean_loss, X, y, num_epoch, batch_size, 1,
                           self._train_step, weight_save_path=weight_save_path, plot_losses=plot_losses)

    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n - 2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)

    def save_weights(self, weight_save_path):
        saver = tf.train.Saver()
        saver.save(sess=self._sess, save_path=weight_save_path)
        print("Weight saved successfully")

    def evaluate(self, X, y):
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)








