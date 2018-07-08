import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class MiniDenoisingNet:

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
        self._X_norm = tf.layers.batch_normalization(self._X, training=self._is_training)
        self._batch_size = tf.shape(self._X_norm
                                    )[0]

        # Create network:

        self._conv_module_1 = self.convolutional_module_with_max_pool(x = self._X_norm, inp_channel = 1, op_channel = 2, name = "module_1", strides = 1)

        self._res_1 = self.residual_module(self._conv_module_1, name = "res_1", inp_channel = 2)
        self._res_2 = self.residual_module(self._res_1, name = "res_2", inp_channel = 2)
        self._res_3 = self.residual_module(self._res_2, name = "res_3", inp_channel = 2)
        self._res_4 = self.residual_module(self._res_3, name = "res_4", inp_channel = 2)


        self._re = tf.reshape(self._res_4, shape = [-1, 512])

        self._W1 = tf.get_variable(name = "W1", shape = [512, 768],
                                   initializer=tf.keras.initializers.he_normal())
        self._b1 = tf.get_variable(name = "b1", shape = [768])
        self._fc1 = tf.nn.relu(tf.matmul(self._re, self._W1) + self._b1)
        self._fc1_batch_norm = tf.layers.batch_normalization(self._fc1, training = self._is_training)

        self._W_decode = tf.get_variable(name = "W_decode", shape = [768, inp_w * inp_h],
                                 initializer=tf.keras.initializers.he_normal())
        self._b_decode = tf.get_variable(name = "b_decode", shape = [inp_w * inp_h])
        self._X_reconstructed = tf.matmul(self._fc1_batch_norm, self._W_decode) + self._b_decode
        self._X_reconstructed_batch_norm = tf.layers.batch_normalization(self._X_reconstructed, training = self._is_training, renorm = True)
        self.__X_reconstructed_dropout = tf.nn.dropout(self._X_reconstructed_batch_norm, keep_prob = self._keep_prob_tensor)
        self._op = tf.sigmoid(self._X_reconstructed_batch_norm)



    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_channel, op_channel, kernel_size=3, strides = 1, padding='VALID',
                            pad = 1, dropout=False, not_activated=False):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x
        W_conv = tf.get_variable("W_" + name, shape=[kernel_size, kernel_size, inp_channel, op_channel],
                                 initializer=tf.keras.initializers.he_normal())
        b_conv = tf.get_variable("b_" + name, initializer=tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides=[1, strides, strides, 1], padding=padding) + b_conv
        a_conv = tf.nn.relu(z_conv)
        h_conv = tf.layers.batch_normalization(a_conv, training = self._is_training, renorm = True)
        # h_conv = self.group_normalization(a_conv, name=name + "_norm", inp_channel=op_channel, G=32)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob=self._keep_prob)
            return a_conv_dropout
        if not_activated:
            return z_conv
        return h_conv

    def depthwise_separable_conv_layer(self, x, name, inp_channel, op_channel, depth_kernel,
                                       depth_multiplier = 1, strides = 1, pad = 0, padding = 'SAME'):
        if pad != 0:
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x

        depth_filter = tf.get_variable(
            name = name + "_depth_kernel",
            shape = [depth_kernel, depth_kernel, inp_channel, depth_multiplier]
        )
        point_filter = tf.get_variable(
            name = name + "point_filter",
            shape = [1, 1, inp_channel * depth_multiplier, op_channel]
        )
        separable_conv = tf.nn.separable_conv2d(
            input = x,
            depthwise_filter = depth_filter,
            pointwise_filter = point_filter,
            strides = [1, strides, strides, 1],
            padding = padding,
            name = name + "_separable_conv"
        )
        return separable_conv


    def deconvolutional_layer(self, x, name, inp_shape, op_shape, kernel_size = 3, strides = 1, padding='VALID'):
        b_deconv = tf.get_variable("b" + name, initializer=tf.zeros(op_shape[3]))
        filter = tf.get_variable("filter" + name, shape=[kernel_size, kernel_size, op_shape[3], inp_shape[3]])
        z_deconv = tf.nn.conv2d_transpose(x, filter=filter, strides=[1, strides, strides, 1], padding=padding,
                                          output_shape=tf.stack(op_shape)) + b_deconv
        # a_deconv = tf.nn.relu(z_deconv)
        # h_conv = tf.layers.batch_normalization(a_conv, axis = 1, training = self._is_training)
        return z_deconv


    def convolutional_module_with_max_pool(self, x, inp_channel, op_channel, name, strides = 1):
        # conv1 = self.convolutional_layer(x, inp_channel = inp_channel, op_channel = op_channel, name = name + "_conv1")
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1", strides = strides)
        conv2 = self.convolutional_layer(conv1, inp_channel=op_channel, op_channel=op_channel, name=name + "_conv2", strides = strides)
        conv2_max_pool = self.max_pool_2x2(conv2)

        return conv2_max_pool

    def convolution_module_with_more_max_pool(self, x, inp_channel, op_channel, name):
        conv1 = self.convolutional_layer(x, inp_channel=inp_channel, op_channel=op_channel, name=name + "_conv1")
        conv1_max_pool = self.max_pool_2x2(conv1)
        conv2 = self.convolutional_layer(conv1_max_pool, inp_channel=op_channel, op_channel=op_channel,
                                         name=name + "_conv2")
        conv2_max_pool = self.max_pool_2x2(conv2)

        return conv2_max_pool

    def residual_module(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "_conv2", inp_channel, inp_channel, not_activated=True)
        # conv3 = self.convolutional_layer(conv2, name + "conv3", inp_channel, op_channel, dropout = True)
        res_layer = tf.nn.relu(tf.add(conv2, x, name="res"))

        batch_norm = tf.contrib.layers.batch_norm(res_layer, is_training=self._is_training, renorm = True)

        return batch_norm

    def inception_module(self, x, name, inp_channel, op_channel):
        tower1_conv1 = self.convolutional_layer(x, kernel_size=1, padding='SAME', inp_channel=inp_channel,
                                                op_channel=op_channel // 3, name=name + "_tower1_conv1", pad=0)
        tower1_conv2 = self.convolutional_layer(tower1_conv1, kernel_size=3, padding='SAME',
                                                inp_channel=op_channel // 3, op_channel=op_channel // 3,
                                                name=name + "_tower1_conv2", pad=0)

        tower2_conv1 = self.convolutional_layer(x, kernel_size=1, padding='SAME', inp_channel=inp_channel,
                                                op_channel=op_channel // 3, name=name + "_tower2_conv1", pad=0)
        tower2_conv2 = self.convolutional_layer(tower2_conv1, kernel_size=5, padding='SAME',
                                                inp_channel=op_channel // 3, op_channel=op_channel // 3,
                                                name=name + "_tower2_conv2", pad=0)

        tower3_max_pool = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        tower3_conv = self.convolutional_layer(tower3_max_pool, name=name + "_tower3_conv", inp_channel=inp_channel,
                                               op_channel=op_channel // 3, kernel_size=1, pad=0)

        return tf.concat([tower1_conv2, tower2_conv2, tower3_conv], axis=-1)

    def xception_module(self, x, name, inp_channel):
        # x_activated = tf.nn.relu(x)
        sep_conv_1 = self.depthwise_separable_conv_layer(
            x,
            name=name + "sep_conv_1",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training = self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training = self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training = self._is_training, renorm = True)

        res = tf.nn.relu(x + sep_conv_3_norm)
        return tf.layers.batch_normalization(res, training = self._is_training)

    def squeeze(self, x):
        return self.global_average_pooling(x)

    def excite(self, x, name, n_channels, reduction_ratio = 16):
        x_shape = tf.shape(x)
        W_1 = tf.get_variable(shape = [n_channels, n_channels // reduction_ratio], name = name + "_W1")
        z_1 = tf.nn.relu(tf.matmul(x, W_1))
        W_2 = tf.get_variable(shape = [n_channels // reduction_ratio, n_channels], name = name + "_W2")
        return tf.nn.sigmoid(tf.matmul(z_1, W_2))


    def se_block(self, x, name, n_channels):
        x_shape = tf.shape(x)
        x_squeezed = self.squeeze(x)
        x_excited = self.excite(x_squeezed, name = name + "_excited", n_channels = n_channels)
        x_excited_broadcasted = tf.reshape(x_excited, shape = [x_shape[0], 1, 1, x_shape[-1]])
        return tf.multiply(x, x_excited_broadcasted)

    def residual_module_with_se(self, x, name, inp_channel):
        conv1 = self.convolutional_layer(x, name + "_conv1", inp_channel, inp_channel)
        conv2 = self.convolutional_layer(conv1, name + "_conv2", inp_channel, inp_channel, not_activated=True)
        conv2_se = self.se_block(conv2, name = name + "_se", n_channels = inp_channel)
        res_layer = tf.nn.relu(tf.add(conv2_se, x, name = name + "res"))
        batch_norm = tf.layers.batch_normalization(res_layer, training = self._is_training, renorm = True)
        return batch_norm

    def xception_module_with_se(self, x, name, inp_channel):
        # x_activated = tf.nn.relu(x)
        sep_conv_1 = self.depthwise_separable_conv_layer(
            x,
            name=name + "sep_conv_1",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_1_norm = tf.layers.batch_normalization(sep_conv_1, training = self._is_training)
        sep_conv_1_norm_activated = tf.nn.relu(sep_conv_1_norm)

        sep_conv_2 = self.depthwise_separable_conv_layer(
            sep_conv_1_norm_activated,
            name=name + "sep_conv_2",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_2_norm = tf.layers.batch_normalization(sep_conv_2, training = self._is_training)
        sep_conv_2_norm_activated = tf.nn.relu(sep_conv_2_norm)

        sep_conv_3 = self.depthwise_separable_conv_layer(
            sep_conv_2_norm_activated,
            name=name + "sep_conv_3",
            depth_kernel=3,
            inp_channel=inp_channel,
            op_channel = inp_channel
        )
        sep_conv_3_norm = tf.layers.batch_normalization(sep_conv_3, training = self._is_training)
        sep_conv_3_norm_se = self.se_block(sep_conv_3_norm, name = name + "_se", n_channels = inp_channel)

        res = tf.nn.relu(x + sep_conv_3_norm_se)
        batch_norm = tf.layers.batch_normalization(res, training = self._is_training, renorm = True)
        return batch_norm


    def hypercolumn(self, layers_list, input_dim):
        layers_list_upsampled = []
        for layer in layers_list:
            layers_list_upsampled.append(tf.image.resize_bilinear(images=layer, size=(input_dim, input_dim)))
        return tf.concat(layers_list, axis=0)

    def feed_forward(self, x, name, inp_channel, op_channel, op_layer=False):
        W = tf.get_variable("W_" + name, shape=[inp_channel, op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape=[op_channel], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return z
        else:
            a = tf.nn.relu(z)
            a_norm = tf.layers.batch_normalization(a, training=self._is_training)
            return a_norm

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, axis=[1, 2])

    # Predict:
    def predict(self, X):
        ans = self._sess.run(self._op,
                             feed_dict={self._X: X, self._is_training: False, self._keep_prob_tensor: 1.0})
        return ans


    # Train:
    def fit(self, X, y, num_epoch = 1, batch_size = 16, weight_save_path=None, weight_load_path=None,
            plot_losses=False, print_every = 1):
        self._y = tf.placeholder(tf.float32, shape=[None, self._w * self._h])
        # self._mean_loss = tf.reduce_mean(tf.square(self._y - self._op))
        # self._mean_loss = -tf.reduce_mean(self._y * tf.log(self._op) + (1 - self._y) * tf.log(1 - self._op))
        # self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self.__X_reconstructed_dropout))
        self._mean_loss = tf.losses.mean_squared_error(labels = self._y, predictions = self._op)
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
            print('Training Denoising Net for ' + str(num_epoch) + ' epochs')
            self.run_model(self._sess, self._op, self._mean_loss, X, y, num_epoch, batch_size, print_every,
                           self._train_step, weight_save_path=weight_save_path, plot_losses=plot_losses)

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
            variables.append(self.psnr(self._mean_loss))
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
                    loss, corr, _, psnr = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                              .format(iter_cnt, loss, np.sum(corr) / actual_batch_size / self._h / self._w))
                        print("PSNR: " + str(psnr))


                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: False,
                                 self._keep_prob_tensor: 1.0}
                    psnr_tensor = self.psnr(self._mean_loss)
                    val_loss, psnr = session.run([self._mean_loss, psnr_tensor], feed_dict=feed_dict)
                    print("Validation loss: " + str(val_loss))
                    print("Validation PSNR: " + str(psnr))
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
            total_correct = correct / (Xd.shape[0] - actual_batch_size)
            total_loss = np.sum(losses) / (Xd.shape[0] - actual_batch_size)
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct / self._h / self._w, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


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

    def psnr(self, mse):
        return 10 * self.log10(tf.constant(1, dtype = tf.float32) / mse)

    def log10(self, x):
        return tf.log(x) / tf.log(tf.constant(10, dtype = x.dtype))

    def evaluate(self, X, y):
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)








