import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from .MiniDenoisingNet import MiniDenoisingNet

class DeconvDenoisingNet(MiniDenoisingNet):

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
        self._batch_size = tf.placeholder(shape = [], dtype = tf.int32)

        # Create network:

        self._conv_module_1 = self.convolutional_module_with_max_pool(x = self._X_norm, inp_channel = 1, op_channel = 2, name = "module_1", strides = 1)

        self._res_1 = self.residual_module(self._conv_module_1, name = "res_1", inp_channel = 2)
        self._res_2 = self.residual_module(self._res_1, name = "res_2", inp_channel = 2)
        self._res_3 = self.residual_module(self._res_2, name = "res_3", inp_channel = 2)
        self._res_4 = self.residual_module(self._res_3, name = "res_4", inp_channel = 2)

        self._deconv_1 = self.deconvolutional_layer(
            self._res_4,
            inp_shape = [self._batch_size, 16, 16, 2],
            op_shape = [self._batch_size, 32, 32, 1],
            kernel_size = 3,
            strides = 2,
            padding = 'SAME',
            name = "deconv1")
        # self._deconv_2 = self.deconvolutional_layer(
        #     self._deconv_1,
        #     inp_shape = [self._batch_size, 32, 32, 2],
        #     op_shape = [self._batch_size, 64, 64, 1],
        #     kernel_size = 3,
        #     strides = 2,
        #     padding = 'SAME',
        #     name = "deconv2")
        self._X_reconstructed_batch_norm = tf.reshape(self._deconv_1, shape = [-1, self._w * self._h])

        self._op = tf.sigmoid(self._X_reconstructed_batch_norm)


