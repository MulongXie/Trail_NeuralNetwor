import tensorflow as tf
import cv2
import numpy as np

import nn_vgg16 as vgg


class RPN:
    def __init__(self):
        self.graph = {}

    def rpn(self, base_net):
        joint_layer = base_net['maxpool5']

        self.graph['convRPN_1'] = None
    #
    # def conv2d_relu(self, pre_layer, layer):
    #     return tf.nn.relu(tf.nn.conv2d(pre_layer, filter=, strides=[1, 1, 1, 1], padding='SAME', name=layer))
    #
    # def maxpool(self, pre_layer):
    #     pool = tf.nn.max_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #     return pool