import numpy as np
import tensorflow as tf


class CONFIG:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224


class VGG16:

    vgg = np.load('D:\\datasets\\VGG/vgg16.npy', encoding='latin1')
    vgg_layers = vgg.item()
    vgg_graph = {}
    transfer_graph = {}

    def __init__(self):
        # retrieve the layers before fc
        graph = {}
        graph['input'] = tf.Variable(np.zeros((CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 3)), dtype='float32')
        graph['conv1_1'] = self.conv2d_relu(graph['input'], 'conv1_1')
        graph['conv1_2'] = self.conv2d_relu(graph['conv1_1'], 'conv1_2')
        graph['maxpool1'] = self.maxpool(graph['conv1_2'])

        graph['conv2_1'] = self.conv2d_relu(graph['maxpool1'], 'conv2_1')
        graph['conv2_2'] = self.conv2d_relu(graph['conv2_1'], 'conv2_2')
        graph['maxpool2'] = self.maxpool(graph['conv2_2'])

        graph['conv3_1'] = self.conv2d_relu(graph['maxpool2'], 'conv3_1')
        graph['conv3_2'] = self.conv2d_relu(graph['conv3_1'], 'conv3_2')
        graph['conv3_3'] = self.conv2d_relu(graph['conv3_2'], 'conv3_3')
        graph['maxpool3'] = self.maxpool(graph['conv3_3'])

        graph['conv4_1'] = self.conv2d_relu(graph['maxpool3'], 'conv4_1')
        graph['conv4_2'] = self.conv2d_relu(graph['conv4_1'], 'conv4_2')
        graph['conv4_3'] = self.conv2d_relu(graph['conv4_2'], 'conv4_3')
        graph['maxpool4'] = self.maxpool(graph['conv4_3'])

        graph['conv5_1'] = self.conv2d_relu(graph['maxpool4'], 'conv5_1')
        graph['conv5_2'] = self.conv2d_relu(graph['conv5_1'], 'conv5_2')
        graph['conv5_3'] = self.conv2d_relu(graph['conv5_2'], 'conv5_3')
        graph['maxpool5'] = self.maxpool(graph['conv5_3'])

        self.vgg_graph = graph


    def conv2d_relu(self, pre_layer, layer):
        w = self.vgg_layers[layer][0]
        b = self.vgg_layers[layer][1]
        w = tf.constant(w)
        b = tf.constant(b)

        conv = tf.nn.conv2d(pre_layer, filter=w, strides=[1, 1, 1, 1], padding='SAME', name=layer) + b
        conv_relu = tf.nn.relu(conv)
        return conv_relu


    def maxpool(self, pre_layer):
        pool = tf.nn.max_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool


    def transfer(self, y_train):
        trans_graph = self.vgg_graph
        joint = self.vgg_graph['maxpool5']

        trans_graph['flatten'] = tf.reshape(joint, [-1, 7*7*512])
        trans_graph['fc6'] = tf.layers.dense(trans_graph['flatten'], 512, name='fc6')
        trans_graph['fc7'] = tf.layers.dense(trans_graph['fc6'], 2, name='fc7')

        self.transfer_graph = trans_graph

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=trans_graph['fc7'], labels=y_train))
        train = tf.train.AdamOptimizer().minimize(cost)

        return train

