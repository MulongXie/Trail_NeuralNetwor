import numpy as np
import tensorflow as tf


class CONFIG:
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64


class VGG16:

    vgg = np.load('D:\\datasets\\VGG/vgg16.npy', encoding='latin1')
    vgg_layers = vgg.item()
    vgg_graph = {}
    renew_graph = {}

    def __init__(self, x):
        # retrieve the layers before fc
        graph = {}
        graph['input'] = x
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
        self.renew_graph = graph

        print("*************** VGG initialized *************")

    def conv2d_relu(self, pre_layer, layer):
        w = self.vgg_layers[layer][0]
        b = self.vgg_layers[layer][1]
        w = tf.constant(w)
        b = tf.constant(b)
        print(np.shape(w))
        return tf.nn.relu(tf.nn.conv2d(pre_layer, filter=w, strides=[1, 1, 1, 1], padding='SAME', name=layer) + b)

    def maxpool(self, pre_layer):
        pool = tf.nn.max_pool(pre_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return pool

    def renew_layers(self, y_train):
        self.renew_graph['flatten'] = tf.contrib.layers.flatten(self.renew_graph['maxpool5'])
        self.renew_graph['fc6'] = tf.contrib.layers.fully_connected(self.renew_graph['flatten'], 1024)
        self.renew_graph['fc7'] = tf.contrib.layers.fully_connected(self.renew_graph['fc6'], 2, activation_fn=None)
        y_hat = tf.add(self.renew_graph['fc7'], 0, name='y_hat')  # the intermediate value used to calculate the accuracy

        print('y_hat type ' + str(type(y_hat)))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y_train))
        model = tf.train.AdamOptimizer(0.001).minimize(cost)

        return cost, model, y_hat

