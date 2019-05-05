import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

import nn_vgg16
import nn_imgProcessing as ip


class CONFIG:
    IMAGE_WIDTH = 112
    IMAGE_HEIGHT = 112

    ITER_NUM = 15
    BATCH_SIZE = 32

    MODEL_NAME = 'flower'


# x: (number_sample, img_hei, img_wid, img_channel)
# y: (number_sample, number_classes)
def train(x_train, y_train, x_test, y_test, save_path, iter_num=CONFIG.ITER_NUM, batch_size=CONFIG.BATCH_SIZE):
    tf.reset_default_graph()

    # *** step1 *** read and convert data
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    sample_num = x_shape[0]
    x_train_float = tf.cast(x_train, tf.float32)
    y_train_float = tf.cast(y_train, tf.float32)
    x_test_float = tf.cast(x_test, tf.float32)
    y_test_float = tf.cast(y_test, tf.float32)
    x = tf.placeholder(tf.float32, [None, x_shape[1], x_shape[2], x_shape[3]], name='input')
    y = tf.placeholder(tf.float32, [None, y_shape[1]], name='output')

    # *** step2 *** batch the data
    batch_num = int(sample_num / batch_size) + 1
    input_queue = tf.train.slice_input_producer([x_train_float, y_train_float])
    img_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)

    input_queue = tf.train.slice_input_producer([x_test_float, y_test_float])
    img_batch_test, label_batch_test = tf.train.batch(input_queue, batch_size=batch_size)

    # *** step3 *** transfer the vgg model
    vgg = nn_vgg16.VGG16(x)
    cost, model, y_hat = vgg.renew_layers(y)

    # *** step4 *** open session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # coordinator of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        try:
            # *** step 5 *** train
            loss = 0
            for i in range(iter_num):
                print('*** iteration num: %d/%d ***' %(i, iter_num))
                for j in range(batch_num):
                    print('--- batch num: %d/%d ---' %(j, batch_num))
                    # active the bach operation
                    x_batch, y_batch = sess.run([img_batch, label_batch])
                    x_batch_test, y_batch_test = sess.run([img_batch_test, label_batch_test])
                    _, loss, y_hat_re = sess.run([model, cost, y_hat], feed_dict={x: x_batch, y: y_batch})

                    # calculate the accuracy
                    correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                    # print the accuracy
                    if j == batch_num - 1:
                        print('--- train accuracy:', accuracy.eval({x: x_batch, y: y_batch}))
                        print('--- test accuracy:', accuracy.eval({x: x_batch_test, y: y_batch_test}))
                        print('\n')

                print('the ' + str(i) + 'th iteration')
                print('loss: %f' % loss)

        # done with reading queue
        except tf.errors.OutOfRangeError:
            print('done')

        # request threads termination
        finally:
            coord.request_stop()
        coord.join(threads)

        # *** step6 *** save the trained model
        saver = tf.train.Saver()
        saver.save(sess, save_path)


def predict(x, load_path):

    x = np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))

    with tf.Session() as sess:
        # load data and restore session
        saver = tf.train.import_meta_graph(load_path + '/' + CONFIG.MODEL_NAME + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_path))

        # fetch the trained end layers
        graph = tf.get_default_graph()
        x_tf = graph.get_tensor_by_name('input:0')
        y_hat = graph.get_tensor_by_name('y_hat:0')
        y_pre = tf.argmax(y_hat, 1)

        pre = sess.run(y_pre, feed_dict={x_tf: x})
        print(ip.num2flower(pre[0]))

    return y_pre


def evaluate(x_test, y_test, load_path):

    with tf.Session() as sess:
        # load data and restore session
        saver = tf.train.import_meta_graph(load_path + '/' + CONFIG.MODEL_NAME + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_path))

        # fetch the trained end layers
        graph = tf.get_default_graph()
        x_tf = graph.get_tensor_by_name('input:0')
        y_hat = graph.get_tensor_by_name('y_hat:0')
        y_pre = tf.argmax(y_hat, 1)

        y_predicts = []
        for x in x_test:
            x = np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
            pre = sess.run(y_pre, feed_dict={x_tf: x})

            print(ip.num2flower(pre[0]))

            y_predicts.append(pre[0])

        print(y_predicts)
        print(y_test)

        acc = accuracy_score(y_test, y_predicts)
        print('**** accuracy:%f ****' %acc)
        return acc
