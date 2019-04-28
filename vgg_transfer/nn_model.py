import nn_vgg16
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


# x: (number_sample, img_hei, img_wid, img_channel)
# y: (number_sample, number_classes)
def train(x_train, y_train, x_test, y_test, iter_num=10, batch_size=32, save_path='D:\\datasets\\Trail_NN\\trained_model/house'):
    tf.reset_default_graph()

    # *** step1 *** read and convert data
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    sample_num = x_shape[0]
    x_train_float = tf.cast(x_train, tf.float32)
    y_train_float = tf.cast(y_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    x = tf.placeholder(tf.float32, [None, x_shape[1], x_shape[2], x_shape[3]], name='input')
    y = tf.placeholder(tf.float32, [None, y_shape[1]], name='output')

    # *** step2 *** batch the data
    batch_num = int(sample_num / batch_size) + 1
    input_queue = tf.train.slice_input_producer([x_train_float, y_train_float])
    img_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)

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
            loss = 0
            for i in range(iter_num):
                print('*** iteration num: %d/%d ***' %(i, iter_num))
                for j in range(batch_num):
                    print('--- batch num: %d/%d ---' %(j, batch_num))
                    x_batch, y_batch = sess.run([img_batch, label_batch])
                    _, loss, y_hat_re = sess.run([model, cost, y_hat], feed_dict={x: x_batch, y: y_batch})

                print('the ' + str(i) + 'th iteration')
                print(loss)

                # correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
                # # convert to float and calculate the mean correct rate
                # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                #
                # print('train accuracy:', accuracy.eval({x: x_train, y: y_train}))
                # print('test accuracy:', accuracy.eval({x: x_test, y: y_test}))

        # done with reading queue
        except tf.errors.OutOfRangeError:
            print('done')

        # request threads termination
        finally:
            coord.request_stop()
        coord.join(threads)

        saver = tf.train.Saver()
        saver.save(sess, save_path)


def predict(x, load_path='D:\\datasets\\Trail_NN\\trained_model'):

    x = np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '/house.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_path))

        graph = tf.get_default_graph()
        x_tf = graph.get_tensor_by_name('input:0')
        y_hat = graph.get_tensor_by_name('y_hat:0')
        y_pre = tf.argmax(y_hat, 1)

        pre = sess.run(y_pre, feed_dict={x_tf: x})
        print(pre)
        if pre[0] == 1:
            print("House")
        else:
            print("Not Building")

    return y_pre


def evaluate(x_test, y_test, load_path='D:\\datasets\\Trail_NN\\trained_model'):

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '/house.meta')
        saver.restore(sess, tf.train.latest_checkpoint(load_path))

        graph = tf.get_default_graph()
        x_tf = graph.get_tensor_by_name('input:0')
        y_hat = graph.get_tensor_by_name('y_hat:0')
        y_pre = tf.argmax(y_hat, 1)

        y_predicts = []
        for x in x_test:
            x = np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
            pre = sess.run(y_pre, feed_dict={x_tf: x})

            if pre[0] == 1:
                print("House")
            else:
                print("Not Building")

            y_predicts.append(pre[0])

        print(y_predicts)
        print(y_test)

        acc = accuracy_score(y_test, y_predicts)
        print('**** accuracy:%f ****' %acc)
        return acc
