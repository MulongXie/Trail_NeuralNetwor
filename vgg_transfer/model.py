import vgg16
import tensorflow as tf
import numpy as np


# x: (number_sample, img_hei, img_wid, img_channel)
# y: (number_sample, number_classes)
def train(x_train, y_train, x_test, y_test, iter_num=100, save_path='trained_model/house'):
    tf.reset_default_graph()

    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    x = tf.placeholder(tf.float32, [None, x_shape[1], x_shape[2], x_shape[3]])
    y = tf.placeholder(tf.float32, [None, y_shape[1]])

    vgg = vgg16.VGG16(x)
    cost, model, y_hat = vgg.renew_layers(y)

    init = tf.global_variables_initializer()

    # open session
    with tf.Session() as sess:
        sess.run(init)

        for i in range(iter_num):
            _, loss = sess.run([model, cost], feed_dict={x: x_train, y: y_train})

            if i % 20 == 0:
                print('the ' + str(i) + 'th iteration')
                print(loss)

                correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
                # convert to float and calculate the mean correct rate
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                print('train accuracy:', accuracy.eval({x: x_train, y: y_train}))
                print('test accuracy:', accuracy.eval({x: x_test, y: y_test}))

        saver = tf.train.Saver()
        saver.save(sess, save_path)
