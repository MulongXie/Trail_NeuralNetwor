import vgg16
import tensorflow as tf
import numpy as np


# x: (number_sample, img_hei, img_wid, img_channel)
# y: (number_sample, number_classes)
def train(x_train, y_train, x_test, y_test, iter_num=10, batch_size=32, save_path='trained_model/house'):
    tf.reset_default_graph()

    # *** step1 *** read and convert data
    x_shape = np.shape(x_train)
    y_shape = np.shape(y_train)
    sample_num = x_shape[0]
    x_train_float = tf.cast(x_train, tf.float32)
    y_train_float = tf.cast(y_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    x = tf.placeholder(tf.float32, [None, x_shape[1], x_shape[2], x_shape[3]])
    y = tf.placeholder(tf.float32, [None, y_shape[1]])

    # *** step2 *** batch the data
    batch_num = int(sample_num / batch_size) + 1
    input_queue = tf.train.slice_input_producer([x_train_float, y_train_float])
    img_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size)

    # *** step3 *** transfer the vgg model
    vgg = vgg16.VGG16(x)
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
                print(y_hat_re[:4, :])
                print(y_batch[:4, :])

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

