import tensorflow as tf
import nn_imgProcessing as ip
import numpy as np

# *** load the flatten image data and decimal number label ***
x, y = ip.load_image_join_path()
# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y, 2)  # label has 2 classes

# *** split the train and test data ***
x_train = x[:-200]
y_train = y[:-200]
x_test = x[-200:]
y_test = y[-200:]

sample_num = np.shape(x_train)[0]
epoch_num = 1
batch_size = 64
batch_num = int(sample_num / batch_size) + 1


img = tf.cast(x_train, tf.float32)
label = tf.cast(y_train, tf.float32)

input_queue = tf.train.slice_input_producer([img, label])
img_batch, label_batch = tf.train.batch(input_queue, batch_size=64)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for i in range(epoch_num):
            print('*********')
            for j in range(batch_num):
                print('-------------')
                x_batch, y_batch = sess.run([img_batch, label_batch])
                print('x:' + str(np.shape(x_batch)))
                print('y:' + str(np.shape(y_batch)))
    except tf.errors.OutOfRangeError:
        print('done')

    # request fot threads termination
    coord.request_stop()
    # wait for threads termination
    coord.join(threads)
