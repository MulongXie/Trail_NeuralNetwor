import tensorflow as tf
import imgProcessing as ip
import numpy as np

# *** load the flatten image data and decimal number label ***
x, y = ip.load_image_join_path()
# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y, 2)  # label has 2 classes

sample_num = np.shape(x)[0]
epoch_num = 1
batch_size = 64
batch_num = int(sample_num / batch_size) + 1


img = tf.cast(x, tf.float32)
label = tf.cast(y, tf.int32)

input_queue = tf.train.slice_input_producer([img, label])
img_batch, label_batch = tf.train.batch(input_queue, batch_size=64)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(epoch_num):
        print('*********')
        for j in range(batch_num):
            print('-------------')
            x_batch, y_batch = sess.run([img_batch, label_batch])
            print('x:' + str(np.shape(x_batch)))
            print('y:' + str(np.shape(y_batch)))

    # request fot threads termination
    coord.request_stop()
    # wait for threads termination
    coord.join(threads)
