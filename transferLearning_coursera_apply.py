import tensorflow as tf
import cv2
import numpy as np
import nst_utils as nu

def content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(tf.transpose(a_C), [n_C, n_H * n_W])
    a_G_unrolled = tf.reshape(tf.transpose(a_G), [n_C, n_H * n_W])
    J_content = (1 / (4 * n_H * n_C * n_W)) * tf.reduce_sum((tf.square((tf.subtract(a_C_unrolled, a_G_unrolled)))))

    return J_content


def gram(A):
    ga = tf.matmul(A, tf.transpose(A))
    return ga

def style_layer_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_W * n_H])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_W * n_H])
    GS = gram(a_S)
    GG = gram(a_G)
    J_style_layer = (1 / (2 * n_C * n_H * n_W)) ** 2 * tf.reduce_sum((tf.square((tf.subtract(GS, GG)))))
    return J_style_layer

def style_cost(model, STYLE_LAYERS, sess):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_s_layer = style_layer_cost(a_S, a_G)
        J_style += coeff * J_s_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=30):
    J = alpha * J_content + beta * J_style
    return J


def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (400, 300))
    return img

def NST_model(num_iter=1000):
    content_img = load_img("con_niu.jpg")
    content_img = nu.reshape_and_normalize_image(content_img)
    style_img = load_img("style_cloud.jpg")
    style_img = nu.reshape_and_normalize_image(style_img)
    generated_img = nu.generate_noise_image(content_img)

    print(np.shape(content_img))
    print(np.shape(style_img))
    print(np.shape(generated_img))

    with tf.Session() as sess:
        model = nu.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
        sess.run(model['input'].assign(content_img))
        out = model['conv4_2']

        a_c = sess.run(out)
        a_g = out
        J_content = content_cost(a_c, a_g)

        STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        sess.run(model['input'].assign(style_img))
        J_style = style_cost(model, STYLE_LAYERS, sess)
        J = total_cost(J_content, J_style)

        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(J)

        tf.global_variables_initializer().run()
        sess.run(model['input'].assign(generated_img))

        for i in range(num_iter):
            sess.run(train_step)
            generated_img = sess.run(model['input'])
            if i % 20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                print(generated_img.shape)
                nu.save_image("output4/" + str(i) + ".png", generated_img)

        nu.save_image("output4/generated_image.png", generated_img)

    return generated_img

NST_model()