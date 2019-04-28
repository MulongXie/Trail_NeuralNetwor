import cv2
import numpy as np
import os
from nn_model import CONFIG


def flower2num(name):
    num = 0
    if name == 'daisy':
        num = 0
    elif name == 'dandelion':
        num = 1
    elif name == 'roses':
        num = 2
    elif name == 'sunflowers':
        num = 3
    elif name == 'tulips':
        num = 4
    return num

def num2flower(num):
    flower = 0
    if num == 0:
        flower = 'daisy'
    elif num == 1:
        flower = 'dandelion'
    elif num == 2:
        flower = 'roses'
    elif num == 3:
        flower = 'sunflowers'
    elif num == 4:
        flower = 'tulips'
    return flower


# transfer int into c dimensions one-hot array
def expand(label, c):
    # return y : (num_class, num_samples)
    y = np.eye(c)[label]
    y = np.squeeze(y)
    return y


def img_read(img_name):
    try:
        img = cv2.imread(img_name + ".jpg")
        img = cv2.resize(img, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    except cv2.error:
        img = cv2.imread(img_name + ".png")
        img = cv2.resize(img, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
    return img


def load_image_flower(root_path):

    imgs = []
    imgs_label = []
    for label in os.listdir(root_path):
        label_num = flower2num(label)
        print("--- label: " + label + ' ' + str(label_num) + ' ---')
        dir_path = os.path.join(root_path, label)

        for _, _, files in os.walk(dir_path):
            for i, f in enumerate(files):
                img_path = os.path.join(dir_path, f)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))

                imgs.append(img)
                imgs_label.append(label_num)

    # shuffle the data
    np.random.seed(0)
    imgs = np.random.permutation(imgs)
    np.random.seed(0)
    imgs_label = np.random.permutation(imgs_label)

    print(np.shape(imgs))
    print(np.shape(imgs_label))

    return imgs, imgs_label


    # shuffle the data
    # np.random.seed(0)
    # label = np.random.permutation(label)
    # np.random.seed(0)
    # imgs = np.random.permutation(imgs)
    #
    # print(np.shape(imgs))
    # print(len(imgs))
    #
    # return imgs, label