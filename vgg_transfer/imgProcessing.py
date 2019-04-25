import cv2
import numpy as np
from vgg16 import CONFIG


# transfer int into c dimensions one-hot array
def expand(label, c=10):
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


def load_image_join_path(label='D:/datasets/dataset_house/train.txt', root_path='D:/datasets/dataset_house/image/'):
    file = open(label, 'r')
    imgs = []
    label = []
    for line_num, line in enumerate(file):
        if line[:len(line) - 1] == '':
            print('****************end*******************')
            break
        if line_num % 2 == 0:
            line = root_path + line[:-1]
            img = img_read(line)
            imgs.append(img)
        else:
            line = int(line[:-1])
            label.append(line)
        print(line)
    label.append(int(line))

    # shuffle the data
    np.random.seed(0)
    label = np.random.permutation(label)
    np.random.seed(0)
    imgs = np.random.permutation(imgs)

    print(np.shape(imgs))
    print(len(imgs))

    return imgs, label