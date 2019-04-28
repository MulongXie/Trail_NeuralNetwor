import imgProcessing as ip
import vgg16
import model

import numpy as np
import cv2

# *** load the flatten image data and decimal number label ***
x, y = ip.load_image_join_path()

# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y, 2)  # label has 2 classes


# *** split the train and test data ***
x_train = x[:-200]
y_train = y[:-200]
x_test = x[-200:]
y_test = y[-200:]

print(np.shape(x_train))
print(np.shape(y_train))

# *** train the NN model ***
train = False
if train:
    model.train(x_train, y_train, x_test, y_test)

# *** test and predict ***
predict = False
if predict:
    test_index = 165
    y_hat = model.predict(x_test[test_index])
    cv2.imshow('img', cv2.resize(x_test[test_index], (200, 200)))
    cv2.waitKey(0)


eval = True
if eval:
    model.evaluate(x_test, y_test)