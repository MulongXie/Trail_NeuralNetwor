import nn_imgProcessing as ip
import nn_model

import numpy as np
import cv2

root = 'D:\\datasets\\Trail_NN\\flower_pic'

# *** load the flatten image data and decimal number label ***
x, y_org = ip.load_image_flower(root)

# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y_org, 5)  # label has 5 classes

# *** split the train and test data ***
split = 700
x_train = x[:-split]
y_train = y[:-split]
x_test = x[-split:]
y_test = y[-split:]
y_org_test = y_org[-split:]

print('*** training sample shape: ' + str(np.shape(x_train)) + '***')

# *** train the NN model ***
train = True
if train:
    nn_model.train(x_train, y_train, x_test, y_test, save_path='D:\\datasets\\Trail_NN\\trained_model_flower\\flower')

# *** test and predict ***
predict = False
if predict:
    test_index = 165
    y_hat = nn_model.predict(x_test[test_index], load_path='D:\\datasets\\Trail_NN\\trained_model_flower')
    cv2.imshow('img', cv2.resize(x_test[test_index], (200, 200)))
    cv2.waitKey(0)

# *** evaluate the test accuracy ***
eval = False
if eval:
    nn_model.evaluate(x_test, y_org_test.tolist(), load_path='D:\\datasets\\Trail_NN\\trained_model_flower')