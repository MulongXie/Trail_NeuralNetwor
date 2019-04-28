import nn_imgProcessing as ip
import nn_model

import numpy as np
import cv2

# *** load the flatten image data and decimal number label ***
x, y_org = ip.load_image_join_path(label='D:/datasets/dataset_house/train.txt', root_path='D:/datasets/dataset_house/image/')

# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y_org, 2)  # label has 2 classes

# *** split the train and test data ***
x_train = x[:-200]
y_train = y[:-200]
x_test = x[-200:]
y_test = y[-200:]
y_org_test = y_org[-200:]

print(np.shape(x_train))
print(np.shape(y_train))

# *** train the NN model ***
train = False
if train:
    nn_model.train(x_train, y_train, x_test, y_test, save_path='D:\\datasets\\Trail_NN\\trained_model/house')

# *** test and predict ***
predict = False
if predict:
    test_index = 165
    y_hat = nn_model.predict(x_test[test_index], load_path='D:\\datasets\\Trail_NN\\trained_model')
    cv2.imshow('img', cv2.resize(x_test[test_index], (200, 200)))
    cv2.waitKey(0)

# *** evaluate the model accuracy ***
eval = True
if eval:
    nn_model.evaluate(x_test, y_org_test.tolist(), load_path='D:\\datasets\\Trail_NN\\trained_model')