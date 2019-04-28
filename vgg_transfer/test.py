import nn_imgProcessing as ip
import nn_model

import numpy as np
import cv2

# *** load the flatten image data and decimal number label ***
x, y_org = ip.load_image_join_path()

# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y_org, 2)  # label has 2 classes

# *** split the train and test data ***
x_train = x[:-200]
y_train = y[:-200]
x_test = x[-200:]
y_test = y[-200:]
y_org_test = y_org[-200:]

print(y_org_test)
print(y_org_test.tolist())