import imgProcessing as ip
import vgg16
import model

import numpy as np

# *** load the flatten image data and decimal number label ***
x, y = ip.load_image_join_path()

# *** transfer the label into class_number dimensions matrix ***
y = ip.expand(y, 2)  # label has 2 classes


# *** split the train and test data ***
x_train_org = x[:-200]
x_train = x_train_org / 255.
y_train = y[:-200]
x_test_org = x[-200:]
x_test = x_test_org / 255.
y_test = y[-200:]

print(np.shape(x_train))
print(np.shape(y_train))

# *** train the NN model ***
train = True
if train:
    model.train(x_train, y_train, x_test, y_test)
