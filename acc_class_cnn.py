from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np

X_train = np.load("train_data.npy")
y_train = np.load("train_label.npy")
X_test = np.load("test_data.npy")
y_test = np.load("test_label.npy")

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

samples = X_train.shape[0]
X_train = np.reshape(X_train, (samples, 100, 50))
samples = X_test.shape[0]
X_test = np.reshape(X_test, (samples, 100, 50))

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = load_model('model_cnn.h5')
y_out = model.predict(X_test)
y_out[y_out < 0.5] = 0
y_out[y_out >= 0.5] = 1
corr = y_out * y_test
gt = np.sum(y_test, axis=0)
pred = np.sum(corr, axis=0)
perc = np.divide(pred, gt) * 100
print("Ground truth", gt)
print("Predicted correct", pred)
print("Percentage correct", perc)