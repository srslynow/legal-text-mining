from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import numpy as np

batch_size = 128
nb_classes = 7
nb_epoch = 12
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

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


model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=(100, 50, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
model.save("cnn_model.h5")
print('Test score:', score[0])
print('Test accuracy:', score[1])