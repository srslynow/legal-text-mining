from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
import numpy as np

max_features = 5000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 512

# Dataset loading
print("Loading databases")
X_train = np.load("train_data_vec.npy")
y_train = np.load("train_label_vec.npy")
X_test = np.load("test_data_vec.npy")
y_test = np.load("test_label_vec.npy")

# Data preprocessing
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
model.add(Dense(7))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)
model.save("lstm_model.h5")
print('Test score:', score)
print('Test accuracy:', acc)