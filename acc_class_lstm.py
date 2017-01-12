from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
import numpy as np

max_features = 5000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
batch_size = 512

# Dataset loading
print("Loading databases")
X_test = np.load("test_data_vec.npy")
y_test = np.load("test_label_vec.npy")

# Data preprocessing
print('Pad sequences (samples x time)')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_test shape:', X_test.shape)

model = load_model('model_lstm.h5')
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