from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# NOTE: Padding is required for dimension consistency. This will pad sequences
# with 0 at the end, until it reaches the max sequence length. 0 is used as a
# masking value by dynamic RNNs in TFLearn; a sequence length will be
# retrieved by counting non zero elements in a sequence. Then dynamic RNN step
# computation is performed according to that length.
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
# Masking is not required for embedding, sequence length is computed prior to
# the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
net = tflearn.fully_connected(net, 2, activation='sigmoid')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='binary_crossentropy')

# Training
model = tflearn.DNN(net)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
batch_size=32)