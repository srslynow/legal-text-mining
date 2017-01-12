## Synopsis

Experiments classifying legal documents into their sub-categories: e.g. civil law, criminal law or administrative law.
Classifiers used:
* k-Nearest Neighbours
* linear Support Vector Machine
* Random Forest
* Convolutional Neural Network
* Long Short Term Memory Neural Network

## Usage
Convert raw XML (from rechtspraak.nl) to plain text using `xmlToPlain.py`
Make bag of words data file using `to_bow_data.py`
Make int vector model using `to_vector_data.py`
Bag of words classifiers can then be executed using `bow_*.py`, the LSTM using `word2vec_lstm.py`