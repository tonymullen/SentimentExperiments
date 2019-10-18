#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:38:45 2019

@author: tonymullen
"""

import numpy as np
import shared

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from nlpia.loaders import get_data

np.random.seed(1337)


word_vectors = get_data('w2v', limit=200000)
data_file_root = '/Users/tonymullen/Dropbox/Northeastern/Classes/NLP/Datasets'
# https://ai.stanford.edu/~amaas/data/sentiment/

number_of_files = 5000

dataset = shared.pre_process_data(data_file_root + '/aclimdb/train',
                                  number_of_files)
# dataset = shared.pre_process_data(data_file_root + '/miniImdb/train')

vectorized_data = shared.tokenize_and_vectorize(dataset, word_vectors)
expected = shared.collect_expected(dataset)
split_point = int(len(vectorized_data) * .8)


x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]


maxlen = 400  # was 400 for reviews
batch_size = 32
embedding_dims = 300  # was 300
filters = 128  # was 250
kernel_size = 3
hidden_dims = 128  # was 250
epochs = 5


x_train = shared.pad_trunc(x_train, maxlen)
x_test = shared.pad_trunc(x_test, maxlen)

# x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)

print("Build model...")
model = Sequential()

model.add(Conv1D(
        filters,
        kernel_size,
        padding='valid',
        activation='relu',
        strides=1,
        input_shape=(maxlen, embedding_dims)))

model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
# model.add(Dense(num_classes) for categorical case
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",  # cf categorical_crossentropy
              optimizer="adam",
              metrics=["accuracy"]
              )

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

model_structure = model.to_json()
with open("cnn_model.json", "w") as json_file:
    json_file.write(model_structure)
model.save_weights("cnn_Weights.h5")


