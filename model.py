#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:38:45 2019

@author: tonymullen
"""

import numpy as np

from random import shuffle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D

from nltk.tokenize import TreebankWordTokenizer
from nltk import sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data

import glob
import os

np.random.seed(1337)

word_vectors = get_data('w2v', limit=200000)
data_file_root = '/Users/tonymullen/Dropbox/Northeastern/Classes/NLP/Datasets'
# https://ai.stanford.edu/~amaas/data/sentiment/

number_of_files = 5000

def pre_process_data(filepath):
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'neg')
    pos_label = 1
    neg_label = 0
    dataset = []
    c = 0
    for filename in glob.glob(os.path.join(positive_path, '*.txt')):
        if c < number_of_files:
            with open(filename, 'r') as f:
                dataset.append((pos_label, f.read()))
            c += 1
    c = 0
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        if c < number_of_files:
            with open(filename, 'r') as f:
                dataset.append((neg_label, f.read()))
            c += 1

    shuffle(dataset)
    return(dataset)


dataset = pre_process_data(data_file_root + '/aclimdb/train')
# dataset = pre_process_data(data_file_root + '/miniImdb/train')


def sentences_split(dataset):
    labeled_sents = []
    for label_txt in dataset:
        for s in sent_tokenize(label_txt[1]):
            labeled_sents.append((label_txt[0], s))
    return labeled_sents


sentence_dataset = sentences_split(dataset)



def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass # no matching token in Google w2v vocab
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


vectorized_data = tokenize_and_vectorize(dataset)
vectorized_data_s =  tokenize_and_vectorize(sentence_dataset)

expected = collect_expected(dataset)
expected_s = collect_expected(sentence_dataset)

split_point = int(len(vectorized_data) * .8)
split_point_s = int(len(vectorized_data_s) * .8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
# x_test = vectorized_data[split_point:]
# y_test = expected[split_point:]

# x_train = vectorized_data_s[:split_point_s]
# y_train = expected_s[:split_point_s]
x_test = vectorized_data_s[split_point_s:]
y_test = expected_s[split_point_s:]



maxlen = 400 # was 400 for reviews
batch_size = 32
embedding_dims = 300 # was 300
filters = 128 # was 250
kernel_size = 3
hidden_dims = 128 # was 250
epochs = 5


def pad_trunc(data, maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)

    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data


x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

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

sent_results = model.predict_classes(x_test)
x_test_sents = sentence_dataset[split_point_s:]
for i in range(len(sent_results)):
    if (sent_results[i][0] == 1 and x_test_sents[i][0] == 1):
        print("Guessed",sent_results[i][0], ":>", x_test_sents[i])


### Load it back up

from keras.models import model_from_json
with open("cnn_model.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights("cnn_weights.h5")


sample_1 = "I hate that the dismal weather had me down for so long, when will it break? Ugh, when does happiness return? The sun is blinding and the puffy clouds are too thin. I can't wait for the weekend."

vec_list = tokenize_and_vectorize([(1, sample_1)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict(test_vec)
model.predict_classes(test_vec)

sample_2 = "In spite of the dark themes, this was a deeply moving movie, full of toughness in the face of adversity and inspiring resilience. I found myself cheering for the main character as if she were my own daughter, laughing at her foibles and crying at her pain."

vec_list = tokenize_and_vectorize([(1, sample_2)])
test_vec_list = pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
model.predict(test_vec)
model.predict_classes(test_vec)

