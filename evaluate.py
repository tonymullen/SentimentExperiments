#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:38:45 2019

@author: tonymullen
"""

import os
import numpy as np
import shared
from keras.models import model_from_json
from gensim.models import KeyedVectors

np.random.seed(1337)

maxlen = 400  # was 400 for reviews
embedding_dims = 300  # was 300
word_vectors = KeyedVectors.load_word2vec_format('word_vectors/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)

# Load model
with open("cnn_model.json", "r") as json_file:
    json_string = json_file.read()
model = model_from_json(json_string)
model.load_weights("cnn_weights.h5")


sample_1 = "I hate that the dismal weather had me down for so long, \
when will it break? Ugh, when does happiness return? The sun is blinding \
and the puffy clouds are too thin. I can't wait for the weekend."

vec_list = shared.tokenize_and_vectorize([(1, sample_1)], word_vectors)
test_vec_list = shared.pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list),
                                      maxlen,
                                      embedding_dims))
model.predict(test_vec)
model.predict_classes(test_vec)

sample_2 = "In spite of the dark themes, this was a deeply moving movie, full \
of toughness in the face of adversity and inspiring resilience. I found \
myself cheering for the main character as if she were my own daughter, \
laughing at her foibles and crying at her pain."

vec_list = shared.tokenize_and_vectorize([(1, sample_2)], word_vectors)
test_vec_list = shared.pad_trunc(vec_list, maxlen)
test_vec = np.reshape(test_vec_list, (len(test_vec_list),
                                      maxlen,
                                      embedding_dims))
model.predict(test_vec)
model.predict_classes(test_vec)

################## Sentences eval ###################3
# Evaluate on *sentences* from the training data
# Similar to what's done in model.py for evaluation, but
# on sentences split from the training files
# https://ai.stanford.edu/~amaas/data/sentiment/

data_file_root = os.getcwd() + '/data'
number_of_files = 5000
dataset = shared.pre_process_data(data_file_root + '/aclimdb/train', number_of_files)

print("Length of dataset pre-sentence split:", len(dataset))
dataset = shared.sentences_split(dataset)
print("Length of dataset post-sentence split:", len(dataset))

vectorized_data_s = shared.tokenize_and_vectorize(dataset, word_vectors)
expected_s = shared.collect_expected(dataset)
split_point_s = int(len(vectorized_data_s) * .8)

print('Starting x_train...')
x_train = vectorized_data_s[:split_point_s]
y_train = expected_s[:split_point_s]
x_train = shared.pad_trunc(x_train, maxlen)
x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)

print('Printing predictions...')
f = open("sentences.txt","w")

for i in range(len(x_train)):
    s = slice(i, i+1)
    exp = y_train[i]
    score = model.predict(x_train[s])[0][0]
    classif = model.predict_classes(x_train[s])[0][0]
    if exp == classif:
        x = 1
    else:
        if exp < classif:
            wrongness = score
        elif exp > classif:
            wrongness = 1 - score

    f.write(f'{i}: expected: {exp}, evaluated: {classif}, confidence: {score}, text: {dataset[i][1]}\n')

    '''print(i, "::", "Actual:", str(exp),
        ", Score:", str(score),
        ", Class: ", str(classif),
        " :: ", wrongness)'''
    # f.write(str(round(wrongness, 2)) + " \t " + dataset[i][1] + "\n")






