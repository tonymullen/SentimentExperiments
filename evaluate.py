#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:38:45 2019

@author: tonymullen
"""

import numpy as np
import shared
from keras.models import model_from_json
from nlpia.loaders import get_data


maxlen = 400  # was 400 for reviews
embedding_dims = 300  # was 300
word_vectors = get_data('w2v', limit=200000)

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
