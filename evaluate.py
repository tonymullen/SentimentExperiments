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

np.random.seed(1337)

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

################## Sentences eval ###################3
# Evaluate on *sentences* from the training data
# Similar to what's done in model.py for evaluation, but
# on sentences split from the training files
word_vectors = get_data('w2v', limit=200000)
data_file_root = '/Users/tonymullen/Dropbox/Northeastern/Classes/NLP/Datasets'
# https://ai.stanford.edu/~amaas/data/sentiment/

number_of_files = 5000

dataset = shared.pre_process_data(data_file_root + '/aclimdb/train',
                                  number_of_files)

print("Length of dataset:", len(dataset))
dataset = shared.sentences_split(dataset)
print("Length of dataset:", len(dataset))

vectorized_data_s = shared.tokenize_and_vectorize(dataset,
                                                  word_vectors)
expected_s = shared.collect_expected(dataset)
split_point_s = int(len(vectorized_data_s) * .8)

x_train = vectorized_data_s[:split_point_s]
y_train = expected_s[:split_point_s]
#x_test = vectorized_data_s[split_point_s:]
#y_test = expected_s[split_point_s:]

x_train = shared.pad_trunc(x_train, maxlen)
#x_test = shared.pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
#x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
#y_test = np.array(y_test)


#model.predict(x_train)
#model.predict_classes(x_train)

#s = slice(0, 1)
#model.predict_classes(x_train[s])


f = open("sentences_new.txt","w")
r = open("training_sentence_results.txt", "w")

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(x_train)):
    s = slice(i, i+1)
    exp = y_train[i]
    score = model.predict(x_train[s])[0][0]
    classif = model.predict_classes(x_train[s])[0][0]
    if exp == classif:
        print(i)
        if classif == 1:
            tp += 1
        else:
            tn += 1
    else:
        if exp < classif:
            fp += 1
            wrongness = score
        elif exp > classif:
            fn += 1
            wrongness = 1 - score

        print(i, "::", "Actual:", str(exp),
            ", Score:", str(score),
            ", Class: ", str(classif),
            " :: ", wrongness)
        pref = "Expected: "+str(exp)+", Evaluated: "+str(classif)+", Wrongness:"+ str(round(wrongness, 2))
        f.write(pref + " \t " + dataset[i][1] + "\n")
        #f.write(str(round(wrongness, 2)) + " \t " + dataset[i][1] + "\n")

if tp > 0 and fp > 0 and fn > 0 and fp > 0:
    print("True positives: \t", tp)
    print("False positives:\t", fp)
    print("True negatives: \t", tp)
    print("False negatives:\t", fp)
    acc = (tp + tn)/(tp + tn + fp +fn)
    rec = tp/(tp + fn)
    prec = tp/(tp + fp)
    f_score = (2 * rec * prec)/(rec + prec)

    print("Accuracy: \t", str(round(acc,2)))
    print("Recall:   \t", str(round(rec,2)))
    print("Precision:\t", str(round(prec,2)))
    print("F-score:  \t", str(round(f_score,2)))

    r.write("True positives: \t"+str(tp)+"\n")
    r.write("False positives:\t"+str(fp)+"\n")
    r.write("True negatives: \t"+str(tn)+"\n")
    r.write("False negatives:\t"+str(fn)+"\n")

    r.write("Accuracy: \t"+str(round(acc,2))+"\n")
    r.write("Recall:   \t"+str(round(rec,2))+"\n")
    r.write("Precision:\t"+str(round(prec,2))+"\n")
    r.write("F-score:  \t"+str(round(f_score,2))+"\n")






