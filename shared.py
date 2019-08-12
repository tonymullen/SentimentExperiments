#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:38:45 2019

@author: tonymullen
"""

from nltk.tokenize import TreebankWordTokenizer
from nltk import sent_tokenize
from random import shuffle

import glob
import os


def pre_process_data(filepath, number_of_files, dataset):
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


def sentences_split(dataset):
    labeled_sents = []
    for label_txt in dataset:
        for s in sent_tokenize(label_txt[1]):
            labeled_sents.append((label_txt[0], s))
    return labeled_sents


def tokenize_and_vectorize(dataset, word_vectors):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                # no matching token in Google w2v vocab
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data


def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


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
