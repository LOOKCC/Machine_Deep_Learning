#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from data_processing import *
from evaluate import accuracy


def calc_mean(values):
    return np.sum(values) / len(values)


def calc_variance(values):
    length = len(values)
    if length == 1:
        return 0.0
    return np.sum((values - calc_mean(values))**2) / (length - 1)


def calc_probability(mean, variance, x):
    stdev = np.sqrt(variance)
    return 1 / np.sqrt(2*np.pi) / stdev * np.exp(-(x-mean)**2 / 2 / variance)


def calNumberProbability(mean, std, y):
    exponent = np.exp(-((y - mean)**2/(2*std)))
    return (1 / (np.sqrt(2*np.pi) * np.sqrt(std))) * exponent


def get_mean_stdev(data_set, label_set):
    data_info = {}
    for i in range(len(label_set)):
        if label_set[i] in data_info:
            info_for_class = data_info[label_set[i]]
            for j in range(len(info_for_class)):
                info_for_class[j].append(data_set[i][j])
        else:
            data_info[label_set[i]] = [[x] for x in data_set[i]]
    for class_type in data_info:
        data_info[class_type] = [(calc_mean(x), calc_variance(x))
                                 for x in data_info[class_type]]
    return data_info


def predict(data_info, test):
    probs = []
    test = np.float64(test)
    for class_type in range(len(data_info)):
        prob = np.ones(len(test))
        class_info = data_info[class_type]
        for i in range(len(class_info)):
            mean, variance = class_info[i]
            prob *= calc_probability(mean, variance, test[:, i])
        probs.append(prob)
    probs = np.array(probs)
    label_prob = [(max(enumerate(x), key=lambda x: x[1])) for x in probs.T]
    return np.array(list(zip(*label_prob)))


if __name__ == '__main__':
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data = load_data('data/iris.csv', labelmap, None)
    data, label = data[:, :-1], data[:, -1]
    train, test, train_label, test_label = train_test_split(data, label)

    data_info = get_mean_stdev(train, train_label)
    result = predict(data_info, test)[0]
    print(accuracy(test_label, result))

