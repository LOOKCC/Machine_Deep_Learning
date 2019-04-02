#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

def load_data(file, labelmap=None):
    data = pd.read_csv(file, header=None)
    values = data.values
    data = values[:, :-1]
    label = values[:, -1]
    if labelmap:
        label = np.array([labelmap[x] for x in label])
    return data, label


def train_test_split(data, label, test_size=0.2):
    n = data.shape[0]
    test_num = round(test_size * n)
    train_index = list(range(n))
    test_index = []
    for i in range(test_num):
        random_index=int(np.random.uniform(0,len(train_index)))
        test_index.append(random_index)
        del train_index[random_index]
    train = np.array([data[x] for x in train_index])
    train_label = np.array([label[x] for x in train_index])
    test = np.array([data[x] for x in test_index])
    test_label = np.array([label[x] for x in test_index])
    return train, test, train_label, test_label


def get_most(labels):
    count = {}
    for label in labels:
        if label in count:
            count[label] += 1
        else:
            count[label] = 1
    most_label = 0
    max_num = 0
    for label, num in count.items():
        if num > max_num:
            max_num = num
            most_label = label
    return most_label


def kNN(Xs, data, label, k = 3):
    length = len(data)
    result = []
    for X in Xs:
        dists = []
        for i in range(length):
            dist = np.sqrt(np.sum((data[i] - X)**2))
            dists.append((dist, i))
        dists = sorted(dists, key=lambda s: s[0])
        h_labels = [label[x[1]] for x in dists[:k]]
        result.append(get_most(h_labels))
    return np.array(result)


if __name__ == '__main__':
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data, label = load_data('data/iris.csv', labelmap)

    train, test, train_label, test_label = train_test_split(data, label)
    print(kNN(test, train, train_label))
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train, train_label)
    print(neigh.predict(test))
    print(test_label)
