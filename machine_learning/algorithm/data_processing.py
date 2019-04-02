#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd


def load_data(file, labelmap=None, header='infer', label_column=-1):
    data = pd.read_csv(file, header=header)
    data = data.values
    if labelmap is not None:
        data[:, label_column] = np.array([labelmap[x] for x in data[:, label_column]])
    np.random.shuffle(data)
    return data


def train_test_split(data, label=None, test_size=0.2):
    n = data.shape[0]
    test_num = round(test_size * n)
    train_index = list(range(n))
    test_index = []
    for i in range(test_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        test_index.append(random_index)
        del train_index[random_index]
    train_data = np.array([data[x] for x in train_index])
    test_data = np.array([data[x] for x in test_index])
    if label is not None:
        train_label = np.array([label[x] for x in train_index])
        test_label = np.array([label[x] for x in test_index])
        return train_data, test_data, train_label, test_label
    else:
        return train_data, test_data


def _test():
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data = load_data('data/iris.csv', labelmap=labelmap, header=None)
    train_data, test_data = train_test_split(data)


if __name__ == '__main__':
    _test()
