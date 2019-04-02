#!/usr/bin/env python
# coding=utf-8
import numpy as np


def load_data():
    train_set = np.array([[0.0,0.0],
                          [0.1,0.0],
                          [1.0,1.0],
                          [1.0,0.9]])
    train_label = np.array([['A'],
                            ['A'],
                            ['B'],
                            ['B']])
    test_set = np.array([[0.1,0.1],
                          [0.9,0.9]])
    test_label = np.array([['A'],
                            ['B']])
    return train_set, train_label, test_set, test_label


def predict(train_set, train_label, example, k):
    if k > len(train_set):
        print("error:k is too big!")
        return
    # 这部分看了很多别人的代码，最后还是感觉XXXX实战的最简洁
    train_setSize = len(train_set)
    diff_mat = np.tile(example, (train_setSize, 1)) - train_set
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_index = distances.argsort()
    near_class = {}
    for i in range(k):
        label = train_label[sorted_index[i]][0]
        if label not in near_class:
            near_class[label] = 1
        else:
            near_class[label] += 1
    sorted_class = sorted(
        near_class.items(), key=lambda item: item[1], reverse=True)
    return sorted_class[0][0]


def exam(train_set, train_label, test_set, test_label, k):
    count = 0
    for i in range(len(test_set)):
        if test_label[i][0] == predict(train_set, train_label, test_set[i], k):
            count += 1
    error = (len(test_set) - count) / len(test_set)
    print("the test error is: " + str(error))


if __name__ == '__main__':
    train_set, train_label, test_set, test_label = load_data()
    exam(train_set, train_label, test_set, test_label, 2)
