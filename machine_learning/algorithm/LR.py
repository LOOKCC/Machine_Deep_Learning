#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logistic(x):
    x = np.array(x, dtype=np.float32)
    return 1.0 / (1 + np.exp(- x))


def load_data(file):
    data = pd.read_csv(file, header=None)
    values = data.values
    return values[:, :-1], values[:, -1]


def LR(data, label, alpha=0.01, iternum=500, regul_para=0, method='BGD', sample_size=5):
    m, n = data.shape
    data = np.c_[np.ones(m), data]
    n += 1
    theta = np.random.random(n)
    if method == 'BGD':
        for i in range(iternum):
            error = label - logistic(np.dot(data, theta))
            theta[1:] += theta[1:] * alpha * regul_para / m
            theta += alpha * np.dot(error, data) / m
    elif method == 'SGD':
        for i in range(iternum):
            random_index = np.random.choice(m, 1)
            h = logistic(np.dot(data[random_index], theta))
            error = label[random_index] - h
            theta[1:] += theta[1:] * alpha * regul_para / m
            theta += alpha * np.dot(error, data[random_index])
    elif method == 'MBGD':
        for i in range(iternum):
            random_index = np.random.choice(m, sample_size)
            sub_data = np.array([data[x] for x in random_index])
            sub_label = np.array([label[x] for x in random_index])
            error = sub_label - logistic(np.dot(sub_data, theta))
            theta[1:] += theta[1:] * alpha * regul_para / m
            theta += alpha * np.dot(error, sub_data)
    elif method == 'N':
        for it in range(iternum):
            A = np.zeros((m, m))
            h = logistic(np.dot(data, theta.T))
            for i in range(m):
                A[i, i] = h[i] * (h[i] - 1)
            H = np.mat(np.dot(np.dot(data.T, A), data))
            error = label - logistic(np.dot(data, theta.T))
            theta -= np.dot(np.array(H.I), np.dot(error, data))
    else:
        raise NameError('Not support optimize method type!')
    return theta


def plot(data, label, theta):
    f = plt.figure(1)
    plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], (label + 1) * 15)
    x = np.arange(-3.0, 4.0)
    y = -(theta[0] + theta[1] * x) / theta[2]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    data, label = load_data('data/binary_classification.csv')
    data[:, 1] = data[:, 1] / 5
    theta = LR(data, label, 0.01, 5000, regul_para=0.3, method='BGD')
    plot(data, label, theta)
