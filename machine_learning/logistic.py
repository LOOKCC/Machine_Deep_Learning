#!/usr/bin/env python
# coding=utf-8
from math import exp
import random
import numpy as np


def sigmoid_nparray(x):
    ret = np.zeros((len(x), 1))
    for i in range(len(x)):
        ret[i][0] = 1.0 / (1.0 + exp(-x[i][0]))
    return ret


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


# 一般梯度下降
def grad_descent(train_set, train_label, alpha, max_cycle):
    m, n = train_set.shape
    weight = np.ones((n, 1))
    for i in range(max_cycle):
        h = sigmoid_nparray(train_set.dot(weight))
        error = train_label - h
        weight += alpha * train_set.transpose().dot(error)
    return weight


# 随机梯度下降
def sorc_grad_decent(train_set, train_label, alpha):
    m, n = train_set.shape
    weight = np.ones((n, 1))
    for i in range(m):
        h = sigmoid(train_set[i].dot(weight)[0])
        error = train_label[i][0] - h
        weight += np.array(train_set[i], ndmin=2).transpose() * alpha * error
    return weight


# 改进的随机梯度下降，alpha可变，打乱顺序训练
def sorc_grad_decent1(train_set, trainlebal, max_cycle):
    m, n = train_set.shape
    weight = np.zeros((n, 1))
    for i in range(max_cycle):
        dataIndex = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(train_set[randIndex].dot(weight)[0])
            error = train_label[randIndex][0] - h
            weight += alpha * error * np.array(train_set[randIndex],ndmin=2).transpose()
            del(dataIndex[randIndex])
        return weight


# 牛顿法
def newton(train_set, train_label, alpha, max_cycle):
    m, n = train_set.shape
    weight = np.zeros((n, 1))
    for i in range(max_cycle):
        prime1 = train_set.transpose().dot \
            (sigmoid_nparray(train_set.dot(weight)) - train_label)
        A = np.zeros((m, m))
        for j in range(m):
            H = sigmoid(train_set[j].dot(weight)[0])
            A[j, j] = H*(1 - H)
        prime2 = (train_set.transpose().dot(A)).dot(train_set) 
        weight -= alpha * np.mat(prime2).I.dot(prime1)
    return weight


def classfy(weight, test_set, test_lebel, value=0.5):
    h = test_set.dot(weight)
    count = 0
    for i in range(len(h)):
        if h[i] >= value and test_lebel[i] == 1:
            count += 1
        elif h[i] < value and test_lebel[i] == 0:
            count += 1
    error = (len(test_lebel) - count) / len(test_lebel)
    print('the error is: ' + str(error))


def loaddata():
    train_set = np.array([[1.0, 1.1, 1.0],
                          [1.0, 1.0, 1.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.1, 1.0]])
    train_label = np.array([[1.0],
                            [1.0],
                            [0.0],
                            [0.0]])
    test_set = np.array([[1.0, 0.9, 1.0],
                         [0.1, 0.1, 1.0]])
    test_label = np.array([[1.0],
                           [0.0]])
    return train_set, train_label, test_set, test_label


if __name__ == '__main__':
    train_set, train_label, test_set, test_label = loaddata()
    print(train_set)
    print(train_label)
    print(test_set)
    print(test_label)
    #weight = sorc_grad_decent(train_set, train_label, 0.01)
    #weight = grad_descent(train_set, train_label, 0.01, 1000)
    #weight = sorc_grad_decent1(train_set,train_label,1000)
    weight = newton(train_set,train_label,0.01,1000)
    classfy(weight, test_set, test_label)
