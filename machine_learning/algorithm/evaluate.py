#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import matplotlib.pyplot as plt


def Fb_score(true_label, predict_label, b=1):
    length = len(true_label)
    assert(length == len(predict_label))
    TP = FN = FP = 0
    for i in length:
        if true_label[i] > 0:
            if predict_label[i] > 0:
                TP += 1
            else:
                FN += 1
        elif predict_label[i] > 0:
            FP += 1
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    score = (1 + b**2) * P * R / ((b**2 * P) + R)
    return score


def recall(true_label, predict_label):
    length = len(true_label)
    assert(length == len(predict_label))
    TP = FN = 0
    for i in length:
        if true_label[i] > 0:
            if predict_label[i] > 0:
                TP += 1
            else:
                FN += 1
    R = TP / (TP + FN)
    return R


def precision(true_label, predict_label):
    length = len(true_label)
    assert(length == len(predict_label))
    TP = FP = 0
    for i in length:
        if predict_label[i] > 0:
            if true_label[i] > 0:
                TP += 1
            else:
                FP += 1
    P = TP / (TP + FP)
    return P


def roc_curve(true_label, scores, plot=False):
    length = len(true_label)
    assert(length == len(scores))

    alldata = list(zip(scores, true_label))
    alldata.sort()
    scores, true_label = zip(*alldata)

    true_label = [x > 0 for x in true_label]
    P = sum(true_label)
    N = length - P

    x_pos = []
    y_pos = []
    for i in range(length, -1, -1):
        TP = sum(true_label[i:])
        FP = length - i - TP
        x_pos.append(FP / N)
        y_pos.append(TP / P)

    if plot:
        plt.plot(x_pos, y_pos, 'o-')
        plt.title('ROC curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()

    return x_pos, y_pos


def auc(true_label, scores, plot=False):
    x_pos, y_pos = roc_curve(true_label, scores, plot)
    auc = pre_x = 0
    for x, y in zip(x_pos, y_pos):
        if x != pre_x:
            auc += (x - pre_x) * y
            pre_x = x
    return auc


def accuracy(true_label, predict_label):
    length = len(predict_label)
    assert len(true_label) == length
    correct = sum([x == y for x, y in zip(true_label, predict_label)])
    return correct / length


if __name__ == '__main__':
    true_label = [1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    scores = [random.random() for i in range(len(true_label))]
    print(auc(true_label, scores, True))
