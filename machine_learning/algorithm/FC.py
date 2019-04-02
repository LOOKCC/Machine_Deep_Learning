#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from data_processing import *
from evaluate import accuracy


class FC:

    def __init__(self, sizes):
        self.sizes = sizes
        net_struct = zip(sizes[:-1], sizes[1:])
        self.num_layers = len(sizes)
        self.weight = [np.random.randn(y, x) for x, y in net_struct]
        self.bias = [np.random.randn(y, 1) for y in sizes[1:]]
        self.pre_nabla_w = [np.zeros(w.shape) for w in self.weight]
        self.pre_nabla_b = [np.zeros(b.shape) for b in self.bias]
        self.grad_w_sum = [np.zeros(w.shape) for w in self.weight]
        self.grad_b_sum = [np.zeros(b.shape) for b in self.bias]
        self.mt_1_w = [np.zeros(w.shape) for w in self.weight]
        self.mt_1_b = [np.zeros(b.shape) for b in self.bias]
        self.vt_1_w = [np.zeros(w.shape) for w in self.weight]
        self.vt_1_b = [np.zeros(b.shape) for b in self.bias]
        self.t = 0

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.float32(x)))

    def sigmoid_derivate(self, x):
        sigmoid_result = self.sigmoid(x)
        return sigmoid_result * (1 - sigmoid_result)

    def forward(self, x):
        for b, w in zip(self.bias, self.weight):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    def backward(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        activation = x
        activations = [x]
        onehot = np.zeros((3, 1))
        onehot[y, 0] = 1
        y = onehot
        outputs = []
        for w, b in zip(self.weight, self.bias):
            out = np.dot(w, activation) + b
            outputs.append(out)
            activation = self.sigmoid(out)
            activations.append(activation)
        delta = (activations[-1] - y) * self.sigmoid_derivate(outputs[-1])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        nabla_b[-1] = delta
        for i in range(2, len(self.sizes)):
            d_sig = self.sigmoid_derivate(outputs[-i])
            delta = np.dot(self.weight[-i+1].transpose(), delta) * d_sig
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
            nabla_b[-i] = delta
        return nabla_w, nabla_b

    def update(self, batch, eta, momentum=0,
               method='SGD', beta1=0.9, beta2=0.999):
        nabla_w_sum = [np.zeros(w.shape) for w in self.weight]
        nabla_b_sum = [np.zeros(b.shape) for b in self.bias]
        for x, y in batch:
            nabla_w, nabla_b = self.backward(x, y)
            nabla_w_sum = [a + b for a, b in zip(nabla_w, nabla_w_sum)]
            nabla_b_sum = [a + b for a, b in zip(nabla_b, nabla_b_sum)]
        batch_size = len(batch)
        eta_para = eta / batch_size
        momentum_para = momentum / batch_size
        if method == 'adagrad':
            self.grad_w_sum = [(a/batch_size)**2 + b for a, b in
                               zip(nabla_w_sum, self.grad_w_sum)]
            self.grad_b_sum = [(a/batch_size)**2 + b for a, b in
                               zip(nabla_b_sum, self.grad_b_sum)]
            epsilon = 1E-8
            self.weight = np.array([w - eta_para * nabla / np.sqrt(np.float32(
                epsilon + sq_sum))
                for w, nabla, sq_sum in zip(self.weight,
                                            nabla_w_sum, self.grad_w_sum)])
            self.bias = np.array([b - eta_para * nabla / np.sqrt(np.float32(
                epsilon + sq_sum))
                for b, nabla, sq_sum in zip(self.bias,
                                            nabla_b_sum, self.grad_b_sum)])
        elif method == 'adam':
            epsilon = 1E-8
            mt_w = [(beta1*mt_1 + (1-beta1) * gt/batch_size)
                    for mt_1, gt in zip(self.mt_1_w, nabla_w_sum)]
            mt_b = [(beta1*mt_1 + (1-beta1) * gt/batch_size)
                    for mt_1, gt in zip(self.mt_1_b, nabla_b_sum)]
            vt_w = [(beta2*vt_1 + (1-beta2)*(gt/batch_size)**2)
                    for vt_1, gt in zip(self.vt_1_w, nabla_w_sum)]
            vt_b = [(beta2*vt_1 + (1-beta2)*(gt/batch_size)**2)
                    for vt_1, gt in zip(self.vt_1_b, nabla_b_sum)]
            (self.mt_1_w, self.mt_1_b, self.vt_1_w,
             self.vt_1_b) = mt_w, mt_b, vt_w, vt_b
            mt_w = [mt / (1 - beta1**self.t) for mt in mt_w]
            mt_b = [mt / (1 - beta1**self.t) for mt in mt_b]
            vt_w = [vt / (1 - beta2**self.t) for vt in vt_w]
            vt_b = [vt / (1 - beta2**self.t) for vt in vt_b]
            delta_w = [eta*m/(np.sqrt(np.float32(v))+epsilon)
                       for m, v in zip(mt_w, vt_w)]
            delta_b = [eta*m/(np.sqrt(np.float32(v))+epsilon)
                       for m, v in zip(mt_b, vt_b)]
            self.weight = np.array(
                [w - delta for w, delta in zip(self.weight, delta_w)])
            self.bias = np.array(
                [b - delta for b, delta in zip(self.bias, delta_b)])
        elif momentum:
            self.weight = np.array([w - nabla * eta_para - pre * momentum_para
                                    for w, nabla, pre in zip(self.weight,
                                                             nabla_w_sum,
                                                             self.pre_nabla_w)
                                    ])
            self.bias = np.array([b - nabla * eta_para - pre * momentum_para
                                  for b, nabla, pre in zip(self.bias,
                                                           nabla_b_sum,
                                                           self.pre_nabla_b)])
            self.pre_nabla_w, self.pre_nabla_b = nabla_w_sum, nabla_b_sum
        else:
            self.weight = np.array([w - nabla * eta_para for w, nabla in
                                    zip(self.weight, nabla_w_sum)])
            self.bias = np.array([b - nabla * eta_para for b, nabla in
                                  zip(self.bias, nabla_b_sum)])

    def train(self, train_data, eta=0.1, epoch_num=100, batch_size=1,
              test_data=None, method='mini_batch', momentum=0):
        for i in range(epoch_num):
            np.random.shuffle(train_data)
            batch_split = range(0, len(train_data), batch_size)
            batches = [train_data[x: x+batch_size] for x in batch_split]
            self.t += 1
            for batch in batches:
                self.update(batch, eta, momentum=momentum, method=method)
            if test_data is not None:
                print('Epoch: ', i,
                      '\t\ttrain accuracy: ', self.evaluate(train_data),
                      '\t\ttest accuracy: ', self.evaluate(test_data))
            else:
                print('Epoch: ', i, 'finished')

    def predict(self, predict_data):
        result = [self.forward(x) for x in predict_data]
        label_prob = [(max(enumerate(x), key=lambda x: x[1])) for x in result]
        return np.array(list(zip(*label_prob)))

    def evaluate(self, test_data):
        test_x, test_y = test_data[:, 0], test_data[:, 1]
        # print(test_y)
        # print(np.array([int(x) for x in self.predict(test_x)[0]]))
        return accuracy(test_y, self.predict(test_x)[0])


def main():
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data = load_data('data/iris.csv', labelmap=labelmap, header=None)
    train_data, test_data = train_test_split(data)
    train_data = np.array(
        [(x[:-1].reshape((len(x)-1, 1)), x[-1]) for x in train_data])
    test_data = np.array([(x[:-1].reshape((len(x)-1, 1)), x[-1])
                          for x in test_data])
    fc_net = FC((4, 10, 3))
    fc_net.train(train_data, 0.05, 200, 10,
                 test_data=test_data, method='adam', momentum=0)

if __name__ == '__main__':
    main()
