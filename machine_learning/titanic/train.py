#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import pandas as pd
import numpy as np
sys.path.append("..")

import C45

def load_data(file, is_train=True):
    data = pd.read_csv(file)
    del data['Name']
    del data['Ticket']
    del data['Cabin']
    sex_map = {'male': 0, 'female': 1}
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    data['Sex'] = data['Sex'].map(sex_map)
    data['Embarked'] = data['Embarked'].map(embarked_map)
    values = data.values
    if is_train:
        label = values[:, 1]
        data = values[:, 2:]
        sortable = [True, False, True, True, True, True, False]
        return data, label, sortable
    else:
        data = values[:, 1:]
        return data


def train_test_split(data, label, test_size=0.2):
    n = data.shape[0]
    test_num = round(test_size * n)
    train_index = list(range(n))
    test_index = []
    for i in range(test_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        test_index.append(random_index)
        del train_index[random_index]
    train = np.array([data[x] for x in train_index])
    train_label = np.array([label[x] for x in train_index])
    test = np.array([data[x] for x in test_index])
    test_label = np.array([label[x] for x in test_index])
    return train, test, train_label, test_label


def main():
    data, label, sortable = load_data('train.csv')
    result_sum = 0
    iter_num = 8
    for i in range(iter_num):
        train, test, train_label, test_label = train_test_split(data, label)
        # predict_data = load_data('test.csv', False)
        dt = C45.DicisionTree()
        dt.train(train, train_label, sortable, 0)
        evaluate_result = dt.evaluate(test, test_label)
        print(evaluate_result)
        print()
        result_sum += evaluate_result
    print(result_sum / iter_num)
    # dt.save('titanic_tree.pkl')
    dt.plot()

if __name__ == '__main__':
    main()