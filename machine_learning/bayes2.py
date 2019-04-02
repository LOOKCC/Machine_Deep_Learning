import numpy as np
import pandas as pd
import random
from math import log


def load_data():
    train_set = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    train_label = [0,1,0,1,0,1]
    return train_set, train_label


def creat_vector(data_set):
    word_list = []
    for i in range(len(data_set)):
        word_list += data_set[i]
    word_set = set(word_list)
    return list(word_set)


def set_train_vector(word_list, input_vector):
    ret = [0] * len(word_list)
    for word in word_list:
        if word in input_vector:
            ret[word_list.index(word)] += 1
    return ret


def calculate_probability(train_vector, train_label):
    num_doc = len(train_vector)
    num_word = len(train_vector[0])
    pos_prop = sum(train_label)/float(num_doc)
    p0_son = np.ones(num_word)
    p1_son = np.ones(num_word)
    p0_mom = 2
    p1_mom = 2
    for i in range(num_doc):
        if train_label[i] == 1:
            p1_son += train_vector[i]
            p1_mom += sum(train_vector[i])
        else:
            p0_son += train_vector[i]
            p0_mom += sum(train_vector[i])
    p0vec = mylog(p0_son/p0_mom)
    p1vec = mylog(p1_son/p1_mom)
    return pos_prop, p0vec, p1vec


def mylog(vector):
    for i in range(len(vector)):
        vector[i] = log(vector[i])
    return vector

def classify(test_vector, pos_prop, p0vec, p1vec):
    test_vector = np.array(test_vector)
    print(test_vector)
    print(p0vec)
    print(test_vector * p0vec)
    p0 = sum(test_vector * p0vec) + log(1-pos_prop)
    p1 = sum(test_vector * p1vec) + log(pos_prop)
    if p1 > p0:
        return 1
    else:
        return 0

 

if __name__ == '__main__':
    train_set, train_label = load_data()
    word_list = creat_vector(train_set)
    train_vector = []
    for i in range(len(train_set)):
        train_vector.append(set_train_vector(word_list,train_set[i]))
    pos_prop, p0vec, p1vec = calculate_probability(train_vector,train_label)
    #print(type(p0vec).__name__)    
    test_set = [['love','my','dalmation'],
                ['stupid','garbage']]
    test_vector = []
    for i in range(len(test_set)):
        test_vector.append(set_train_vector(word_list,test_set[i]))
    for i in range(len(test_vector)):
        print(classify(test_vector[i],pos_prop,p0vec,p1vec))
    

