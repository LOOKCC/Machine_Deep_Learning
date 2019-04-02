import numpy as np
import pandas as pd
import random
import math

def load_data(file_name):
    data_frame = pd.read_csv(file_name)
    value =data_frame.values
    data = value.tolist()
    return data


def split_data(data,split_ratio):
    train_size = len(data)*split_ratio
    train_set = []
    data_copy = data[:]
    while len(train_set) < train_size:
        index = random.randrange(len(data_copy))
        train_set.append(data_copy.pop(index))
    return train_set,data_copy


def split_class(data_set):
    splited = {}
    for i in range(len(data_set)):
        row = data_set[i]
        if row[-1] not in splited:
            splited[row[-1]] = []
        splited[row[-1]].append(row)
    return splited


def mean(number):
    return sum(number)/len(number)


def std(number):
    average = mean(number)
    variance = sum([pow(x - average,2) for x in number])/(len(number)-1)
    return math.sqrt(variance)


def mean_std(data_set):
    result = [(mean(data), std(data)) for data in zip(*data_set)]
    del(result[-1])
    return result


def get_all_mean_std(data_set):
    splited = split_class(data_set)
    result = {}
    for key in splited.keys():
        result[key] = mean_std(splited[key])
    return result


def calculate_probability(x,mean,std):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    return (1/(math.sqrt(2*math.pi)*std)) * exponent

def get_all_probability(means_stds, test_vector):
    probabilities = {}
    for key in means_stds.keys():
        probabilities[key] = 1
        for i in range(len(means_stds[key])):
            mean,std = means_stds[key][i]
            x = test_vector[i]
            probabilities[key] *= calculate_probability(x,mean,std)
    return probabilities


def predict(means_stds,test_vector):
    probabilities = get_all_probability(means_stds,test_vector)
    best_label, best_prob = None, -1
    for key in probabilities.keys():
        if best_label is None or probabilities[key] > best_prob:
            best_label = key
            best_prob = probabilities[key]
    return best_label


def get_predictions(means_stds,test_set):
    predictions = []
    for i in range(len(test_set)):
        result = predict(means_stds,test_set[i])
        predictions.append(result)
    return predictions


def get_acc(test_set,predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return correct/len(test_set)


if __name__ == '__main__':
    data = load_data('data.csv')
    train_set,test_set = split_data(data,0.8)
    means_stds = get_all_mean_std(train_set)
    predictions = get_predictions(means_stds,test_set)
    acc = get_acc(test_set,predictions)
    print(acc)

