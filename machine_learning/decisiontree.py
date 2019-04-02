#!/usr/bin/env python
# coding=utf-8
import operator
from math import log
import numpy as np
import pandas as pd
import plottree 


def load_data_watermenlon(file_name):
    ld = pd.read_csv(file_name)
    data_set = ld.values[:, 1:].tolist()
    data_full = data_set[:]
    label = ld.columns.values[1:-1].tolist()
    label_full = label[:]O
    return data_set, label, data_full, label_full


def load_data_old(file_name):
    ld = pd.read_csv(file_name)
    temp_data = ld.values[:, 1:]
    data = temp_data[:, 1:]
    data = np.insert(data, data.shape[1], values=temp_data[:, 0], axis=1)
    data_set = data.tolist()
    data_full = data_set[:]
    label = ld.columns.values[2:].tolist()
    label_full = label[:]
    cow = range(1, len(data_set) + 1)
    column = label[:]
    column.append("Survived")
    print(column)
    print(label)
    data_save = pd.DataFrame(data=data_set, index=cow, columns=column)
    data_save.to_csv('train2.csv')
    return data_set, label, data_full, label_full


def load_data(file_name):
    ld = pd.read_csv(file_name)
    del(ld['Name'])
    del(ld['Ticket'])
    del(ld['Cabin'])
    ld.fillna(value=100)
    temp_data = ld.values[:, 1:]
    print(temp_data.shape)
    data_set = temp_data.tolist()
    data_full = data_set[:]
    label = ld.columns.values[1:-1].tolist()
    label_full = label[:]
    print(label)
    return data_set, label, data_full, label_full

# 计算香农墒


def cal_shannon_ent(data_set):
    total = len(data_set)
    label_count = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_count:
            label_count[current_label] = 1
        else:
            label_count[current_label] += 1
    shannon_ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / total
        shannon_ent = -prob * log(prob, 2)
    return shannon_ent


# 对标签axis（列）的 值为value的样本进行划分，针对离散型变量
def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduce_feat_vec)
    return ret_data_set


# 同上针对连续型变量
# 其中decidion是指选择大于还是小于的 大于是0 小于是1
def split_continue_data_set(data_set, axis, value, decision):
    ret_data_set = []
    for feat_vec in data_set:
        if decision == 0:
            if feat_vec[axis] > value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduce_feat_vec)
        if decision == 1:
            if feat_vec[axis] <= value:
                reduce_feat_vec = feat_vec[:axis]
                reduce_feat_vec.extend(feat_vec[axis + 1:])
                ret_data_set.append(reduce_feat_vec)
    return ret_data_set


# 选择应该划分数据的方式
def choose_best_feature_to_split(data_set, label):
    num_features = len(data_set[0]) - 1    # 还需要区分的属性的长度
    base_entropy = cal_shannon_ent(data_set)  # Ent(D)  刚开始的香农熵
    best_info_gain = -100                 # 某次迭代后的最大的信息增益
    best_feature = -1                   # 某次迭代后最大的信息增益对应的属性
    best_split_dict = {}                 # 用于记录连续值的分割方法
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        # 如果数据类型是连续的
        if(type(feat_list[0]).__name__ == 'float' or
           type(feat_list[0]).__name__ == 'int'):
            # 这将产生n-1个可以 分开的点
            # 先排序 然后使用split_list 来记录这些分开的点，这些点通过取排好序的两点的平均值确定
            sort_feat_list = sorted(feat_list)
            split_list = []
            for j in range(len(sort_feat_list) - 1):
                split_list.append((sort_feat_list[j] +
                                  sort_feat_list[j + 1]) / 2.0)
            best_split_entropy = 100000  # 初始化为一个很大的数
            for j in range(len(split_list)):
                value = split_list[j]
                new_entropy = 0.0
                sub_data_set0 = split_continue_data_set(data_set, i, value, 0)
                sub_data_set1 = split_continue_data_set(data_set, i, value, 1)
                prob0 = len(sub_data_set0) / len(data_set)
                new_entropy += prob0 * cal_shannon_ent(sub_data_set0)
                prob1 = len(sub_data_set1) / len(data_set)
                new_entropy += prob1 * cal_shannon_ent(sub_data_set1)
                if new_entropy < best_split_entropy:
                    best_split_entropy = new_entropy
                    best_split = j
            best_split_dict[label[i]] = split_list[best_split]
            info_gain = base_entropy - best_split_entropy
        # 如果数据类型是离散的
        else:
            values = set(feat_list)
            new_entropy = 0.0
            for value in values:
                sub_data_set = split_data_set(data_set, i, value)
                prob = len(sub_data_set) / len(data_set)
                new_entropy += prob * cal_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
        if info_gain >= best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    # 如果是连续值作为分割的话，需要将属性改为 xxx< xxx 并且将这一个属性下的值改为二值化
    if (type(data_set[0][best_feature]).__name__ == 'float' or type(data_set[0][best_feature]).__name__ == 'int'):
        best_split_value = best_split_dict[label[best_feature]]
        label[best_feature] = label[best_feature] + \
            '<=' + str(best_split_value)
        for i in range(len(data_set)):
            if data_set[i][best_feature] <= best_split_value:
                data_set[i][best_feature] = 1
            else:
                data_set[i][best_feature] = 0
    return best_feature


# 如果最后当特征分完后，也就是说在一个叶子节点那里，既有正样本，还有负样本
# 这时候需要进行投票。多的作为这个节点的结果
def finall_vote(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 1
        else:
            class_count[vote] += 1
    return max(class_count)


# 递归构造树
def create_tree(data_set, label, data_full, label_full):
    if len(data_set) == 0:
        return
    class_list = [example[-1] for example in data_set]
    # 如果当前节点包含的样本属于同一类别，则返回，无需划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果当前属性集合为空 也就是长度只剩 1 即label 的时候
    if len(data_set[0]) == 1:
        return finall_vote(class_list)
    best_feat = choose_best_feature_to_split(data_set, label)
    best_feat_label = label[best_feat]
    my_tree = {best_feat_label: {}}
    feat_values = [example[best_feat] for example in data_set]
    values = set(feat_values)
    '''
    这里是一个很玄学的地方，很多人的代码都没有这部分，我看了很久才看懂：
    这里会有很多字符串的判断，主要做的是：有时会出先这种情况，在进入一个节点之后，
    某些样本因为某些原因，失去了一些标签，这时候如果在判断的时候缺少这些标签，
    就会造成决策数树无法决策的问题,我们可以采取多数投票的方式来解决这个问题，补上这些树枝
    '''
    if type(data_set[0][best_feat]).__name__ == 'str':
        current_label = label_full.index(label[best_feat])
        feat_values_full = [example[current_label] for example in data_full]
        values_full = set(feat_values_full)
    del(label[best_feat])
    for value in values:
        sub_label = label[:]  # 值拷贝
        if type(data_set[0][best_feat]).__name__ == 'str':
            values_full.remove(value)
        my_tree[best_feat_label][value] = create_tree(split_data_set(
            data_set, best_feat, value), sub_label, data_full, label_full)
    if type(data_set[0][best_feat]).__name__ == 'str':
        for value in values_full:
            my_tree[best_feat_label][value] = finall_vote(class_list)
    return my_tree


def classify(tree, test):
    result = -1
    if type(tree).__name__ == 'int':
        print(tree)
        return tree
    label = list(tree.keys())[0]
    if label.find('<=') != -1:
        str_list = label.split('<=')
        label_str = str_list[0]
        label_count = float(str_list[-1])
        if float(test[label_str]) <= label_count:
            result = classify(tree[label][1], test)
        else:
            result = classify(tree[label][0], test)
    else:
        label_list = list(tree[label].keys())
        for i in range(len(label_list)):
            if test[label] == label_list[i]:
                result = classify(tree[label][label_list[i]], test)
    return result


def test_error(tree, test_set, test_label):
    count = 0
    error_data = 0
    for i in range(len(test_set)):
        if classify(tree, test_set[i]) == 1 and test_label[i] == 0:
            count += 1
        if classify(tree, test_set[i]) == 0 and test_label[i] == 1:
            comut += 1
        if classify(tree, test_set[i]) == -1:
            print(test_set[i] + '  have something wrong.')
            error_data += 1
    return count / len(data_set)
    # return count/(len(test_set) - error_data)


def pruning(tree, data_set, test_set, label):
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    class_list = [example[-1] for example in data_set]
    feat_key = copy.deepcopy(first_str)
    if first_str.find('<=') != -1:
        str_list = first_str.split('<=')
        feat_key = str_list[0]
        feat_value = float(str_list[-1])
    label_index = label.index(feat_key)
    temp_label = copy.deepcopy(label)
    del(label[label_index])
    for keys in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            if type(data_set[0][label_index]).__name__ == 'str':
                tree[first_str][key] = pruning(second_dict[key],
                split_data_set(data_set, label_index, key),
                split_data_set(test_set, label_index, key),
                copy.deepcopy(label))
            else:
                tree[first_str][key] = pruning(second_dict[key],
                split_continue_data_set(
                    data_set, label_index, feat_value, key),
                split_continue_data_set(
                    test_set, label_index, feat_value, key),
                copy.deepcopy(label))
    if test_error(tree, test_set, ([example[-1] for example in test_set])
                  <= test_major(finall_vote(class_list), test_set):
        return tree
    return finall_vote(class_list)


def test_major(label, data):
    count=0
    for i in range(len(data)):
        if label != data[-1]:
            count += 1
    return count / len(data)

if __name__ == '__main__':
    data_set, label, data_full, label_full=load_data_watermenlon(
        'watermelon.csv')
    tree=create_tree(data_set, label, data_full, label_full)
    print(tree)
    test={'texture': 'little_blur', 'touch': 'soft_stick', 'density': 0.35}
    print(classify(tree, test))
