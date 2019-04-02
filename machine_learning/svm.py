import numpy as np
import random
import math


def load_data(filename):
    data = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        content = line.strip().split()
        data.append([float(content[0]), float(content[1])])
        label.append([float(content[2])])
    return np.mat(data), np.mat(label)


# 将x限定在min和max之间
def clip(x, min, max):
    if x >= max:
        return max
    elif x <= min:
        return min
    else:
        return x


# 线性核函数
def line_kernel(data, row):
    return data * row.T

# 高斯核函数


def gauss_kernel(data, one_row, delta):
    m, n = data.shape
    result = np.mat(np.zeros((m, 1)))
    for j in range(m):
        row = data[j, :] - one_row
        result[j] = row * row.T
    result = np.exp(result / (-1 * delta**2))
    return result

# 获得error xi 与 yi 之差


def get_error(model, i):
    return float(np.multiply(model.alpha, model.label).T * model.K[:, i] + model.b) - float(model.label[i])


class SVM:
    def __init__(self, train_set, train_label, C, toler, gauss_delta):
        self.data = train_set   # 训练集
        self.label = train_label  # 训练标签
        self.C = C                # 软间隔参数C
        self.toler = toler          # 松弛变量
        self.m, self.n = train_set.shape
        self.alpha = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.K = np.mat(np.zeros((self.m, self.m)))
        self.error_cache = np.mat(np.zeros((self.m, 2)))
        for i in range(self.m):
            self.K[:, i] = gauss_kernel(
                self.data, self.data[i, :], gauss_delta)
        
        #print(self.K   )
        #print(self.K[0][0])


# 在SMO中取出最大的i 后，如果没有距离最大的j那么就随选择j
# 即在0和m中选取一个不是i的随机数
def get_rand_j(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

# 选择i对应的最远的j 如果没有就随机生成j


def get_j(model, i, error_i):
    max_j = -1
    error_j = 0
    max_delta_error = 0
    effect_error = np.nonzero(model.error_cache[:, 0].A)[0]
    if len(effect_error) > 1:
        for k in effect_error:
            if k == i:
                continue
            error_k = get_error(model, k)
            delta_error = abs(error_i - error_k)
            if delta_error > max_delta_error:
                max_delta_error = delta_error
                max_j = k
                error_j = error_k
        return max_j, error_j
    else:
        max_j = get_rand_j(i, model.m)
        error_j = get_error(model, max_j)
        return max_j, error_j


def update_error(model, k):
    model.error_cache[k] = [1, get_error(model, k)]


def optim_i(model, i):
    error_i = get_error(model, i)
    # 查看i是否满足kkt条件
    # kkt条件
    # a = 0 && yg >= 1
    # 0 < a < C && yg == 1
    # a > C && yg <= 1
    # 将这些条件反过来 然后
    # y(g-y) = yg-y^2 = gy -1 所以转化为 ey 与 0 的关系
    # 然后 不满足kkt条件就转化为
    # ey < 0 &&  a < C   ||   ey > 0 && a > 0
    # 再将0 用松弛变量替换
    if(((error_i * model.label[i] < -model.toler)and(model.alpha[i] < model.C)) or
       ((error_i * model.label[i] > model.toler)and(model.alpha[i] > 0))):
        j, error_j = get_j(model, i, error_i)
        alpha_i = model.alpha[i].copy()
        alpha_j = model.alpha[j].copy()
        if model.label[i] != model.label[j]:
            L = max(0, alpha_j - alpha_i)
            H = min(model.C, model.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - model.C)
            H = min(model.C, alpha_i + alpha_j)
        if L == H:
            print('L==H')
            return 0
        eta = model.K[i,i] + model.K[j,j] - 2.0 * model.K[i,j]
        # eta 是一个平方数，所以不能小于零
        if eta < 0:
            print('eta < 0')
            return 0
        model.alpha[j] += model.label[j] * (error_i - error_j) / eta
        model.alpha[j] = clip(model.alpha[j], L, H)
        update_error(model, j)
        if (abs(model.alpha[j] - alpha_j) < model.toler):
            print('move too small')
            return 0
        model.alpha[i] += alpha_i + model.label[i] * \
            model.label[j]*(alpha_j - model.alpha[j])
        update_error(model, i)
        b1_new = -error_i - model.label[i] * model.K[i,i] * (model.alpha[i] - alpha_i) -  \
                model.label[j] * model.K[j,i] * \
                    (model.alpha[j] - alpha_j) + model.b
        b2_new = -error_j - model.label[i] * model.K[i,j] * (model.alpha[i] - alpha_i) -  \
                model.label[j] * model.K[j,j] * \
                    (model.alpha[j] - alpha_j) + model.b
        if 0 < b1_new < model.C:
            model.b = b1_new
        elif 0 < b2_new < model.C:
            model.b = b2_new
        else:
            model.b = (b1_new + b2_new) / 2
        return 1
    else:
        return 0


def SMO(data_set, label_set, C, torler, epoch):
    model = SVM(data_set, label_set, C, torler, 1.3)
    # 对整个优化还是对边界优化
    optim_entire=True
    # 标记alpha是否改变
    alpha_changed=0.0
    iter = 0
    # 每次优化先对全部进行优化，之后对边界进行优化，边界优化不动就 再去优化全部，全部优化不动就退出
    while(iter < epoch):
        alpha_changed=0.0
        if optim_entire == True:
            for i in range(model.m):
                alpha_changed += optim_i(model, i)
            iter += 1
            print('iter' + str(iter))
            optim_entire=False
            if alpha_changed == 0.0:
                print('out')
                break
        else:
            bound=np.nonzero((model.alpha.A > 0) * (model.alpha.A < model.C))[0]
            for i in bound:
                alpha_changed += optim_i(model, i)
            iter += 1
            print('iter' + str(iter))            
            if alpha_changed == 0:
                optim_entire=True
    return model.b, model.alpha


if __name__ == '__main__':
    train_set, train_label=load_data('train.txt')
    test_set, test_label=load_data('test.txt')
    # print(train_set.shape)
    # print(train_label.shape)
    # print(test_set.shape)
    # print(test_label.shape)
    b,alpha = SMO(train_set,train_label,200,0.0001,10000)
    train_m,train_n = train_set.shape
    train_error_count = 0
    #train_label = train_label.transpose()
    support_vector_index = np.nonzero(alpha)[0]
    support_vector_set = train_set[support_vector_index]
    support_vector_label = train_label[support_vector_index]
    for i in range(train_m):
        kernel = gauss_kernel(support_vector_set,train_set[i,:],1.3)
        predict = kernel.T * np.multiply(support_vector_label,alpha[support_vector_index]) + b
        # print(predict)
        if np.sign(predict) != np.sign(train_label[i]):
            train_error_count += 1
    print('train error '+ str(train_error_count/train_m))
    test_m, test_n = test_set.shape
    test_error_count = 0
    for i in range(test_m):
        kernel = gauss_kernel(support_vector_set,test_set[i,:],1.3)
        predict = kernel.T * np.multiply(support_vector_label,alpha[support_vector_index]) + b
        if np.sign(predict) != np.sign(test_label[i]):
            test_error_count += 1
    print('test error '+ str(test_error_count/test_m))
