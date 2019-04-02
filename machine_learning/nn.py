import numpy as np
import random
import math
import cv2
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

random.seed(1)
class NeuralNet(object):

    # 初始化，sizes是一个list [input, 隐藏层1，隐藏层2...., output]
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.ws = [np.random.randn(y, x) for x , y in zip(sizes[:-1], sizes[1:])]
        self.bs = [np.random.randn(x, 1) for x in sizes[1:]]

    
    # sigmoid 函数
    def sigmoid(self,x):
        return 1.0/(1.0 + np.exp(-x))


    # sigmiid 的导数
    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1-self.sigmoid(x))

    
    # relu 函数
    def relu(self, x):
        return np.maximum(0,x)


    # relu 的导数
    def relu_primer(self ,x):
        row = x.shape[0]
        for i in range(row):
            if x[i][0] > 0:
                x[i][0] = 1
            else:
                x[i][0] = 0
        return x
    

    # 计算cost
    def cost(self,output,y):
        return (output - y)


    # sigmoid 向前传播
    def feed_forward_sigmoid(self, x):
        for b, w in zip(self.bs, self.ws):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    
    # relu 向前传播
    def feed_forward_relu(self, x):
        for b, w in zip(self.bs, self.ws):
            x = self.relu(np.dot(w, x) + b)
        return x
    

    # sigmoid 向后传播 获得更新的delta w b
    def feed_back_sigmoid(self, x, y):
        # 定义返回值这里的xxxb是g f ×（1-f）×（label-y） xxxw是b输入 （西瓜书里的）
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        nabla_w = [np.zeros(w.shape) for w in self.ws]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.bs, self.ws):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = self.sigmoid(z)
            activations.append(activation)
        
        delta = self.cost(activations[-1],y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.ws[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    
    # relu 向后传播 获得更新的delta w b
    def feed_back_relu(self, x, y):
        # 定义返回值这里的xxxb是g f ×（1-f）×（label-y） xxxw是b输入 （西瓜书里的）
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        nabla_w = [np.zeros(w.shape) for w in self.ws]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.bs, self.ws):
            z = np.dot(w, activation) + b
            zs.append(z)

            activation = self.relu(z)
            activations.append(activation)
        
        delta = self.cost(activations[-1],y) * self.relu_primer(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2,self.layers):
            z = zs[-l]
            sp = self.relu_primer(z)
            delta = np.dot(self.ws[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        

    # 批量进行传播更新参数 使用sigmoid
    def updae_batch_sigmoid(self, batch_data, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        nabla_w = [np.zeros(w.shape) for w in self.ws]
        for x, y in batch_data:
            delta_nabla_b, delta_nabla_w = self.feed_back_sigmoid(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.ws = [w - eta * nw /len(batch_data) for w, nw in zip(self.ws, nabla_w)]
        self.bs = [b - eta * nb /len(batch_data) for b, nb in zip(self.bs, nabla_b)]
    
    
    # 批量更新 relu
    def updae_batch_relu(self, batch_data, eta):
        nabla_b = [np.zeros(b.shape) for b in self.bs]
        nabla_w = [np.zeros(w.shape) for w in self.ws]
        for x, y in batch_data:
            delta_nabla_b, delta_nabla_w = self.feed_back_relu(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.ws = [w - eta * nw /len(batch_data) for w, nw in zip(self.ws, nabla_w)]
        self.bs = [b - eta * nb /len(batch_data) for b, nb in zip(self.bs, nabla_b)]    
    

    # SGD sigmoid
    def SGD_sigmoid(self,train_set,epochs,batch_size, eta):
        data_len = len(train_set)
        for j in range(epochs):
            random.shuffle(train_set)
            batchs = [train_set[k:k+batch_size] for k in range(0,data_len,batch_size)]
            for batch in batchs:
                self.updae_batch_sigmoid(batch,eta)
            print('epoch:' + str(j))

    
    # SGD relu
    def SGD_relu(self,train_set,epochs, batch_size,eta ):
        data_len = len(train_set)
        for j in range(epochs):
            random.shuffle(train_set)
            batchs = [train_set[k:k+batch_size] for k in range(0,data_len,batch_size)]
            for batch in batchs:
                self.updae_batch_relu(batch,eta)
            print('epoch:' + str(j))
    


    def predict(self,test_set):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_set]
        count = 0
        for x in test_results:
            if x[1][x[0]][0] == 1:
                count += 1
        return count
            

def loadKindDate(filename,n):
    dataSet = []
    labelSet = []
    with open(filename) as fp:
        lines = fp.readlines()
        for i in range(len(lines)):
            labelArr = np.zeros((10,1))
            labelArr[n] = 1
            labelSet.append(labelArr)
        for line in lines:
            linelist = line.strip().split()
            lineArr = np.zeros((len(linelist),1))
            for i in range(len(linelist)):
                lineArr[i][0] = float(linelist[i])
            dataSet.append(lineArr)
    return dataSet,labelSet
def loadAllDate():
    dataSet = []
    labelSet = []
    for i in range(10):
        filename = './data/'+str(i)+'.txt'        
        dataSet_temp,labelSet_temp = loadKindDate(filename,i)
        dataSet += dataSet_temp
        labelSet += labelSet_temp
    return dataSet,labelSet


if __name__ == '__main__':
    INPUT = 7
    OUTPUT = 10
    net = NeuralNet([INPUT,50,40,40,OUTPUT])
    
    dataSet,labelSet = loadAllDate()
    train_set = list(zip(dataSet,labelSet))
    #print(labelSet)
    net.SGD_relu(train_set, 100, 20, 0.1)
    for b,w in zip(net.bs,net.ws):
        print (b)
        print(w)
    
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        # Capture frame-by-frame
        cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
        crop_img = frame[100:300, 100:300]
        blurred = cv2.GaussianBlur(crop_img, (17, 17), 0)
        # cv2.imshow('blurred', blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([90, 40, 50])
        upper_skin = np.array([125, 130, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)       
        image, contours, hierarchy = cv2.findContours(mask.copy(),
                                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours):
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(cnt)
            drawing = np.zeros(crop_img.shape, np.uint8)
            epsilon = 0.03*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
            cv2.drawContours(drawing, [approx], 0, (0, 255, 0), 0)
            hu = cv2.HuMoments(cv2.moments(cnt))[:2]
            hu1 = round(hu[0][0] * 10, 2)
            hu2 = round(hu[1][0] * 1000, 2)
            L = cv2.arcLength(cnt, True)
            S = cv2.contourArea(cnt)
            LdivideS = round(S / L, 2)
            Lhull = cv2.arcLength(hull, True)
            leftmost = cnt[cnt[:, :, 0].argmin()][0][0]
            rightmost = cnt[cnt[:, :, 0].argmax()][0][0]
            BdivideLh = round((L - Lhull) / (rightmost - leftmost), 2)
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            count_concave = 0
            count_convex = 0
            longest = 0
            shortest = 1000000
            lastfinger = [0, 0]
            longestdist = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a = (end[0] - start[0])**2 + (end[1] - start[1])**2
                    b = (far[0] - start[0])**2 + (far[1] - start[1])**2
                    c = (end[0] - far[0])**2 + (end[1] - far[1])**2
                    if a < b + c:
                        longest = max(longest, b, c)
                        shortest = min(shortest, b, c)
                        count_concave += 1
            for i in range(0, len(approx)):
                pre = i - 1
                nex = i + 1
                if i == 0:
                    pre = len(approx) - 1
                elif i == len(approx) - 1:
                    nex = 0
                if approx[i][0][1] < approx[pre][0][1] and approx[i][0][1] < approx[nex][0][1]:
                    a = math.sqrt((approx[pre][0][0] - approx[nex][0][0])**2 + (approx[pre][0][1] - approx[nex][0][1])**2)
                    b = math.sqrt((approx[i][0][0] - approx[nex][0][0])**2 + (approx[i][0][1] - approx[nex][0][1])**2)
                    c = math.sqrt((approx[pre][0][0] - approx[i][0][0])**2 + (approx[pre][0][1] - approx[i][0][1])**2)
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                    if angle < 50:
                        count_convex += 1
                        if lastfinger[0]:
                            longestdist = max(longestdist, (approx[i][0][
                                                0] - lastfinger[0])**2 + (approx[i][0][1] - lastfinger[1])**2)
                        lastfinger = approx[i][0]
            # print(hu1, hu2, count_concave, count_convex, longest / 1000, shortest / 1000, math.sqrt(longestdist) / 10)
            features = [hu1, hu2, count_concave, count_convex, longest /
                        1000, shortest / 1000, math.sqrt(longestdist) / 10]
            new = np.zeros((7,1))
            for hh in range(7):
                new[hh][0] = features[hh] 
            #print(features)
            #cv2.imshow('drawing', drawing)
            res = net.feed_forward_relu(new)
            #print(res)
            index = res.argmax()

            print(index)
        cv2.imshow('hh',mask)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break            
