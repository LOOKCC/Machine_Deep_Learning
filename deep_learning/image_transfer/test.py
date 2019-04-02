'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

import torch.utils.model_zoo as model_zoo
import math

import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()

cnn = models.vgg19(pretrained=True).features
model = nn.Sequential()
if use_cuda:
    cnn = cnn.cuda()
    model = model.cuda()



conv_layer = 1
pool_layer = 1
relu_layer = 1
for layer in list(cnn):
    if isinstance(layer,nn.Conv2d):
        name = 'conv_'+str(conv_layer)
        model.add_module(name,layer)
        print(name)
        conv_layer += 1
    if isinstance(layer,nn.ReLU):
        name = 'relu_'+str(relu_layer)
        model.add_module(name,layer)
        print(name)
        relu_layer += 1
    if isinstance(layer,nn.MaxPool2d):
        name = 'pool_'+str(pool_layer)
        model.add_module(name,layer)
        print(name)
        pool_layer += 1


class myvgg(nn.Module):
    def __init__(self,mymodel):
        super(myvgg, self).__init__()
        self.features = mymodel

    def forward(self, x):
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        result_1 = x        
        x = self.pool_1(x)
        x = self.relu_3(self.conv_3(x))
        x = self.relu_4(self.conv_4(x))
        result_2 = x        
        x = self.pool_2(x)
        x = self.relu_5(self.conv_5(x))
        x = self.relu_6(self.conv_6(x))
        x = self.relu_7(self.conv_7(x))
        x = self.relu_8(self.conv_8(x))
        result_3 = x
        x = self.pool_3(x)
        x = self.relu_9(self.conv_9(x))
        x = self.relu_10(self.conv_10(x))
        x = self.relu_11(self.conv_11(x))
        x = self.relu_12(self.conv_12(x))
        result_4 = x
        x = self.pool_4(x)
        x = self.relu_13(self.conv_13(x))
        x = self.relu_14(self.conv_14(x))
        x = self.relu_15(self.conv_15(x))
        x = self.relu_16(self.conv_16(x))
        result_5 = x
        x = self.pool_5(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        return [result_1,result_2,result_3,result_4,result_5]

my_vgg = myvgg(model)





class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19,self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d (kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    
        self.pool2 = nn.MaxPool2d (kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d (kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d (kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool5 = nn.MaxPool2d (kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

 
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        result1 = x        
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        result2 = x        
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.relu(self.conv3_4(x))
        result3 = x
        x = self.pool3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.relu(self.conv4_4(x))
        result4 = x
        x = self.pool4(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.relu(self.conv5_4(x))
        result5 = x
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return [result1,result2,result3,result4,result5]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math

import numpy as np
from PIL import Image

use_cuda = torch.cuda.is_available()

cnn = models.vgg19(pretrained=True).features
mymodel = nn.Sequential()
if use_cuda:
    cnn = cnn.cuda()
    mymodel = mymodel.cuda()

conv_layer = 1
pool_layer = 1
relu_layer = 1
for layer in list(cnn):
    if isinstance(layer,nn.Conv2d):
        name = 'conv_'+str(conv_layer)
        mymodel.add_module(name,layer)
        #print(name)
        conv_layer += 1
    if isinstance(layer,nn.ReLU):
        name = 'relu_'+str(relu_layer)
        mymodel.add_module(name,layer)
        #print(name)
        relu_layer += 1
    if isinstance(layer,nn.MaxPool2d):
        name = 'pool_'+str(pool_layer)
        mymodel.add_module(name,layer)
        #print(name)
        pool_layer += 1

print(mymodel)

class myvgg(nn.Module):
    def __init__(self,mymodel):
        super(myvgg, self).__init__()
        self.model = mymodel

    def forward(self, x):
        x = nn.relu_1(nn.conv_1(x))
        x = nn.relu_2(nn.conv_2(x))
        result_1 = x        
        x = nn.pool_1(x)
        x = nn.relu_3(nn.conv_3(x))
        x = nn.relu_4(nn.conv_4(x))
        result_2 = x        
        x = nn.pool_2(x)
        x = nn.relu_5(nn.conv_5(x))
        x = nn.relu_6(nn.conv_6(x))
        x = nn.relu_7(nn.conv_7(x))
        x = nn.relu_8(nn.conv_8(x))
        result_3 = x
        x = nn.pool_3(x)
        x = nn.relu_9(nn.conv_9(x))
        x = nn.relu_10(nn.conv_10(x))
        x = nn.relu_11(nn.conv_11(x))
        x = nn.relu_12(nn.conv_12(x))
        result_4 = x
        x = nn.pool_4(x)
        x = nn.relu_13(nn.conv_13(x))
        x = nn.relu_14(nn.conv_14(x))
        x = nn.relu_15(nn.conv_15(x))
        x = nn.relu_16(nn.conv_16(x))
        result_5 = x
        x = nn.pool_5(x)
        return [result_1,result_2,result_3,result_4,result_5]


def load_data():
    content_image = Image.open('./data/content.jpg').getdata()
    style_image = Image.open('./data/style.jpg').getdata()
    content_image =  np.reshape(content_image,(3,300,300))
    content_image = torch.from_numpy(content_image)
    style_image = np.reshape(style_image,(3,300,300))
    style_image = torch.from_numpy(style_image)
    # print(content_image.shape)
    # print(style_image.shape)
    return content_image,style_image


def gram(image):
    k,i,j = image.size()
    result = np.zeros((k,k))
    for k_row in range(k):
        for k_column in range(k):
            total = 0
            for i1 in range(i):
                for j1 in range(j):
                    total += image[k_row][i1][j1] * image[k_column][i1][j1]
            result[k_row][k_column] = total
    return result


def train(content_image, style_image, alpha, beta, m_lr,epoch):
    net = myvgg(mymodel)
    if use_cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        net = net.cuda()
    content_feat = net(content_image)
    style_feat = net(style_feat)
    # 根据论文，这里选择第四层的输出作为内容的部分的优化目标
    content_aim = Variable(content_feat[3].data)
    # 根据论文，这里采用每一层的gram矩阵作为优化的目标
    style_aim = [Variable(gram(x).data) for x in style_feat]
    # 跟据论文，采用噪声或者content或者style优化是一样的
    input_image = Variable(content_image.data)
    # github用的Adam
    optimizer = optim.Adam(input_image,lr = m_lr )
    # 设置误差计算方法为欧式距离
    calculate_loss = torch.nn.MSELoss()
    for i in range(epoch):
        optimizer.zero_grad()
        predict = net(input_image)
        content_loss = alpha * calculate_loss(predict[3],content_aim)

        style_loss = 0
        for j in range(5):
            input_gram = gram(predict[j])
            style_loss += calculate_loss(input_gram,style_aim[j])
        style_loss *= beta

        total_loss = content_loss + style_loss
        total_loss.backward()
        optimizer.step()
        if i%10 == 0:
            print(str(i) + 'loss: '+str(total_loss))
    return input



if __name__ == '__main__':
    content_image, style_image = load_data()
    image = train(content_image,style_image,0.1,0.1,0.1,100)





