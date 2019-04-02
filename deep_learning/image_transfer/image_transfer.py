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
print(use_cuda)
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
        self.content_return = ['26'] 
        self.style_return = ['3','8','17','26','35']

    def forward(self, x):
        #x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        #return x
        content_outputs = []
        style_outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.content_return:
                content_outputs += [x]
            if name in self.style_return:
                style_outputs += [x]
        return content_outputs, style_outputs

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    content_image = Image.open('./data/content.jpg')
    style_image = Image.open('./data/style.jpg')
    content_image = transform(content_image).unsqueeze(0)
    style_image = transform(style_image).unsqueeze(0)
    content_image = content_image.type(torch.FloatTensor)
    style_image = style_image.type(torch.FloatTensor)
    print(content_image.size())
    print(style_image.size())
    return content_image,style_image


def gram(image):
    '''
    _,k,i,j = image.size()
    result = torch.FloatTensor(k,k)
    for k_row in range(k):
        for k_column in range(k):
            total = 0
            for i1 in range(i):
                for j1 in range(j):
                    total += image[0][k_row][i1][j1] * image[0][k_column][i1][j1]
            print(total[0])
            result[k_row][k_column] = total.type(torch.FloatTensor)
    return result
    '''
    _,c,h,w = image.size()
    temp = image.view(c, h*w)
    return torch.mm(temp, temp.t())


def train(content_image, style_image, alpha, beta, m_lr,epoch):
    net = vgg19(True)
    print('model ok')
    if use_cuda:
        content_image = content_image.cuda()
        style_image = style_image.cuda()
        net = net.cuda()
    # 根据论文，这里选择第四层的输出作为内容的部分的优化目标
    # 根据论文，这里采用每一层的gram矩阵作为优化的目标
    # 跟据论文，采用噪声或者content或者style优化是一样的,这里使用内容图像进行初始化
    temp = torch.rand(1,3,300,300)
    temp = temp.cuda()
    # 使用随机
    # input_image = Variable(temp.clone(), requires_grad = True)    
    # 使用content
    input_image = Variable(content_image.clone(), requires_grad = True)
    # 使用style
    # input_image = Variable(style_image.clone(), requires_grad = True)
    content_aim,_ = net(Variable(content_image,requires_grad = True))
    _,style_aim = net(Variable(style_image, requires_grad = True))
    # github用的Adam
    optimizer = optim.Adam([input_image],lr = m_lr )
    # 设置误差计算方法为欧式距离
    #calculate_loss = torch.nn.MSELoss()
    for i in range(epoch):
        optimizer.zero_grad()
        predict_content, predict_style = net(input_image)
        # content_loss = alpha * calculate_loss(predict_content[0].data,content_aim.data)
        content_loss = alpha * torch.mean((predict_content[0].data - content_aim[0].data)**2)

        style_loss = 0
        for j in range(5):
            _, c, h, w = predict_style[j].size()
            input_gram = gram(predict_style[j])
            style_gram = gram(style_aim[j])
            style_loss += torch.mean((input_gram - style_gram)**2)/(c*h*w)
        style_loss *= beta

        total_loss = content_loss + style_loss
        total_loss.backward(retain_graph = True)
        optimizer.step()
        if i%10 == 0:
            print(str(i) + 'loss: '+str(total_loss))
    return input_image



if __name__ == '__main__':
    content_image, style_image = load_data()
    print('load over')
    image = train(content_image,style_image,1,1000,0.05,1000)
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    image = image.clone().cpu().squeeze()
    image = denorm(image.data).clamp_(0, 1)
    torchvision.utils.save_image(image, 'output.png')
    


# 测试的几个结论
# 0.最后接近最后的收敛点时会来回波动，500 迭代足够
# 1.使用content初始化，a/b = 1/1000 0.1 800  正常 output
# 2.使用style初始化，  a/b = 1/1000 0.1 500    a/b = 1000/1 0.1 500    出现梯度不下降的情况 1/1000 得到style 1000/1 同左
# 3.使用噪声初始化，    a/b = 1/1000 0.1 800    a/b = 1/1000 0.1 800   全部为纹理  output0 output1


# 猜测
# 从上面的实验可以看出，content loss部分基本没有作用，于是我做了下一个实验
# 0,1000,0.1,800  得到了和output一样的结果。