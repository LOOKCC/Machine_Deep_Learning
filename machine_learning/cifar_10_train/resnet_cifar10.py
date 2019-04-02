#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import pickle
import math
from torch.autograd import Variable


def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print('==== Loading data.. ====')
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


cfg = {
    18: (0, 2, 2, 2, 2),
    34: (0, 3, 4, 6, 3),
    50: (1, 3, 4, 6, 3),
    101: (1, 3, 4, 23, 3),
    152: (1, 3, 8, 36, 3),
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.relu(self.bn1_2(self.conv1(x)))
        out = self.relu(self.bn1_2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_layers, in_channels=3, out_classes=10):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, num_layers[0])
        self.layer2 = self._make_layer(block, 128, num_layers[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layers[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layers[3], 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, out_classes)
        self._initialize_weights()

    def _make_layer(self, block, planes, num_layer, stride=1):
        shortcut = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, shortcut)]
        self.inplanes = planes * block.expansion
        for i in range(num_layer-1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def train(res_num, trainloader, net=None, start_epoch=0, epoch_num=2):
    if not net:
        block = Bottleneck if cfg[res_num][0] else BasicBlock
        net = ResNet(block, cfg[res_num][1:])
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=5e-4)
    print('====   Training..   ====')
    net.train()

    for epoch in range(start_epoch, start_epoch+epoch_num):
        print('<---- epoch: %d ---->' % (epoch, ))
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch+1, i+1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')
    return net


def test(net, testloader, classes):
    correct = 0
    total = 0
    class_correct = [0.0] * 10
    class_total = [0.0] * 10
    net.eval()
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images.cuda()))
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        result = (predicted == labels.cuda())
        correct += result.sum()
        c = result.squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    trainloader, testloader, classes = get_data()
    res_num = 18
    start_epoch = 0
    net = None
    epoch_num = 10
    train_num_each_test = 2
    file_name = './net/resnet%d.pkl' % (res_num, )
    if os.path.exists(file_name):
        (start_epoch, net) = pickle.load(open(file_name, 'rb'))
    for i in range(epoch_num // train_num_each_test):
        net = train(res_num, trainloader, net,
                    start_epoch, train_num_each_test)
        print('----  Saving.. ----')
        start_epoch += train_num_each_test
        pickle.dump((start_epoch, net), open(file_name, 'wb'))
        # net = pickle.load(open(file_name, 'rb'))
        print('---- Testing.. ----')
        test(net, testloader, classes)
