#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


cfg = {
    11: (64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'),
    13: (64, 64, 'M', 128, 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M'),
    16: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
         512, 512, 512, 'M', 512, 512, 512, 'M'),
    19: (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
         512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'),
}


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
        )
        # self.classifier = nn.Linear(512, 10)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.classifier(x)
        return x

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
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, i, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(i))
            layers.append(nn.ReLU(inplace=True))
            in_channels = i
    return nn.Sequential(*layers)


def train(vgg_num, trainloader, net=None, start_epoch=0, epoch_num=2):
    if not net:
        net = VGG(make_layers(cfg[vgg_num], True))
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
    vgg_num = 16
    start_epoch = 0
    net = None
    epoch_num = 10
    train_num_each_test = 2
    file_name = './net/vgg%d_net.pkl' % (vgg_num, )
    if os.path.exists(file_name):
        (start_epoch, net) = pickle.load(open(file_name, 'rb'))
    for i in range(epoch_num // train_num_each_test):
        net = train(vgg_num, trainloader, net,
                    start_epoch, train_num_each_test)
        print('----  Saving.. ----')
        start_epoch += train_num_each_test
        pickle.dump((start_epoch, net), open(file_name, 'wb'))
        # net = pickle.load(open(file_name, 'rb'))
        print('---- Testing.. ----')
        test(net, testloader, classes)
