#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchvision
import pickle
import os
from torchvision import transforms
from torch.autograd import Variable
from torch import nn


def get_data():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()
    train_set = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=train_transform)
    train_set = torch.utils.data.DataLoader(train_set, batch_size=128,
                                              shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=test_transform)
    test_set = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return train_set, test_set, classes


class FakeVGG(nn.Module):
    def __init__(self, vgg_num, in_channels, out_channels):
        super(FakeVGG, self).__init__()
        cfg = {
            11: (64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096),
            13: (64, 64, 128, 128, 256, 256, 512, 512, 512, 512, 4096, 4096),
            16: (64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 4096, 4096),
            19: (64, 64, 128, 128, 256, 256, 256, 256,
                 512, 512, 512, 512, 512, 512, 512, 512, 4096, 4096),
        }
        part_cfg = {
            11: (1, 1, 2, 2, 2),
            13: (2, 2, 2, 2, 2),
            16: (2, 2, 3, 3, 3),
            19: (2, 2, 4, 4, 4),
        }
        self.fc_cfg = (4096, 4096)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net_struct = cfg[vgg_num]
        self.part_cfg = part_cfg[vgg_num]
        self.conv, self.fc = self._make_layers()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def _make_layers(self):
        conv = []
        fc = []
        in_channels = self.in_channels
        count = 0
        for conv_num in self.part_cfg:
            for i in range(conv_num):
                out_channels = self.net_struct[count]
                conv.append(nn.Conv2d(in_channels, out_channels, 3, padding=1).cuda())
                conv.append(nn.BatchNorm2d(out_channels).cuda())
                conv.append(nn.ReLU(True).cuda())
                in_channels = out_channels
                count += 1
            conv.append(nn.MaxPool2d(2, 2).cuda())
        for out_channels in self.net_struct[count:-1]:
            fc.append(nn.Linear(in_channels, out_channels).cuda())
            fc.append(nn.ReLU(True).cuda())
            fc.append(nn.Dropout().cuda())
            in_channels = out_channels
        fc.append(nn.Linear(in_channels, self.out_channels).cuda())
        return nn.Sequential(*conv), nn.Sequential(*fc)


def train(train_set, vgg_num, start_epoch, epoch_num, net=None):
    if net is None:
        net = FakeVGG(vgg_num, 3, 10)
    net.cuda()
    net.train()
    print(net)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(start_epoch, start_epoch+epoch_num):
        print('---- epoch: %d ----' % epoch)
        running_loss = 0.0
        for (i, data) in enumerate(train_set, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                            (epoch+1, i+1, running_loss / 50))
                running_loss = 0.0
    print('Finished Training')
    return net


def test(net, test_set, classes):
    correct = 0
    total = 0
    class_correct = [0.0] * 10
    class_total = [0.0] * 10
    net.test()
    for data in test_set:
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
    train_set, test_set, classes = get_data()
    vgg_num = 16
    filename = './net/fake_vgg%dnet.pkl' % vgg_num
    start_epoch = 0
    epoch_num = 2
    net = None
    if os.path.exists(filename):
        (start_epoch, net) = pickle.load(open(filename, 'rb'))
    net = train(train_set, vgg_num, start_epoch, epoch_num, net)
    pickle.dump((start_epoch, net), open(filename, 'wb'))
    test(net, test_set, classes)