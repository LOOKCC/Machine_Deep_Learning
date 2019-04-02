import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os 
import argparse

from net import *
from utils import progress_bar
from torch.autograd import Variable
# 命令行参数 --lr 是 学习速率 --resume 是使用已经保存的模型接着训练
parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
parser.add_argument('--lr',default=0.1,type=float,help='learning rate')
parser.add_argument('--resume','-r',action='store_true',help='resume form checkpoint')
args = parser.parse_args()
#  初始化一些变量，包括是否使用cuda 还有最好速率 开始的迭代次数
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch = 0

# 准备数据
# RandomCrop 随机切割并周围补齐
# RandomHorizontalFlip 以0.5 的概率随机水平翻转图像
# ToTensor 转化为tensor
# Normalize 归一化 用均值和 标准差对图片进行归一化
# (0.4941,0.4822,0.4465),(0.2023,0.1994,0.2010) this is from utils/get_mean_and_std 
print('Preparing data:')
transforms_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4941,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4941,0.4822,0.4465),(0.2023,0.1994,0.2010)),
])
# 加载数据 num_wokers 是进程数
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=False,transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=128, shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transforms_test)
testloader = torch.utils.data.DataLoader(testset,batch_size=100, shuffle=False,num_workers=2)
# 分类结果
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'shop', 'truck')
# 根据命令行参数确定各种不同的情况
# 如果是继续训练 找到路径 加载数据
if args.resume:
    print('resume data:')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoinrt in this dir'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:                #否则的话选择模型
    print('model:VGG19')
    net = VGG('VGG19')
# 是否使用GPU
if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 使用随机梯度下降 动量因子，权重衰减（L2正则化）
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0,weight_decay=5e-4)
# 训练
def train(epoch):
    print('\nEpch: %d' % epoch)
    # 设置为训练模式
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(trainloader):
        if use_cuda:
            inputs,targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        # 计算总的 loss total correct
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # 按照进度条 输出 各类信息
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: % .3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct,total))


def test(epoch):
    global best_acc
    # 进入评估模式
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs,targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 排除子图提高效率
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs,targets)

        test_loss += loss.data[0]
        _,predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader),'Loss: %.3f | Acc: % .3f%% (%d/%d)'
            % (test_loss/(batch_idx+1),100.*correct/total, correct,total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('save the best model')
        state = {
            'net' : net.module if use_cuda else net,
            'acc' : acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+3):
    train(epoch)
    test(epoch)
        
