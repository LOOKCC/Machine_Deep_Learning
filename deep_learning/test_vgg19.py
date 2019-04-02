import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms

print(torch.cuda.is_available())

transform=transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root = './data', train = True, download = False, transform = transform)


trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = False, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = False, transform = transform)
 
testloader = torch.utils.data.DataLoader(testset, batch_size = 128, 
                                          shuffle = False, num_workers = 2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1) 
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1) 
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)        

        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1) 
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1) 
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1) 
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1) 
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1) 
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1) 
        self.conv5_4 = nn.Conv2d(512, 512, 3, padding=1) 
        self.pool5 = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU(inplace=True)
        #self.batch = nn.BatchNorm2d()
        self.classifier = nn.Linear(512,10)
 
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.pool1(x)
        #print(x.size())
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        #print(x.size())
        # 不一样
        x = self.conv3_1(x)
        x = self.relu(x)
        # 不一样
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        # 不一样        
        x = self.conv3_4(x)
        
        x = self.relu(x)
        # 一样了
        x = self.pool3(x)
        #print(x.size())
        x = self.conv4_1(x)
        
        x = self.relu(x)
        x = self.conv4_2(x)
        
        x = self.relu(x)
        x = self.conv4_3(x)
        
        x = self.relu(x)
        x = self.conv4_4(x)
        
        x = self.relu(x)
        x = self.pool4(x)
        #print(x.size())                
        x = self.conv5_1(x)
        
        x = self.relu(x)
        x = self.conv5_2(x)
        
        x = self.relu(x)
        x = self.conv5_3(x)
        
        x = self.relu(x)
        x = self.conv5_4(x)
        
        x = self.relu(x)
        x = self.pool5(x)
        #print(x.size())
        x = x.view(-1, 512)
        #print(x)
        #print(x.size())        
        x = self.classifier(x)
        #print(x)
        #print(x.size())
        return x
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
 
    def forward(self, x):
        #print('begin')
        #print(x.size())
        x = self.conv1(x)
        #print(x.size())
        x = self.pool(F.relu(x))
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.pool(F.relu(x))
        #print(x.size())
        #print('end')                
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16*5*5)
        #print(x.size())        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.size())
        return x
'''
net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss() #叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0,weight_decay=5e-4)  #使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9
 
for epoch in range(2): # 遍历数据集两次
     
    running_loss = 0.0
    #enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0): 
        #print(i)
        # get the inputs
        inputs, labels = data   #data的结构是：[4x3x32x32的张量,长度4的张量]

        #inputs = inputs.cuda()
        #labels = labels.cuda()

        # wrap them in Variable
        #inputs, labels = Variable(inputs), Variable(labels)  #把input数据从tensor转为variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
         
        # zero the parameter gradients
        optimizer.zero_grad() #将参数的grad值初始化为0
         
        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.size())
        # print(labels.size())
        #print(labels)
        loss = criterion(outputs, labels) #将output和labels使用叉熵计算损失
        loss.backward() #反向传播
        optimizer.step() #用SGD更新参数
         
        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]  #loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        #print(i)
        if i % 50 == 49: # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 50))
            running_loss = 0.0
 
print('Finished Training')
 
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images).cuda())
    #print outputs.data
    _, predicted = torch.max(outputs.data, 1)  #outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()   #两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

