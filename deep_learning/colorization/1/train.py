import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import net
import loaddata

dir = './VOCdevkit/VOC2012/JPEGImages/'
learning_rate = 0.1
epoch = 1
use_cuda = torch.cuda.is_available()
filenames = loaddata.getdir(0,10000)
loader = loaddata.dataloader(filenames,dir,32)
print('load ok')
colornet = net.ColorNet()
if use_cuda:
    colornet = colornet.cuda() 
mes_loss = nn.MSELoss()
optimizer = optim.Adam(colornet.parameters(),lr = learning_rate)
for i in range(epoch):
    for batch_idx,(inputs,targets,origin) in enumerate(loader):
        if use_cuda:
            inputs,targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = colornet(inputs)
        loss = mes_loss(outputs,targets) 
        loss.backward()
        optimizer.step()
    if i%10 == 0:
        print(str(i+1) + ' loss: '+str(loss))
