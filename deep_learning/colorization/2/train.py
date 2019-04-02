import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import net
import utli

# y = 0.2125 R + 0.7154 G + 0.0721 B

#dir = '/input/data/VOC2012/JPEGImages/'
total_image = 17125
dir = '/home/wangzhihao/code/AI_new_man/colorization/VOCdevkit/VOC2012/JPEGImages/'
learning_rate = 0.1
epoch = 1
use_cuda = torch.cuda.is_available()
filenames_trian = utli.getdir(0,10)
filenames_test = utli.getdir(17110,17120)
loader = utli.dataloader(filenames_trian,dir,1)
test_image = utli.dataloader(filenames_test,dir,1)
print('load ok')
colornet = net.ColorNet()
if use_cuda:
    colornet = colornet.cuda() 
mes_loss = nn.MSELoss()
optimizer = optim.Adam(colornet.parameters(),lr = learning_rate)
def train():
    for i in range(epoch):
        for batch_idx,(inputs,origin) in enumerate(loader):
            y = torch.zeros(1,1,224,224)
            if use_cuda:
                inputs,origin,y = inputs.cuda(), origin.cuda(), y.cuda()
            optimizer.zero_grad()
            inputs = Variable(inputs)
            print(inputs.shape)
            origin = Variable(origin)
            y = Variable(y)
            outputs,twice_gray = colornet(inputs,y)
            print(outputs.shape)
            rgb_loss = mes_loss(outputs,origin)
            gray_loss = mes_loss(twice_gray,inputs)
            loss = rgb_loss + gray_loss
            loss.backward()
            optimizer.step()
        if i%10 == 0:
            print(str(i+1) + ' loss: '+str(loss))


def test():
    for batch_idx,(inputs,origin) in enumerate(test_image):
        y = torch.zeros(1,1,224,224)
        if use_cuda:
            inputs,origin,y = inputs.cuda(), origin.cuda(), y.cuda()
        inputs = Variable(inputs)
        origin = Variable(origin)
        y = Variable(y)
        outputs,twice_gray = colornet(inputs,y)
        rgb_loss = mes_loss(outputs,origin)
        gray_loss = mes_loss(twice_gray,inputs)
        loss = rgb_loss + gray_loss
        print(loss)
        for i in range(len(outputs)):
            img = transforms.ToPILImage()(outputs[i])
            img.show()

if __name__ == '__main__':
    train()
    print('train ok')


    

