import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet,self).__init__()
        self.cov1_1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.cov1_2 = nn.Conv2d(64,128,kernel_size=1,stride=1,padding=0)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.cov1_3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.bn1_3 = nn.BatchNorm2d(128)

        self.cov2_1 = nn.Conv2d(128,256,kernel_size=1,stride=1,padding=0)
        self.bn2_1 = nn.BatchNorm2d(256)
        self.cov2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.cov2_3 = nn.Conv2d(256,256, kernel_size=1, stride=1,padding=0)
        self.bn2_3 = nn.BatchNorm2d(256)
        self.cov2_4 = nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1)
        self.bn2_4 = nn.BatchNorm2d(256)
    
        self.cov3_1 = nn.Conv2d(256,512,kernel_size=1,stride=1,padding=0)
        self.bn3_1 = nn.BatchNorm2d(512)
        self.cov3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)
        self.cov3_3 = nn.Conv2d(512,512, kernel_size=1, stride=1,padding=0)
        self.bn3_3 = nn.BatchNorm2d(512)
        self.cov3_4 = nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1)
        self.bn3_4 = nn.BatchNorm2d(512)

        self.cov4_1 = nn.Conv2d(512,256, kernel_size=1,stride=1,padding=0)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.cov4_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,padding=0)
        self.bn4_2 = nn.BatchNorm2d(128)
        self.cov4_3 = nn.Conv2d(128,64, kernel_size=1, stride=1,padding=0)
        self.bn4_3 = nn.BatchNorm2d(64)
        self.cov4_4 = nn.Conv2d(64,3, kernel_size=1, stride=1,padding=0)
        self.bn4_4 = nn.BatchNorm2d(3)


    def forward(self,x,y):
        x = F.relu(self.bn1_1(self.cov1_1(x)))
        x = F.relu(self.bn1_2(self.cov1_2(x)))
        x = F.relu(self.bn1_3(self.cov1_3(x)))

        x = F.relu(self.bn2_1(self.cov2_1(x)))
        x = F.relu(self.bn2_2(self.cov2_2(x)))
        x = F.relu(self.bn2_3(self.cov2_3(x)))
        x = F.relu(self.bn2_4(self.cov2_4(x)))
        
        x = F.relu(self.bn3_1(self.cov3_1(x)))
        x = F.relu(self.bn3_2(self.cov3_2(x)))
        x = F.relu(self.bn3_3(self.cov3_3(x)))
        x = F.relu(self.bn3_4(self.cov3_4(x)))
        
        x = F.relu(self.bn4_1(self.cov4_1(x)))
        x = F.relu(self.bn4_2(self.cov4_2(x)))
        x = F.relu(self.bn4_3(self.cov4_3(x)))
        x = F.relu(self.bn4_4(self.cov4_4(x)))

        for i in range(224):
            for j in range(224):
                y[0,0,i,j] = 0.2125*x[0,0,i,j] + 0.7154*x[0,1,i,j] + 0.0721*x[0,2,i,j]
        return x, y
    
