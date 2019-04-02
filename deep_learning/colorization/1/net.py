import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet,self).__init__()
        # low
        self.lowconv1 = nn.Conv2d(1,64,kernel_size=3,stride=2,padding=1)
        self.lowbn1 = nn.BatchNorm2d(64)
        self.lowconv2 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
        self.lowbn2 = nn.BatchNorm2d(128)
        self.lowconv3 = nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1)
        self.lowbn3 = nn.BatchNorm2d(128)
        self.lowconv4 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1)
        self.lowbn4 = nn.BatchNorm2d(256)
        self.lowconv5 = nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1)
        self.lowbn5 = nn.BatchNorm2d(256)
        self.lowconv6 = nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1)
        self.lowbn6 = nn.BatchNorm2d(512)
        # middle
        self.middleconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=1)
        self.middlebn1 = nn.BatchNorm2d(512)
        self.middleconv2 = nn.Conv2d(512,256, kernel_size=3, stride=1,padding=1)
        self.middlebn2 = nn.BatchNorm2d(256)
        # global
        self.globalconv1 = nn.Conv2d(512,512, kernel_size=3, stride=2,padding=1)
        self.globalbn1 = nn.BatchNorm2d(512)
        self.globalconv2 = nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1)
        self.globalbn2 = nn.BatchNorm2d(512)
        self.globalconv3 = nn.Conv2d(512,512, kernel_size=3, stride=2,padding=1)
        self.globalbn3 = nn.BatchNorm2d(512)
        self.globalconv4 = nn.Conv2d(512,512, kernel_size=3, stride=1,padding=1)
        self.globalbn4 = nn.BatchNorm2d(512)
        self.globalfc1 = nn.Linear(25088,1024)
        self.globalbn5 = nn.BatchNorm2d(1024)
        self.globalfc2 = nn.Linear(1024,512)
        self.globalbn6 = nn.BatchNorm2d(512)
        self.globalfc3 = nn.Linear(512,256)
        self.globalbn7 = nn.BatchNorm2d(256)
        # color
        self.colorfc1 = nn.Linear(512,256)
        self.colorbn1 = nn.BatchNorm2d(256)
        self.colorconv1 = nn.Conv2d(256,128, kernel_size=3, stride=1,padding=1)
        self.colorbn2 = nn.BatchNorm2d(128)
        self.colorconv2 = nn.Conv2d(128,64, kernel_size=3, stride=1,padding=1)
        self.colorbn3 = nn.BatchNorm2d(64)
        self.colorconv3 = nn.Conv2d(64,64, kernel_size=3, stride=1,padding=1)
        self.colorbn4 = nn.BatchNorm2d(64)
        self.colorconv4 = nn.Conv2d(64,32, kernel_size=3, stride=1,padding=1)
        self.colorbn5 = nn.BatchNorm2d(32)
        self.colorconv5 = nn.Conv2d(32,2, kernel_size=3, stride=1,padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self,x):
        x1 = F.relu(self.lowbn1(self.lowconv1(x)))                         
        x1 = F.relu(self.lowbn2(self.lowconv2(x1)))        
        x1 = F.relu(self.lowbn3(self.lowconv3(x1)))
        x1 = F.relu(self.lowbn5(self.lowconv4(x1)))
        x1 = F.relu(self.lowbn4(self.lowconv5(x1)))
        x1 = F.relu(self.lowbn6(self.lowconv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.lowbn1(self.lowconv1(x)))                         
            x2 = F.relu(self.lowbn2(self.lowconv2(x2)))        
            x2 = F.relu(self.lowbn3(self.lowconv3(x2)))
            x2 = F.relu(self.lowbn5(self.lowconv4(x2)))
            x2 = F.relu(self.lowbn4(self.lowconv5(x2)))
            x2 = F.relu(self.lowbn6(self.lowconv6(x2)))
        
        x1 = F.relu(self.middlebn1(self.middleconv1(x1)))
        x1 = F.relu(self.middlebn2(self.middleconv2(x1)))

        x2 = F.relu(self.globalbn1(self.globalconv1(x2)))
        x2 = F.relu(self.globalbn2(self.globalconv2(x2)))
        x2 = F.relu(self.globalbn3(self.globalconv3(x2)))
        x2 = F.relu(self.globalbn4(self.globalconv4(x2)))
        x2 = x2.view(-1,25088) # 7 7 512
        x2 = F.relu(self.globalbn5(self.globalfc1(x2)))
        x2 = F.relu(self.globalbn6(self.globalfc2(x2)))
        x2 = F.relu(self.globalbn7(self.globalfc3(x2)))

        w = x1.shape[2]
        h = x1.shape[3]
        x2 = x2.unsqueeze(2).unsqueeze(2).expand_as(x1)
        x1_x2 = torch.cat((x1,x2),1)
        x1_x2 = x1_x2.permute(2,3,0,1).contiguous()
        # 28 28 5 512
        x1_x2 = x1_x2.view(-1,512)
        x1_x2 = self.colorbn1(self.colorfc1(x1_x2))
        x1_x2 = x1_x2.view(w,h,-1,256)
        x1_x2 = x1_x2.permute(2,3,0,1)
        x1_x2 = F.relu(self.colorbn2(self.colorconv1(x1_x2)))
        x1_x2 = self.upsample(x1_x2)        
        x1_x2 = F.relu(self.colorbn3(self.colorconv2(x1_x2)))
        x1_x2 = F.relu(self.colorbn4(self.colorconv3(x1_x2)))
        x1_x2 = self.upsample(x1_x2)        
        x1_x2 = F.sigmoid(self.colorbn5(self.colorconv4(x1_x2)))
        x1_x2 = self.upsample(self.colorconv5(x1_x2))

        return x1_x2 
    
