import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models
from torch.autograd import Variable

class L8(nn.Module):

    def __init__(self):
        super(L8, self).__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=11,
            stride=1,
            padding=5# if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1  
            )
        self.conv2=nn.Conv2d(32,64,3,1,1)
        self.conv3=nn.Conv2d(64,64,3,1,1)
        self.conv4=nn.Conv2d(64,64,3,1,1)
        self.conv5=nn.Conv2d(64+32+32,64,1,1,0)
        self.conv6=nn.Conv2d(64,64,5,1,2)
        self.conv7=nn.Conv2d(64+32,128,1,1,0)
        self.conv8=nn.Conv2d(128,128*4,1,1,0)
        self.conv9=nn.Conv2d(128,3,5,1,2)
        self.downsampling=nn.Conv2d(32, 32, 4, 2, 1)
        self.upsampling=nn.PixelShuffle(2)
        #self.relu=nn.ReLU(inplace=False)#not sure

    def forward(self,x,vgg_feature):
        x1=F.relu(self.conv1(x))
        x1=F.relu(self.downsampling(x1))
        x=F.relu(self.conv2(x1))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))

        x2=self.upsampling(self.upsampling(vgg_feature))
        [a1,b1,c1,d1]=x1.size()
        [a2,b2,c2,d2]=x2.size()     
        diff_c=c1-c2
        diff_d=d1-d2
        if(diff_c>0):
            x2=torch.cat([x2, Variable(torch.ones(a2,b2,diff_c,d2).cuda())], 2)
        if(diff_d>0):
            x2=torch.cat([x2, Variable(torch.ones(a2,b2,c2+diff_c,diff_d).cuda())], 3)   

        x3=torch.cat([x, x1, x2], 1)#the dimensionality of Variable is [number,channel,height,width]
        x3=F.relu(self.conv5(x3))
        x3=F.relu(self.conv6(x3))
        x4=torch.cat([x3, x1], 1)
        x4=F.relu(self.conv7(x4))
        x4=F.relu(self.conv8(x4))
        x4=self.upsampling(x4)
        x4=F.relu(self.conv9(x4))
        return x4




#branch:vgg16
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=False).features#It will dawnload the parameters,which will spend some time
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
