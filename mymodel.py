import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models
from torch.autograd import Variable

#residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.conv(x)
        residual = self.prelu(residual)
        residual = self.conv(residual)
        out      = x + residual

        return out


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


class res15(nn.Module):
    def __init__(self):
        super(res15, self).__init__()
        self.conv1=nn.Conv2d(3,64,9,1,(9-1)/2)
        self.prelu=nn.PReLU()
        self.downsampling=nn.Conv2d(64, 64, 4, 2, 1)#conv2d or pooling or ConvTranspose2d or m = nn.MaxPool2d(3, stride=2)
        self.upsampling=nn.PixelShuffle(2)        
        self.resnet5 = nn.ModuleList([ResidualBlock(64) for i in range(5)])
        self.resnet10 = nn.ModuleList([ResidualBlock(64+32) for i in range(10)])
        self.conv2=nn.Conv2d(96,3,9,1,(9-1)/2)
        self.conv3=nn.Conv2d(96,96*4,3,1,1)  
    def forward(self,inputs,vgg_feature):#
        x1=self.prelu(self.conv1(inputs))
        x1=self.downsampling(x1)
        for layer in self.resnet5:
            x1 = layer(x1)
        x2=self.upsampling(self.upsampling(vgg_feature))#?this variable may not require gradient!

        ############################
        '''make the patch size same:because vgg use pooling for downsampling,while x2 use conv2d.
        pooling will lead to even numbers but sometimes conv2d will lead to odd numbers'''
        [a1,b1,c1,d1]=x1.size()
        [a2,b2,c2,d2]=x2.size()     
        diff_c=c1-c2
        diff_d=d1-d2
        if(diff_c>0):
            x2=torch.cat([x2, Variable(torch.ones(a2,b2,diff_c,d2).cuda())], 2)
        if(diff_d>0):
            x2=torch.cat([x2, Variable(torch.ones(a2,b2,c2+diff_c,diff_d).cuda())], 3)
        #############################   

        x=torch.cat([x1, x2], 1)
        for layer in self.resnet10:
            x = layer(x)

        x=self.upsampling(self.conv3(x))
        x=self.conv2(self.prelu(x))
        return x



'''
#something wrong when i use the BatchNorm2d layer,so i did not choose this structure.
#residual block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.conv(x)
        residual = self.bn(residual)
        residual = self.prelu(residual)
        residual = self.conv(residual)
        residual = self.bn(residual)
        out      = x + residual
        out=self.prelu(out)

        return out
'''