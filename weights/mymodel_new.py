import torch
import torch.nn as nn

'''
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


class res15(nn.Module):

    def __init__(self):
        super(res15, self).__init__()
        self.conv1=nn.Conv2d(3,64,7,1,(7-1)/2)
        self.prelu=nn.PReLU()
        self.resnet15 = nn.ModuleList([ResidualBlock(64) for i in range(10)])
        self.conv2=nn.Conv2d(64,3,5,1,(5-1)/2)
        
    def forward(self,inputs):
        x=self.prelu(self.conv1(inputs))
        for layer in self.resnet15:
            x = layer(x)

        x=self.conv2(x)
        return x




