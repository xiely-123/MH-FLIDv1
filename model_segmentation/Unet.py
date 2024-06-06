import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as F
import numpy as np



def global_share(in_planes,out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=64,kernel_size=3,stride=stride,padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=stride,padding=2, groups = 64),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=7,stride=stride,padding=3, groups = 64),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(in_channels=64,out_channels=out_planes,kernel_size=7,stride=stride,padding=3, groups = 64),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class selfAttention(nn.Module) :
    def __init__(self, input_size, hidden_size):
        super(selfAttention, self).__init__()
        self.attention_head_size = hidden_size
        self.key_layer = nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.query_layer = nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.value_layer = nn.Conv2d(input_size, hidden_size, kernel_size=1, padding=0, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, y):
        key = self.gap(self.key_layer(x)).view(x.size(0), -1)
        query = self.gap(self.query_layer(y)).view(y.size(0), -1)
        m,n=y.size(2),y.size(3)
        value_heads = self.value_layer(y).reshape(y.size(0), y.size(1), -1)
        #print(key.size(),query.size(),value_heads.size(),m,n)
        key_heads = torch.unsqueeze(key,-1)
        query_heads = torch.unsqueeze(query,1)
        #print(key_heads.shape,query_heads.shape)
        attention_scores = torch.matmul(key_heads,query_heads)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print(attention_scores.shape)
        attention_probs = F.softmax(attention_scores, dim = -1)
        #print(attention_probs.size())

        context = torch.matmul(attention_probs, value_heads)
        #print(context.size())
        context = y.reshape(context.size(0), context.size(1), m, n)
        #print(context.size())
        return context

def global_share_decoder(in_planes,out_planes, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, 64, kernel_size=2, stride=2),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, out_planes, kernel_size=2, stride=2),
    )


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.FA = nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False)
        self.att = selfAttention(512, 512)
        self.global_share_decoder = global_share_decoder(512,2)
        self.global_share = global_share(3,512)
        

    def forward(self, x_in, stage):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #print(x5.size())
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        x_d = self.global_share(x_in)
        #print(x_d.size())
        if stage == 1:
           x_d = self.att(x_d,self.FA(x5))
        else:
           x_d = self.att(self.FA(x5),x_d)
        
        y = self.global_share_decoder(x_d)
        #print(y.size())
        
        
        
        return logits,y#, x
        
"""
x = torch.rand(2,3,512,512)
net = UNet(3,2)
y, x1= net(x)
print(y.size(), x1.size())
"""