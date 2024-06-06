"""senet in pytorch



[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        #nn.Linear(dim, 128),
        # nn.ReLU(inplace=True),
        # nn.Linear(128, 128),
        nn.BatchNorm1d(dim),
        nn.ReLU(inplace=True),
        nn.Linear(dim, projection_size)
    )


def global_share(in_planes,out_planes, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=64,kernel_size=3,stride=stride,padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=stride,padding=2, groups = 64),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(in_channels=64,out_channels=out_planes,kernel_size=7,stride=stride,padding=2, groups = 64),
        #nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class selfAttention(nn.Module) :
    def __init__(self, input_size, hidden_size):
        super(selfAttention, self).__init__()
        self.attention_head_size = hidden_size
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)


    def forward(self, x, y):
        key = self.key_layer(x)
        query = self.query_layer(y)
        value_heads = torch.unsqueeze(y,-1)
        key_heads = torch.unsqueeze(key,-1)
        query_heads = torch.unsqueeze(query,1)
        #print(key_heads.shape,query_heads.shape)
        attention_scores = torch.matmul(key_heads,query_heads)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print(attention_scores.shape)
        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)

        context = torch.squeeze(context,-1)
        return context
        
class BasicResidualSEBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=100):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.fc = nn.Linear(self.in_channels, class_num)
        self.FA = nn.Linear(self.in_channels, 512)
        self.att = selfAttention(512, 512)
        self.mlp = MLP(512,class_num,256)
        self.global_share = global_share(1,512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x_in, stage):
        x = self.pre(x_in)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x1 = self.fc(x)

        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(x))
        else:
           x2 = self.att(self.FA(x),x2)
        y = self.mlp(x2)

        return x1/4,y/4


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)

def seresnet18(class_num):
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2],class_num)

def seresnet34(class_num):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3],class_num)

def seresnet50(class_num):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3],class_num)

def seresnet101(class_num):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3],class_num)

def seresnet152(class_num):
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3],class_num)
    
    
# net = seresnet18(8)

# image = torch.rand(2,3,256,256)
# a,b = net(image)
# print(a.size(),b.size())