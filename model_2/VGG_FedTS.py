"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import torch.nn.functional as F

cfg = {
    'G' : [64,     'M', 128,      'M', 256, 256,           'M', 512,           'M', 512,           'M'],
    'F' : [64,     'M', 128,      'M', 256, 256,           'M', 512,           'M', 512,           'M'],
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

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

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.FA = nn.Linear(512, 512)
        self.att = selfAttention(512, 512)
        #self.att1 = selfAttention(512, 512)
        self.mlp = MLP(512,num_class,256)
        self.global_share = global_share(1,512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x, stage):
        output = self.avg(self.features(x))
        output = output.view(output.size()[0], -1)
        x1 = self.classifier(output)
        
        x2 = self.global_share(x)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(output))
        else:
           x2 = self.att(self.FA(output),x2)
        
        
        y = self.mlp(x2)
        
        

        return x1/4,y/4

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 1
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)







def vgg11_bn(num_class):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_class)

def vgg13_bn(num_class):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_class)

def vgg16_bn(num_class):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_class)

def vgg19_bn(num_class):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_class)
    
    
# net = vgg19_bn(8)

# image = torch.rand(2,3,256,256)
# a,b = net(image)
# print(a.size(),b.size())