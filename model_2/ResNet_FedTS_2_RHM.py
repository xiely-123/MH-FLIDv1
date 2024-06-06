import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import torch.nn.functional as F
#print("PyTorch Version: ",torch.__version__)
#print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet5', 'ResNet8','ResNet11','ResNet14', 'ResNet17','ResNet20','ResNet23', 'ResNet32','ResNet50', 'ResNet101','ResNet152']

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


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )







class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 1, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=128, block=blocks[3], stride=2)

        self.avgpool = nn. AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512,num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.FA = nn.Linear(512, 512)
        self.att = selfAttention(512, 512)
        self.mlp = MLP(512,num_classes,256)
        self.global_share = global_share(1,512)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x_in, stage):
        x = self.conv1(x_in)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(x))
        else:
           x2 = self.att(self.FA(x),x2)
        x1 = self.fc(x)
        y = self.mlp(x2)
        # x1 = self.softmax(x1)
        # y = self.softmax(y)
        return x1/4,y/4

class ResNet_sm(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet_sm,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512,places=128, block=blocks[2], stride=2)
        # self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512,num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.FA = nn.Linear(512, 512)
        self.mlp = MLP(512,num_classes,256)
        self.global_share = global_share(3,512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x_in):
        x = self.conv1(x_in)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.FA(x) + x2
        x1 = self.fc(x)
        y = self.mlp(x2)
        x1 = self.softmax(x1)
        y = self.softmax(y)
        return x1,y

class ResNet_sm_1(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet_sm_1,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        #self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        # self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn. AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512,num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.FA = nn.Linear(512, 512)
        self.mlp = MLP(512,num_classes,256)
        self.global_share = global_share(3,512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x_in):
        x = self.conv1(x_in)

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.FA(x) + x2
        x1 = self.fc(x)
        y = self.mlp(x2)
        x1 = self.softmax(x1)
        y = self.softmax(y)
        return x1,y

class ResNet_sm_2(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet_sm_2,self).__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places = 64, places= 128, block=blocks[0], stride=1)
        #self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
        #self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        # self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn. AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512,num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.FA = nn.Linear(512, 512)
        self.mlp = MLP(512,num_classes,256)
        self.global_share = global_share(3,512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x_in):
        x = self.conv1(x_in)

        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.FA(x) + x2
        x1 = self.fc(x)
        y = self.mlp(x2)
        x1 = self.softmax(x1)
        y = self.softmax(y)
        return x1,y

def ResNet11(num_classes):
    return ResNet_sm([1, 1, 1],num_classes)

def ResNet8(num_classes):
    return ResNet_sm_1([1, 1],num_classes)

def ResNet5(num_classes):
    return ResNet_sm_2([1],num_classes)


def ResNet14(num_classes):
    return ResNet([1, 1, 1, 1],num_classes)

def ResNet17(num_classes):
    return ResNet([1, 1, 1, 2],num_classes)
    
def ResNet20(num_classes):
    return ResNet([1, 1, 2, 2],num_classes)
    
def ResNet23(num_classes):
    return ResNet([1, 2, 2, 2],num_classes)

def ResNet32(num_classes):
    return ResNet([1, 3, 3, 3],num_classes)

def ResNet50(num_classes):
    return ResNet([3, 4, 6, 3],num_classes)

def ResNet101(num_classes):
    return ResNet([3, 4, 23, 3],num_classes)

def ResNet152(num_classes):
    return ResNet([3, 8, 36, 3],num_classes)


# net = ResNet50(4)

# image = torch.rand(2,1,256,256)
# a,_ = net(image)
# print(a.size())








