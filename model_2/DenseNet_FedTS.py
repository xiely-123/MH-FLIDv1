import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["densenet100bc", "densenet190bc"]

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

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn_1 = nn.BatchNorm2d(in_planes)
        self.conv_1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv_2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv_1(F.relu(self.bn_1(x)))
        out = self.conv_2(F.relu(self.bn_2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, depth, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        nblocks = (depth - 4) // 6
        num_planes = 2 * growth_rate
        self.conv_1 = nn.Conv2d(1, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans_2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense_3 = self._make_dense_layers(block, num_planes, nblocks)
        num_planes += nblocks * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_planes, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.FA = nn.Linear(num_planes, 512)
        self.att = selfAttention(512, 512)
        self.mlp = MLP(512,num_classes,256)
        self.global_share = global_share(1,512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, stage):
        out = self.conv_1(x)
        out = self.trans_1(self.dense1(out))
        out = self.trans_2(self.dense2(out))
        out = self.dense_3(out)
        out = self.avg(F.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        x1 = self.fc(out)
        
        x2 = self.global_share(x)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(out))
        else:
           x2 = self.att(self.FA(out),x2)
        y = self.mlp(x2)

        return x1,y



def densenet50bc(num_classes):
    return DenseNet(Bottleneck, depth=50, growth_rate=12, num_classes=num_classes)


def densenet100bc(num_classes):
    return DenseNet(Bottleneck, depth=100, growth_rate=12, num_classes=num_classes)


def densenet190bc(num_classes):
    return DenseNet(Bottleneck, depth=190, growth_rate=40, num_classes=num_classes)



# net = densenet50bc(8)

# image = torch.rand(2,3,256,256)
# a,b = net(image)
# print(a.size(),b.size())