from torch import nn
import torch
import torchvision
import numpy as np
import math
import torch.nn.functional as F

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
        nn.Conv1d(in_channels=in_planes,out_channels=64,kernel_size=3,stride=stride,padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=7, stride=3, padding=2),
        nn.Conv1d(in_channels=64,out_channels=64,kernel_size=5,stride=stride,padding=2, groups = 64),
        #nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(kernel_size=7, stride=3, padding=2),
        nn.Conv1d(in_channels=64,out_channels=out_planes,kernel_size=7,stride=stride,padding=2, groups = 64),
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


class TCN(nn.Module):
    def __init__(self, class_number):
        super(TCN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7,
                      stride=3, bias=False, padding=(7//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = 127
        self.logits = nn.Linear(model_output_dim * 64, class_number)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.FA = nn.Linear(model_output_dim * 64, 128)
        self.att = selfAttention(128, 128)
        self.mlp = MLP(128,class_number,256)
        self.global_share = global_share(1,128)
    def forward(self, x_in, stage):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        #print(x.size())
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        
        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(x_flat))
        else:
           x2 = self.att(self.FA(x_flat),x2)
        y = self.mlp(x2)
        
        return logits/4, y/4
        
        
class TransFormer(nn.Module):
    def __init__(self, class_number):
        super(TransFormer, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=33,
                      stride=3, bias=False, padding=(33//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.layer1 = nn.MultiheadAttention(32, 4)
        self.layer2 = nn.MultiheadAttention(32, 4)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        model_output_dim = 251
        self.logits = nn.Linear(model_output_dim * 32, class_number)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.FA = nn.Linear(model_output_dim * 32, 128)
        self.att = selfAttention(128, 128)
        self.mlp = MLP(128,class_number,256)
        self.global_share = global_share(1,128)
    def forward(self, x_in, stage):
        x = self.conv_block1(x_in)
        x=x.permute(0, 2, 1)
        x,_ = self.layer1(x,x,x)
        x=x.permute(0, 2, 1)
        x = self.pooling(x)
        x=x.permute(0, 2, 1)
        x,_= self.layer2(x,x,x)
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        
        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(x_flat))
        else:
           x2 = self.att(self.FA(x_flat),x2)
        y = self.mlp(x2)
        
        return logits/4, y/4

        



class RNN(nn.Module):
    def __init__(self, class_number):
        super(RNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=33,
                      stride=3, bias=False, padding=(33//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.1)
        )

        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=16,   # rnn 隐藏单元数
                            num_layers=3,     # rnn 层数
                            batch_first=True)
        model_output_dim = 16
        self.logits = nn.Linear(model_output_dim * 501, class_number)
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.FA = nn.Linear(model_output_dim * 501, 128)
        self.att = selfAttention(128, 128)
        self.mlp = MLP(128,class_number,256)
        self.global_share = global_share(1,128)
    def forward(self, x_in, stage):
        x = self.conv_block1(x_in)
        x=x.permute(0, 2, 1)
        x,_ = self.lstm(x)
        #print(x.size())
        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)

        x2 = self.global_share(x_in)
        x2 = self.avgpool(x2)
        x2 = x2.view(x2.size(0), -1)
        if stage == 1:
           x2 = self.att(x2,self.FA(x_flat))
        else:
           x2 = self.att(self.FA(x_flat),x2)
        y = self.mlp(x2)
        
        return logits/4, y/4
        
        
        
        
        
        
        
# path = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/train.pt"
# path1 = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/val.pt"

# dataset = torch.load(path1)
# X_train = dataset["samples"]#3000
# y_train = dataset["labels"].sort()###0-4

# print(y_train[0])

# net = RNN(5)

# image = torch.rand(2,1,3000)
# a, b= net(image,1)
# print(a.size(),b.size())

# net = TCN(5)

# image = torch.rand(2,1,3000)
# a, b= net(image,1)
# print(a.size(),b.size())

# net = TransFormer(5)

# image = torch.rand(2,1,3000)
# a, b= net(image,1)
# print(a.size(),b.size())

# net = TCN(5)

# image = torch.rand(2,1,3000)
# a, b= net(image,1)
# print(a.size(),b.size())