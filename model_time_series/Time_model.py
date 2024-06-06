from torch import nn
import torch




class TCN(nn.Module):
    def __init__(self, configs):
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
        self.logits = nn.Linear(model_output_dim * 64, 5)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        #print(x.size())
        x_flat = x.reshape(x.shape[0], -1)


        logits = self.logits(x_flat)
        return logits
        
        
class TransFormer(nn.Module):
    def __init__(self, configs):
        super(TF, self).__init__()

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
        self.logits = nn.Linear(model_output_dim * 32, 5)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x=x.permute(0, 2, 1)
        x,_ = self.layer1(x,x,x)
        x=x.permute(0, 2, 1)
        x = self.pooling(x)
        x=x.permute(0, 2, 1)
        x,_= self.layer2(x,x,x)
        x_flat = x.reshape(x.shape[0], -1)


        logits = self.logits(x_flat)
        return logits

        



class RNN(nn.Module):
    def __init__(self, configs):
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
                            hidden_size=128,   # rnn 隐藏单元数
                            num_layers=3,     # rnn 层数
                            batch_first=True)
        model_output_dim = 128
        self.logits = nn.Linear(model_output_dim * 1, 5)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x=x.permute(0, 2, 1)
        x,_ = self.lstm(x)
        #print(x.size())
        x_flat = x[:,-1,:].reshape(x.shape[0], -1)


        logits = self.logits(x_flat)
        return logits
        
        
        
        
        
        
        
# path = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/train.pt"
# path1 = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/val.pt"



# dataset = torch.load(path1)
# X_train = dataset["samples"]#3000
# y_train = dataset["labels"].sort()###0-4

# print(y_train[0])

# net = RNN(5)

# image = torch.rand(2,1,3000)
# a = net(image)
# print(a.size())

# path = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/train.pt"
# path1 = "/home/xly/Time_Contrastive_learning/TDPRNN/data/sleepEDF/val.pt"



# dataset = torch.load(path1)
# X_train = dataset["samples"]#3000
# y_train = dataset["labels"].sort()###0-4

# print(y_train[0])

# net = TransFormer(5)

# image = torch.rand(2,1,3000)
# a = net(image)
# print(a.size())
