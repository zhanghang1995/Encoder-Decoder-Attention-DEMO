import torch as t
from torch import nn
import torch.nn.functional as F
from model.base.tcn import TemporalConvNet
from  torch import autograd

"""
build the encoder-decoder model
the encoder model use LSTM to extract the features()
the decider model use CNN to generate the nclass
the input dim = {batch, sequence, feature} 480*200*30 for each person
output = {480*200*28*28} for batch*sequence*features(Encoder)
"""

class EncoderLSTM(nn.Module):
    def __init__(self, num_layers, batch_size, input_size, time_step, hidden_dim, output_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.time_step = time_step
        self.output_size = output_size
        self.hidden = self.init_hidden()


        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                            dropout=0.1, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_size)

        for l in range(num_layers):
            # torch.nn.init.orthogonal_(self.lstm.all_weights[l][0])
            # torch.nn.init.orthogonal_(self.lstm.all_weights[l][1])
            # torch.nn.init.constant_(self.lstm.all_weights[l][2], 0.01)
            # torch.nn.init.constant_(self.lstm.all_weights[l][3], 0.01)
            t.nn.init.xavier_uniform_(self.lstm.all_weights[l][0])
            t.nn.init.xavier_uniform_(self.lstm.all_weights[l][1])
            t.nn.init.constant_(self.lstm.all_weights[l][2], 0.001)
            t.nn.init.constant_(self.lstm.all_weights[l][3], 0.001)
    def init_hidden(self):
        return (t.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                t.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    def forward(self, input):
        lstm_in = input
        lstm_out, hidden = self.lstm(lstm_in, self.hidden)
        # outs = []    # save all predictions
        # for time_step in range(self.time_step):    # calculate output for each time step
        #     outs.append(t.nn.functional.log_softmax(self.hidden2out(lstm_out[:, time_step, :]),dim=-1))
        out = t.nn.functional.log_softmax(self.hidden2out(lstm_out[:, :, :]), dim=-1)
        hidden = (t.autograd.Variable(hidden[0]), t.autograd.Variable(hidden[1]))
        return out, hidden

"""
build the decoder model
CNN，the input 
"""
# 两层卷积
class DecoderCNN(nn.Module):
    def __init__(self):
        super(DecoderCNN, self).__init__()
        # 使用序列工具快速构建
        #input （64，1，28，28）
        self.conv1 = nn.Sequential(
            nn.Conv2d(200, 128, kernel_size=5, padding=2),
            #output  （128，28，28）
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))
            #(128,14,14)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            #(64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
            # (64,7,7)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=5, padding=2),
            #(16,7,7)
            nn.BatchNorm2d(16),
            nn.LogSoftmax(dim=-1))
            # (16,7,7)
        self.fc = nn.Linear(7 * 7 * 16, 16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out



#test the model

if __name__ == "__main__":
    #batch size 54  200个序列，每个序列对应20维
    encoder = EncoderLSTM(2,16,30,300,128,28*28)
    inputs = t.randn(16,200,30)
    output,hidden = encoder(inputs)
    print(output.shape)
    #the output of the encoder as the input of the CNN
    decoder = DecoderCNN()
    print(decoder)
    # inputs = t.randn(output.reshape(64,200,28,28))
    output = decoder(output.reshape(64,200,28,28))
    print(output)
