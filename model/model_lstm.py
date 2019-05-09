# -*- coding:utf-8 -*-
from torch import nn
from torch import autograd
import torch as t
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self,input_size,output_size,batch_size,num_layers,hidden_dim):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layes = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size= self.hidden_dim,num_layers = self.num_layes
                            ,batch_first=True)
        #if batch_first set to true:(batch, seq, feature)
        #全连接层，map the hidden state to the output
        self.hidden2out = nn.Linear(self.hidden_dim,self.output_size)
        self.hidden = self.init_hidden()
        #build the structure

        for i in range(num_layers):
            nn.init.xavier_uniform_(self.lstm.all_weights[i][0])
            nn.init.xavier_uniform_(self.lstm.all_weights[i][1])
            nn.init.constant_(self.lstm.all_weights[i][2], 0.001)
            nn.init.constant_(self.lstm.all_weights[i][3], 0.001)
    def init_hidden(self):
        return (t.zeros(self.num_layes,self.batch_size,self.hidden_dim),
                t.zeros(self.num_layes, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        lstm_in = inputs
        lstm_out, hidden = self.lstm(lstm_in,self.hidden)
        #target
        out_put = self.hidden2out(lstm_out)
        target = F.log_softmax(out_put)[:,-1,:]
        hidden = (autograd.Variable(hidden[0]),autograd.Variable(hidden[1]))
        return target,hidden


#test the build model
if __name__ == "__main__":
    model = LSTM(input_size=200,output_size=16,batch_size=64,num_layers=4,hidden_dim=128)
    print(model)
    """
    input(seq_len, batch, input_size) 
    h0(num_layers * num_directions, batch, hidden_size) 
    c0(num_layers * num_directions, batch, hidden_size)
    """
    input = t.randn(64,1,200) #batch_size*sequence length  * input_size

    """
    output(seq_len, batch, hidden_size * num_directions) 
    hn(num_layers * num_directions, batch, hidden_size) 
    cn(num_layers * num_directions, batch, hidden_size)
    """
    output ,_= model(input) #(16,1,1) the last 1 is output
    print(output.size())