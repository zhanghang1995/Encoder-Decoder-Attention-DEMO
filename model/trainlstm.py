import torch
import time
from torch import nn
from model.model_lstm import LSTM
import torch.optim as optim
from model.dataProcessing import train_loader
import importlib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.autograd import Variable

#begin to train lstm

writer = importlib.import_module('tensorboardX').SummaryWriter(log_dir='./log/tensorboard_{}'.format(time.time()))
EPOCH = 10000
model = LSTM(input_size=200, output_size=16, batch_size=200, num_layers=8, hidden_dim=128)
optimizer= optim.Adam(model.parameters(),lr=0.001)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        # print(x.shape,y.shape)
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        model.zero_grad()
        optimizer.zero_grad()

        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        # Step 2 . 前向传播
        output,_ = model(x.float())
        # Step 3 . 计算损失和梯度值
        # print(output.shape)
        # print(output)
        # print("___________")
        # print(output,"------------",y.squeeze())
        loss = loss_func(output, y.squeeze())
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss.backward()                     # backpropagation, compute gradients
        print('Loss',loss.item())
        optimizer.step()                    # apply gradients
        # tensorboard
        writer.add_scalar('loss', loss,step)

        # start to epoch
        if step % 1000 == 0:
            # print(loss.data[0])
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
            torch.save(model.state_dict(),'./checkpoints/'+'checkpoint_epoch_{}.pth'.format(epoch))

torch.save(model,'train_linlin.pkl')