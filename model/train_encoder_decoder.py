# -*- coding:utf-8 -*-
import torch
import time
from torch import nn
from model.model_lstm import LSTM
import torch.optim as optim
from model.dataProcessing import train_loader,test_loader
import importlib
from model.EncoderDecoder import EncoderLSTM,DecoderCNN
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
torch.cuda.set_device(0)

#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#begin to train lstm

writer = importlib.import_module('tensorboardX').SummaryWriter(log_dir='./log/tensorboard_{}'.format(time.time()))
EPOCH = 1000
encoder = EncoderLSTM(2, 16, 30, 300, 128, 28 * 28).to(device)
decoder = DecoderCNN().to(device)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001,
                               betas=(0.5, 0.999))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001,
                               betas=(0.5, 0.999))
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x)
        x = x.cuda()
        y = Variable(y)
        y = y.cuda()
        # print(x.shape,y.shape)
        # Step 1. 请记住 Pytorch 会累加梯度
        # 每次训练前需要清空梯度值
        # 此外还需要清空 LSTM 的隐状态
        # 将其从上个实例的历史中分离出来
        # 重新初始化隐藏层数据，避免受之前运行代码的干扰,如果不重新初始化，会有报错。
        # Step 2 . 前向传播
        output_encoder,_ = encoder(x.float())
        output = decoder(output_encoder.reshape(16,200,28,28))
        # Step 3 . 计算损失和梯度值
        # print(output.shape)
        # print(output)
        # print("___________")
        # print(output,"------------",y.squeeze())
        loss = loss_func(output, y.squeeze())
        # Step 4. 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()                     # backpropagation, compute gradients
        print('Loss',loss.item())
        encoder_optimizer.step()                    # apply gradients
        decoder_optimizer.step()
        # tensorboard
        writer.add_scalar('loss', loss,step)

        # start to epoch
        if step % 1000 == 0:
            # print(loss.data[0])
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
        if step % 1000 == 0:
            torch.save(encoder.state_dict(),'./checkpoints/'+'encoder_epoch_{}.pth'.format(epoch))
            torch.save(decoder.state_dict(),'./checkpoints/'+'decoder_epoch_{}.pth'.format(epoch))

torch.save(encoder,'train_linlin_encoder.pkl')
torch.save(encoder,'train_linlin_decoder.pkl')