import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.io import loadmat
import numpy as np
import pandas as pd
import xlrd


batch_size = 16
#前30均值，后30方差 A,B,C分别代表三根不同天线

if os.path.exists(os.path.join('./trainingfiles/', 'train_file_A.pth')):
    train_A_mean_data , train_label = torch.load('./trainingfiles/train_file_A.pth')
    test_A_mean_data ,test_label = torch.load('./testingfiles/test_file_A.pth')
    print('train_file.pth is loaded')
else:
    print('Processing the data...')
    filepath_A = './data/data_480_30_200/lin_antennaA_all.xls'
    filepath_B = './data/data_480_30_200/lingchen_antennaA_all.xls'
    filepath_C = './data/data_480_30_200/tianfeng_antennaA_all.xls'
    labelpath = './data/data_480_30_200/label.xlsx'
    data_A_O = np.array(pd.read_excel(filepath_A, header=None))
    data_B_O = np.array(pd.read_excel(filepath_B, header=None))
    data_C_O = np.array(pd.read_excel(filepath_C, header=None))
    label = np.array(pd.read_excel(labelpath,header=None))
    re_data_A = data_A_O.reshape(30,480,200)
    re_data_B = data_A_O.reshape(30,480,200)
    re_data_C = data_A_O.reshape(30,480,200)

    #交换矩阵维度
    data_A_O = re_data_A.reshape(480,30,200)
    data_A = data_A_O.reshape(480,200,30)
    data_B_O = re_data_B.reshape(480, 30, 200)
    data_B = data_B_O.reshape(480, 200, 30)
    data_C_O = re_data_C.reshape(480, 30, 200)
    data_C = data_C_O.reshape(480, 200, 30)


    #slice the data
    #A
    trian_mean_A_data = np.array(data_A)[0:336]
    test_mean_A_data = np.array(data_A)[336:480]
    #B
    trian_mean_B_data = np.array(data_B)[0:336]
    test_mean_B_data = np.array(data_B)[336:480]
    #C
    trian_mean_C_data = np.array(data_C)[0:336]
    test_mean_C_data = np.array(data_C)[336:480]

    #transfrom to tensor
    label = np.array(label).reshape(480,1)
    tarin_label = label[0:336]
    test_label = label[336:480]


    #A tensor(test and train)
    train_A_mean_data = torch.tensor(trian_mean_A_data,dtype=torch.long)
    test_A_mean_data = torch.tensor(test_mean_A_data, dtype=torch.long)
    #B tensor(test and train)
    train_B_mean_data = torch.tensor(trian_mean_B_data, dtype=torch.long)
    test_B_mean_data = torch.tensor(test_mean_B_data, dtype=torch.long)
    #C tensor(test and train)
    train_C_mean_data = torch.tensor(trian_mean_C_data, dtype=torch.long)
    test_C_mean_data = torch.tensor(test_mean_C_data, dtype=torch.long)
    #label tensor
    train_label = torch.tensor(tarin_label,dtype=torch.long)
    test_label = torch.tensor(test_label,dtype=torch.long)

    #save the file train
    torch.save((train_A_mean_data,train_label),'./trainingfiles/train_file_A.pth')
    torch.save((train_B_mean_data,train_label),'./trainingfiles/train_file_B.pth')
    torch.save((train_C_mean_data,train_label),'./trainingfiles/train_file_C.pth')
    #save the file test
    torch.save((test_A_mean_data,test_label),'./testingfiles/test_file_A.pth')
    torch.save((test_B_mean_data,test_label),'./testingfiles/test_file_B.pth')
    torch.save((test_C_mean_data,test_label),'./testingfiles/test_file_C.pth')


#封装成为Dataset
train_dataSet = Data.TensorDataset(train_A_mean_data,train_label)
test_dataSet = Data.TensorDataset(test_A_mean_data,test_label)
# test_dataSet = Data.TensorDataset()

torch.manual_seed(1)    # reproducible
#封装成为Loader
# Data Loader for easy mini-batch return in training, the time_series batch shape will be (16, 30, 30)
train_loader = Data.DataLoader(
    dataset=train_dataSet,
    batch_size=batch_size,
    shuffle=True
)

test_loader = Data.DataLoader(
    dataset=test_dataSet,
    batch_size=batch_size,
    shuffle=True
)


# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         print(x.shape,y.shape) #torch.Size([16, 1, 30]) torch.Size([16])
#         b_x = Variable(x.view(-1, 1*30))   # batch x, shape (batch, 28*28)
#         b_y = Variable(x.view(-1, 1*30))   # batch y, shape (batch, 28*28)
#         # print(b_x.shape,b_y.shape) #torch.Size([16, 30]) torch.Size([30, 16])
#         b_label = Variable(y)               # batch label
#         # print(b_label.shape) #[16]
#
#         encoded, decoded = autoencoder(b_x)
#         # print(encoded.shape,decoded.shape) #torch.Size([16, 3]) torch.Size([16, 16])
#
#         loss = loss_func(decoded, b_y)      # mean square error
#
#         optimizer.zero_grad()               # clear gradients for this training step
#         loss.backward()                     # backpropagation, compute gradients
#         optimizer.step()                    # apply gradients
#         # start to epoch
#         if step % 100 == 0:
#             # print(loss.data[0])
#             print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
