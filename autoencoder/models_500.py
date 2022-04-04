# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CAEenc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 6, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(63)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.lstm1 = nn.GRU(6, 128, 10, bias=False, bidirectional=True, batch_first=True)
        # self.pool = nn.MaxPool1d(kernel_size=3, stride= 2)
        self.linear = nn.Linear(128*2*63, 4096)
        self.linear2 = nn.Linear(4096, 2048)
        self.linear3 = nn.Linear(2048, 512)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        #torch.nn.init.kaiming_uniform_(self.lstm1.weight)
        torch.nn.init.kaiming_uniform_(self.linear2.weight)
        torch.nn.init.kaiming_uniform_(self.linear3.weight)

    def forward(self, x):
        # dprint(x.shape)
        f = self.conv1(x)
        f = self.pool(f)
        f = F.gelu(f)
        # Aqui se truncan las dimensiones, donde la principal son los time frames para evaluar en cada una de las celdas
        f = f.permute(0, 2, 1)
        f = self.lstm1(f)[0]
        f = self.bn1(f)
        # Se retorna a la dimension original en caso de ser usado en otra layer. En este no importa mucho porque es una lineal
        f = f.reshape((f.size(0), -1))
        # Aqui depronto es mejor otra funcion de activacion, como una GELU, SELU, Tanh.. ReLU tiende a enviar muchas salidas a cero
        f = self.linear(f)
        f = self.linear2(f)
        f = F.sigmoid(self.linear3(f))
        return f

    def test_tensor(self, x):

        if torch.isnan(x).any():
            print("tensor has nan values")

        sd = torch.std(x, 0)
        # if sd.any()==0:
        #    print("the standard deviation of the tensor is 0")
        self.plot_tensor(x)

    def plot_tensor(self, x):

        # channel 0

        for c in range(x.size()[1]):
            plt.figure(figsize=(20, 10))
            N = x.size()[0]
            for k in range(N):
                plt.subplot(int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N))), k + 1)
                plt.imshow(x.data.numpy()[k, c, :, :])
        plt.show()


class CAEdec(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(63)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.lstm1 = nn.GRU(256, 3, 10, bias=False, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(512, 2048)
        self.linear2 = nn.Linear(2048, 4096)
        self.linear3 = nn.Linear(4096, 128 * 2 * 63)

        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        torch.nn.init.kaiming_uniform_(self.linear.weight)
        torch.nn.init.kaiming_uniform_(self.linear2.weight)
        torch.nn.init.kaiming_uniform_(self.linear3.weight)
    def forward(self, x):
        # dprint(x.shape)
        f = self.linear(x)

        f = self.linear2(f)

        f = F.sigmoid(self.linear3(f))

        # Reshape for lstm again
        f = f.reshape((f.size(0), 63, 256))
        # f = f.reshape(8, 125, 128)

        f = self.bn1(self.lstm1(f)[0])
        f = f.permute(0, 2, 1)
        # f = f.reshape((f.size(0), 8, 125))
        # f = F.leaky_relu(F.interpolate(f, scale_factor=2))
        # f = f.permute(0, 2, 1)
        f = F.interpolate(f, scale_factor=2)
        f = self.conv1(f)
        f = torch.sigmoid(f)
        f = f[:, :, :125]
        """
        # Aqui se truncan las dimensiones, donde la principal son los time frames para evaluar en cada una de las celdas
        f = f.permute(0, 2, 1)
        f =self.lstm1(f)[0]
        f = self.bn1(f)
        # Se retorna a la dimension original en caso de ser usado en otra layer. En este no importa mucho porque es una lineal
        f = f.reshape((f.size(0), -1))
        #Aqui depronto es mejor otra funcion de activacion, como una GELU, SELU, Tanh.. ReLU tiende a enviar muchas salidas a cero
        f = F.sigmoid(f)
        """

        return f


class CAEn(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CAEenc()
        self.decoder = CAEdec()

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return bottleneck, x
