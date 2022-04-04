import pandas as pd
import os

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from preprocessing import preprocessing_pipeline
import pickle
import numpy as np
import torch
from datetime import datetime
import torch.optim as optim
from torch.nn import MSELoss, HuberLoss
from autoencoder.models_500 import CAEn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device))
    # Data Loading
    torch.cuda.current_device()
    window_size = 125
    batch_size = 32
    train = np.load('train.pickle.npy', allow_pickle=True)
    test = np.load('test.pickle.npy', allow_pickle=True)
    print('shape of data is Train {}, Test {}'.format(train.shape, test.shape))
    # Model parameters
    model = CAEn().to(device)
    criterion = HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # Split data into train and val
    X_train, X_val = train, test
    X_train = torch.Tensor(X_train)  # transform to torch tensor
    X_val = torch.Tensor(X_val)
    train_dataset = TensorDataset(X_train)  # create your datset
    val_dataset = TensorDataset(X_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2,
                                  drop_last=True)  # create your dataloader
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, drop_last=True)
    # Initialize parameters
    # Initializing in a separate cell so we can easily add more epochs to the same run
    EPOCHS = 20
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    losses = []
    vlosses = []
    for epoch in range(EPOCHS):
        _vloss = 0.0
        _loss = 0.0
        print('EPOCH {}:'.format(epoch_number + 1))
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

        model.train()
        for i, inputs in enumerate(train_dataloader):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            bottleneck, outputs = model(inputs[0][None, ...][0].to(device))
            loss = criterion(outputs, inputs[0].to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            _loss += loss
        losses.append(_loss.cpu().detach().numpy())
        model.eval()
        for i, vdata in enumerate(val_dataloader):
            vbottleneck, voutputs = model(vdata[0][None, ...][0].to(device))
            vloss = criterion(voutputs, vdata[0].to(device))
            _vloss += vloss
        vlosses.append(_vloss.cpu().detach().numpy())

        print('LOSS train {} valid {}'.format(_loss, _vloss))

        # Track best performance, and save the model's state
        if _loss < _vloss:
            print('FINAL LOSS train {} valid {}'.format(_loss, _vloss))
            print('EPOCH {}:'.format(epoch_number))

        epoch_number += 1

    model_path = './model/model_{}_{}'.format(timestamp, epoch_number)
    torch.save(model.state_dict(), model_path)
    plt.plot(losses, label='Train loss')
    plt.plot(vlosses, label='Validation loss')
    plt.legend()
    plt.show()
