#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets

import torch 
from torch import nn
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn import preprocessing
import os

import scipy.stats
from scipy.signal import convolve2d, fftconvolve
import math

import sys
import warnings

if (len(sys.argv) != 2):
    print('Usage: Ca_train.py net_width')
    sys.exit()

net_width = int(sys.argv[1])

shuffled_df = pd.read_csv('./train_shuffled.csv')

df_X = shuffled_df.iloc[:, :-1]
df_y = shuffled_df.iloc[:, -1]

split_idx = len(df_X) // 2
X_train = df_X.iloc[:split_idx].values
y_train = df_y.iloc[:split_idx].values
X_test = df_X.iloc[split_idx:].values
y_test = df_y.iloc[split_idx:].values

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)


# In[11]:


X_train = X_train.to(torch.float32)
y_train = y_train.to(torch.float32)
X_test = X_test.to(torch.float32)
y_test = y_test.to(torch.float32)

y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)


device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


# Setting seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# Set seed for reproducibility
#set_seed(123)

batch_size = 128


from torch.utils import data
def load_data(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# In[48]:



train_iter = load_data((X_train, y_train), batch_size)
test_iter = load_data((X_test, y_test), batch_size)

net = nn.Sequential(nn.Linear(331, net_width), nn.ReLU(), nn.Linear(net_width, 1))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
#        nn.init.normal_(m.weight, std=0.01)

net.to(torch.float32)
net.apply(init_weights)
net = net.cuda()

# In[50]:


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[51]:


def train_epoch_end_loss(net, train_iter, X_train, y_train, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(2)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y.reshape(y_hat.shape))
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), y.size().numel())
        else:
            l.sum().backward()
            for param in net.parameters():
                print(param.grad)
            updater(X.shape[0])
            metric.add(float(l.sum()), y.numel())

    if isinstance(net, nn.Module):
        net.eval()
    with torch.no_grad():
        y_hat = net(X_train)
        end_loss = loss(y_hat, y_train.reshape(y_hat.shape)).item()
    
    return metric[0] / metric[1], end_loss


# define function to calculate test loss at each epoch
def test_epoch_end_loss(net, test_iter, X_test, y_test, loss):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            l = loss(y_hat, y.reshape(y_hat.shape))
            metric.add(float(l) * len(y), y.size().numel())

        y_hat = net(X_test)
        end_loss = loss(y_hat, y_test.reshape(y_hat.shape)).item()

    #return float(l), l.item()
    return metric[0] / metric[1], end_loss


# define the full training function
def train_net_ELoss_no_print(net, train_iter, test_iter, X_train, y_train, X_test, y_test, loss, num_epochs, updater):
    train_loss_sum = []
    test_loss_sum = []

    train_EL_sum = []
    test_EL_sum = []

    for epoch in range(num_epochs):
        train_metrics, train_end_loss = train_epoch_end_loss(net, train_iter, X_train, y_train, loss, updater)
        test_metrics, test_end_loss = test_epoch_end_loss(net, test_iter, X_test, y_test, loss)
        #if epoch % 10 == 0:
            #print(f"epoch {epoch+1}: train loss {train_metrics}, test loss {test_metrics}")
            #print(f"epoch {epoch+1}: train end loss {train_end_loss}, test end loss {test_end_loss}")

        train_loss_sum.append(train_metrics)
        test_loss_sum.append(test_metrics)

        train_EL_sum.append(train_end_loss)
        test_EL_sum.append(test_end_loss)

    return train_EL_sum, test_EL_sum


steps = 100
num_epochs = 250

train_iter = load_data((X_train, y_train), batch_size)
test_iter = load_data((X_test, y_test), batch_size=len(X_test), is_train=False)

train_loss1 = []
test_loss1 = []

train_loss_traj = []
test_loss_traj = []

for i in range(steps):
    net.apply(init_weights)

    loss = nn.MSELoss(reduction='mean')
    lr = 3e-4
    updater_sgd = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    train_EL_sum, test_EL_sum = train_net_ELoss_no_print(net, train_iter, test_iter,  X_train, y_train,
                                                         X_test, y_test, loss, num_epochs, updater_sgd)

    print(f"Iteration {i+1}: train loss {train_EL_sum[-1]}, test loss {test_EL_sum[-1]}")

    train_loss1.append(train_EL_sum[-1])
    test_loss1.append(test_EL_sum[-1])

    train_loss_traj.append(train_EL_sum)
    test_loss_traj.append(test_EL_sum)


with open('./SGD_train_loss_traj.pickle', 'wb') as pickle_file:
    pickle.dump(train_loss_traj, pickle_file)

with open('./SGD_test_loss_traj.pickle', 'wb') as pickle_file:
    pickle.dump(test_loss_traj, pickle_file)

with open('./SGD_train_end_loss.pickle', 'wb') as pickle_file:
    pickle.dump(train_loss1, pickle_file)

with open('./SGD_test_end_loss.pickle', 'wb') as pickle_file:
    pickle.dump(test_loss1, pickle_file)

print(np.mean(train_loss1)), print(np.mean(test_loss1))

# save one net, used only when steps == 1
with open('./SGD_net.pickle', 'wb') as pickle_file:
    pickle.dump(net, pickle_file)

