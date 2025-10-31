#!/usr/bin/env python
# coding: utf-8

# In[61]:
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pickle
import os

import scipy.stats
from scipy.signal import convolve2d, fftconvolve
import math

import sys
import warnings

import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

seed=int(sys.argv[1])
print("seed={}".format(seed))
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download and load the MNIST data
mnist = torchvision.datasets.MNIST(root='~/data/', train=True, download=True,
                                   transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

# Convert the PyTorch tensors to numpy arrays
X = mnist.data.numpy()
y = mnist.targets.numpy()

# use only numbers zero and one
#mask = (y==0) | (y==1)
#X, y = X[mask], y[mask]

# Reshape the image data to be 2D rather than 1D (28x28 instead of 784)
X = X.reshape(X.shape[0], 1, 28, 28)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, random_state=1)

# Convert the numpy arrays to PyTorch tensors
X_train_WL = torch.tensor(X_train).to(torch.float32).to(device)
y_train_WL = torch.tensor(y_train).to(torch.long).to(device)


data_num = len(X_train_WL)
print('Number of WL data: ' + str(data_num))
cut = data_num // 2

train_X = X_train_WL[:cut]
train_y = y_train_WL[:cut]

val_X = X_train_WL[cut:cut*2]
val_y = y_train_WL[cut:cut*2]




from torch.utils import data
def load_data(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# In[21]:


batch_size=4
train_iter = load_data((train_X, train_y), batch_size)
test_iter = load_data((val_X, val_y), batch_size)



# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_channels = 2
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(9*num_channels, 10)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x0 = self.relu(self.conv1(x))
        x0 = self.maxpool(x0)

        x1 = self.relu(self.conv2(x0))
        x1 = self.relu(self.conv3(x1))
        #x1 = x1 + x0

        x1 = self.maxpool(x1)

        x2 = self.relu(self.conv4(x1))
        x2 = self.relu(self.conv5(x2))
        #x2 = x2 + x1

        x2 = self.maxpool(x2)

        x2 = self.flatten(x2)
        x2 = self.fc(x2)

        return x2

# Create an instance of the network
net = Net().to(device)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.001)

# In[26]:

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# In[27]:


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # Sum of training loss, sum of training accuracy, no. of examples
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.size().numel())
        else:
            l.sum().backward()
            for param in net.parameters():
                print(param.grad)
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2]


# In[28]:


loss = nn.CrossEntropyLoss()

lr = 0.003
updater = torch.optim.SGD(net.parameters(), lr=lr)


# In[30]:
def test_metrics(net, test_iter, loss):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]  # Return test loss and test accuracy

# In[31]:


def train_net(net, train_iter, test_iter, loss, num_epochs, updater):
    train_loss_sum, train_acc_sum, test_loss_sum, test_acc_sum = [], [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_loss, test_acc = test_metrics(net, test_iter, loss)
        if epoch % 100 == 0:
            print(f"epoch {epoch+1}: train loss {train_loss}, train acc {train_acc}, test loss {test_loss}, test acc {test_acc}")
        train_loss_sum.append(train_loss)
        train_acc_sum.append(train_acc)
        test_loss_sum.append(test_loss)
        test_acc_sum.append(test_acc)
    return train_loss_sum, train_acc_sum, test_loss_sum, test_acc_sum

# In[36]:


net.apply(init_weights)
num_epochs = 2000


# ### test GPU time

# In[34]:


train_loss_sum, train_acc_sum, test_loss_sum, test_acc_sum = train_net(net, train_iter, test_iter, loss, num_epochs, updater)


# In[35]:

np.savez_compressed("metrics.npz", train_loss=train_loss_sum, train_acc=train_acc_sum, test_loss=test_loss_sum, test_acc=test_acc_sum)



