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
from torchvision import transforms 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

seed=int(sys.argv[1])
print("seed={}".format(seed))
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download and load data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B each channel is normalized
])
trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform_train)

# Convert the PyTorch tensors to numpy arrays
X = np.array(trainset.data)
y = np.array(trainset.targets)

# CIFAR-10 labels:
# airplane : 0, automobile : 1, bird: 2, cat : 3, deer : 4, dog : 5, frog : 6, horse : 7, ship : 8, truck : 9
# use only deers and frogs
mask = (y==3) | (y==5)
X, y = X[mask], y[mask]
y = (y-3)//2

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, random_state=1)

# Convert the numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train).to(torch.float32).to(device)
X_train = X_train.permute(0, 3, 1, 2)
y_train = torch.tensor(y_train).to(torch.long).to(device)
X_test = torch.tensor(X_test).to(torch.float32).to(device)
X_test = X_test.permute(0, 3, 1, 2)
y_test = torch.tensor(y_test).to(torch.long).to(device)


# Convert the numpy arrays to PyTorch tensors
X_train_WL = X_train
y_train_WL = y_train


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


batch_size=64
train_iter = load_data((train_X, train_y), batch_size)
test_iter = load_data((val_X, val_y), batch_size)



# Define the CNN
class LayerNormResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerNormResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # Assuming the input size is not changed by conv1 due to padding=1 and kernel_size=3
        self.ln1 = nn.LayerNorm([out_channels], elementwise_affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # The size should be the same for conv2 if stride=1, padding=1 and kernel_size=3
        self.ln2 = nn.LayerNorm([out_channels], elementwise_affine=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                #nn.LayerNorm([out_channels, 1, 1], elementwise_affine=True)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # We need to adapt the LayerNorm to the actual size of the output
        out = self.ln1(out.transpose(1, 3)).transpose(1, 3)
        out = self.relu(out)

        out = self.conv2(out)
        # Same for the second norm layer
        out = self.ln2(out.transpose(1, 3)).transpose(1, 3)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 6
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, self.inchannel,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, self.inchannel*2, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, self.inchannel*2, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, self.inchannel*2, 2, stride=2)
        self.fc = nn.Linear(self.inchannel, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Create an instance of the network
net = ResNet(ResidualBlock=LayerNormResNetBlock, num_classes=2).to(device)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_out')

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

lr = 0.0001
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
num_epochs = 1501


# ### test GPU time

# In[34]:


train_loss_sum, train_acc_sum, test_loss_sum, test_acc_sum = train_net(net, train_iter, test_iter, loss, num_epochs, updater)


# In[35]:

np.savez_compressed("metrics.npz", train_loss=train_loss_sum, train_acc=train_acc_sum, test_loss=test_loss_sum, test_acc=test_acc_sum)



