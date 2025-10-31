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
from collections import deque

if (len(sys.argv) != 4):
    print('Usage: average_entropy.py start_epoch epoch_step process_id')
    sys.exit()

start_epoch = int(sys.argv[1])
epoch_step = int(sys.argv[2])
process_id = int(sys.argv[3])

torch.set_num_threads(4)
torch.manual_seed(process_id+start_epoch)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Download and load the MNIST data
mnist = torchvision.datasets.MNIST(root='~/data/', train=True, download=True,
                                   transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

# Convert the PyTorch tensors to numpy arrays
X = mnist.data.numpy()
y = mnist.targets.numpy()

# Reshape the image data to be 2D rather than 1D (28x28 instead of 784)
X = X.reshape(X.shape[0], 1, 28, 28)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, random_state=1)

# Convert the numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train).to(torch.float32).to(device)
y_train = torch.tensor(y_train).to(torch.long).to(device)
X_test = torch.tensor(X_test).to(torch.float32).to(device)
y_test = torch.tensor(y_test).to(torch.long).to(device)

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

# In[13]:

bin_begin=-20
bin_end=20
bin_width=0.02
gaussianSigma=10*bin_width
gaussianCutoff=4*gaussianSigma
margin=0  #number of bins in a margin. No need for that if bin_begin and bin_end are sufficiently large

kT = 1.0
frictionCoefficient = 1e-5 #used in Langevin thermostat

#for regions with training loss larger than a threshold, we don't want to waste time studying
#use an initial value of entropy to bias the study toward the lower-train-loss region
maxX_WLToStudy = 2.0
minX_WLToStudy = -7.0
forbiddenRegionEntropyIncrease = 3000

#Wang-Landau factor, it first grows linearly, after reaching the highest value, it decays as 1/t
WLFactorMax=3
WLFactorRiseUntil=2e5
WLFactorDropStart=10e6
WLFactorDropPartOffset=8e6

time_step=3e-6
smoothed_accuracy_slope=5



bin_num = int((bin_end-bin_begin)/bin_width+2*margin+2)

binEdges = (np.array(range(bin_num))-margin)*bin_width+bin_begin
xx, yy = np.meshgrid(binEdges, binEdges)
if process_id == 0 and start_epoch == 0:
    np.savez_compressed("../XAndY_optimized.npz", x=xx, y=yy)

#for regions with training loss larger than a threshold, we don't want to waste time studying
#use an initial value of entropy to bias the study toward the lower-train-loss region
entropyBias = forbiddenRegionEntropyIncrease*(xx>maxX_WLToStudy)*(xx-maxX_WLToStudy)*(xx-maxX_WLToStudy)
entropyBias += forbiddenRegionEntropyIncrease*(xx<minX_WLToStudy)*(minX_WLToStudy-xx)*(minX_WLToStudy-xx)


# In[22]:

# if there is pre-saved entropy data, load it, otherwise initialize it with 0
# note here we are reading the averaged entropy in the previous folder (the folder that we run the bash script)
if os.path.isfile('../ave_entropy.pickle'):
    with open('../ave_entropy.pickle', 'rb') as pickled_file:
        entropy = pickle.load(pickled_file)
else:
    entropy = torch.zeros((bin_num, bin_num))


# In[43]:

# same as entropy, but for velocity, we do not need to initialize it manually
if os.path.isfile('velocity.pickle'):
    with open('./velocity.pickle', 'rb') as pickled_file:
        velocity = pickle.load(pickled_file)
else:
    velocity = None


# In[24]:

# we always want to reset histogram when run a new iteration
histogram = np.zeros((bin_num, bin_num))


# In[36]:


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


nParameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total number of parameters is: ' + str(nParameters))
langevinForceStddev = np.sqrt(frictionCoefficient * (2-frictionCoefficient) * kT)

# if there is previous saved net, we load it, otherwise we initialize one
if os.path.isfile('net.pth'):
    net.load_state_dict(torch.load('./net.pth'))
else:
    net.to(torch.float32)
    net.apply(init_weights)

net = net.to(device)


# In[38]:


loss = nn.CrossEntropyLoss()
X_train_WL = X_train[:]
y_train_WL = y_train[:]
len(X_train_WL), len(y_train_WL)


# In[49]:


def Train_WLMD(net, X_train_WL, y_train_WL, entropy, entropyBias, histogram, numepochs, bin_num, bin_width, start_epoch, time_step=None, velocity=None):

    """
    #####
    Training a given network using the WLMD algorithm.
    #####

    Args:
        net: (Sequential) network
        X_train_WL: (Tensor) full training data, will be divided into train and validation set
        y_train_WL: (Tensor) labels of the training data, will be divided into train and validation set
        entropy: (np.array or Tensor) stores the entropy, data type depends on put it on cpu or gpu
        entropyBias: (np.array or Tensor) stores the bias of large entropy bins, thus the model is unlikely to visit this places
        histogram: (np.array or Tensor) stores the enhistogramtropy, data type depends on put it on cpu or gpu
        numepochs: (int) number of epochs
        bin_num: (int) number of bins
        bin_width: (float) width if each bin
        start_epoch: (int) number of the starting step
        time_step: (float) learning rate, if None, then it uses 1.0e-2
        velocity: (list) stores parameters' velocity (a Tensor) of each layer

    Returns:
        losses_train: (list) stores the train loss
        losses_test: (list) stores the test loss
        net: (Sequential) network
        entropy: (np.array or Tensor)
        histogram: (np.array or Tensor)
        velocity: (list) stores parameters' velocity (a Tensor) of each layer
        E_traj (list): stores the trajectory of system energy
    """

    # read the current simulation step
    print('Starting step is: ' + str(start_epoch))

    # entropy with the Bias
    entropy += entropyBias

    data_num = len(X_train_WL)
    print('Number of WL data: ' + str(data_num))
    cut = data_num // 2

    train_X = X_train_WL[:cut]
    train_y = y_train_WL[:cut]

    val_X = X_train_WL[cut:cut*2]
    val_y = y_train_WL[cut:cut*2]

    E_traj = []

    # initialize losses
    losses_train = []
    losses_test = []

    # MD param setting
    gau_1dloc=np.arange(-gaussianCutoff, 1.0000001*gaussianCutoff, bin_width)
    x, y=np.meshgrid(gau_1dloc, gau_1dloc)
    gau_dist = scipy.stats.multivariate_normal.pdf(np.dstack((x, y)), (0, 0), gaussianSigma**2) * bin_width
    gau_dist=gau_dist/np.sum(gau_dist)
    gau_dist=torch.tensor(gau_dist)
    gau_dist=gau_dist.to(device)
    entropy=entropy.to(device)
    detailQueue=deque()

    # do fft on GPU
    #gau_dist = cupy.array(gau_dist)

    # initially, we set all velocity to be zero
    if velocity is None:
        velocity = []
        for param in net.parameters():
            velocity.append(0 * param.data)

        for vi in range(len(velocity)):
            velocity[vi] = torch.normal(0, np.sqrt(kT), velocity[vi].shape)
            velocity[vi] = velocity[vi].to(device)
        #print(velocity)

    # set the bound for parametres
    para_bound = []
    for name, param in net.named_parameters():
        num_neuron = param.numel()
        if len(param.size()) > 1:
            num_neuron /= param.size()[1]

        para_bound.append(2 / np.sqrt(num_neuron))

    # set the learning rate
    if time_step is None:
        print("Error: time_step is not set. This version of program no longer has a default time_step.")
        exit(1)

    #scale_factor = 0.1

    def SL_grad(S, x_WL, y_WL, bin_width, bin_num):
        bin_id_x, bin_id_y = bin_id_cal_meta(x_WL, y_WL, bin_width, bin_num)
        y_grad = (S[bin_id_y+1][bin_id_x]-S[bin_id_y-1][bin_id_x]) / bin_width / 2
        x_grad = (S[bin_id_y][bin_id_x+1]-S[bin_id_y][bin_id_x-1]) / bin_width / 2
        return torch.as_tensor(x_grad, device=device), torch.as_tensor(y_grad, device=device)

    def bin_id_cal_meta(x_WL, y_WL, bin_width, bin_num):
        bin_id_x = torch.div(x_WL-bin_begin, bin_width, rounding_mode='floor')+margin
        bin_id_y = torch.div(y_WL-bin_begin, bin_width, rounding_mode='floor')+margin
        return int(bin_id_x), int(bin_id_y)

    def histogram_updates(histogram, x_WL, y_WL, bin_width, bin_num):
        bin_id_x, bin_id_y = bin_id_cal_meta(x_WL, y_WL, bin_width, bin_num)
        histogram[bin_id_y][bin_id_x] += 1
        return histogram

    def entropy_updates_meta(entropy, x_WL, y_WL, gau_dist, scale_factor, bin_width, bin_num):
        width = gau_dist.shape[0] // 2
        bin_id_x, bin_id_y = bin_id_cal_meta(x_WL, y_WL, bin_width, bin_num)
        entropy[bin_id_y-width:bin_id_y+width+1, bin_id_x-width:bin_id_x+width+1] += scale_factor * gau_dist
        return entropy

    #these two variables keep track of the range of kinetic energy between two calls of the "print" function for monitoring.
    minKineticEnergy = 1e30
    maxKineticEnergy = 0

    # loop over epochs
    sgm=torch.nn.Sigmoid()
    for epoch in range(start_epoch, start_epoch+numepochs):

        E_total = []
        for vel in velocity:  ## might be able to accerlerate
            E_total.append((vel**2).sum())
        kineticEnergy=0.5*sum(E_total)
        minKineticEnergy = min(kineticEnergy, minKineticEnergy)
        maxKineticEnergy = max(kineticEnergy, maxKineticEnergy)

        y_hat_train = net(train_X)
        loss_train=loss(y_hat_train, train_y)
        ln_loss_train = torch.log(loss_train)

        y_hat_val = net(val_X)
        loss_val=loss(y_hat_val, val_y)

        #this is normally-defined accuracy code written by GPT-4
        total = val_y.size(0)
        _, predicted = torch.max(y_hat_val.data, 1)
        correct = (predicted == val_y).sum().item()
        accuracy = correct / total

        #find a smoothed accuracy that should be fairly close to the normal accuracy
        value, index = torch.topk(input=y_hat_val, k=2, dim=1)
        correct = (val_y == index[:,0])
        highestWrongLogit = value[:,0] + correct*(value[:,1]-value[:,0]) # the highest logit among all wrong classes. If correct, then highestWrongLogit=value[:,1], otherwise, highestWrongLogit=value[:,0]
        correctLogit = y_hat_val[torch.arange(y_hat_val.size(0)), val_y]
        logit_diff=correctLogit - highestWrongLogit
        smoothed_accuracy=torch.mean(sgm(smoothed_accuracy_slope*logit_diff))
        smoothing_error=smoothed_accuracy-accuracy


        x_WL=ln_loss_train
        epsilon=1e-4
        y_WL=-torch.log(1.0/(epsilon+(1-2*epsilon)*smoothed_accuracy)-1.0)

        X_grad, Y_grad = SL_grad(entropy, x_WL, y_WL, bin_width, bin_num)
        combined_loss=-1*X_grad*x_WL-1*Y_grad*y_WL

        # get the val loss gradient, dL/dw
        net.zero_grad()
        combined_loss.backward()
        with torch.no_grad():
            m = 0
            for param in net.parameters():
                velocity[m] += param.grad * time_step
                velocity[m] -= frictionCoefficient*velocity[m] + langevinForceStddev*torch.randn_like(velocity[m], device=device)
                param += velocity[m] * time_step
                # upper bound
                zeroTensor=torch.tensor([0.0]).to(device)
                isAboveUB=torch.heaviside(param.data - para_bound[m], zeroTensor)
                isIncreasing=torch.heaviside(velocity[m], zeroTensor)
                isReflecting=isAboveUB*isIncreasing
                velocity[m]=velocity[m]-2*velocity[m]*isReflecting
                reflectedParam=2*para_bound[m]-param
                param+=isReflecting*(reflectedParam-param)
                # lower bound
                isBelowLB=torch.heaviside(-param.data - para_bound[m], zeroTensor)
                isDecreasing=1-isIncreasing
                isReflecting=isBelowLB*isDecreasing
                velocity[m]=velocity[m]-2*velocity[m]*isReflecting
                reflectedParam=-2*para_bound[m]-param
                param+=isReflecting*(reflectedParam-param)
                m += 1



        if epoch < WLFactorRiseUntil:
            scale_factor = WLFactorMax / WLFactorRiseUntil * epoch
        elif epoch < WLFactorDropStart:
            scale_factor = WLFactorMax
        else:                       # factor should scale as 1/t
            scale_factor = WLFactorMax * (WLFactorDropStart-WLFactorDropPartOffset) / (epoch-WLFactorDropPartOffset)

        histogram = histogram_updates(histogram, x_WL, y_WL, bin_width, bin_num)
        entropy = entropy_updates_meta(entropy, x_WL, y_WL, gau_dist, scale_factor, bin_width, bin_num)

        detail = f"epoch {epoch}: x_WL {x_WL.item():g}, y_WL {y_WL.item():g}, kinetic energy {minKineticEnergy:g}-{maxKineticEnergy:g}, WL factor {scale_factor:g}, smoothing_error {smoothing_error:g}, x_grad {X_grad:g}, y_grad {Y_grad:g}"
        detailQueue.append(detail)
        if len(detailQueue)>100:
            detailQueue.popleft()
        #if epoch>10000 and maxKineticEnergy > 10000:
        #    print("kinetic energy very high. Exiting! Recent detail is:")
        #    for s in detailQueue:
        #        print(s)
        #    quit()
        if epoch % 1000 == 0:
            E_traj.append(kineticEnergy)
            losses_train.append(loss_train.item())
            losses_test.append(loss_val.item())
            print(detail)
            minKineticEnergy = 1e30
            maxKineticEnergy = 0
            if epoch % 1000000 == 0 and process_id == 0:
                temp=np.copy(entropy.cpu())
                temp-=entropyBias
                temp[temp<1e-5]=0
                np.savez_compressed("../entropy"+str(epoch//1000000)+"M.npz", entropy=temp, histogram=histogram)

    entropy=entropy.to('cpu')
    entropy -= entropyBias
    return losses_train, losses_test, net, entropy, histogram, velocity, E_traj


# In[50]:

# running the WLMD, note here we can adjust the number of epochs as we need
train_loss_sum, test_loss_sum, net, entropy, histogram, velocity, E_traj = Train_WLMD(net,
        X_train_WL, y_train_WL, entropy, entropyBias, histogram, epoch_step+1, bin_num, bin_width,
        start_epoch, time_step=time_step, velocity=velocity)

# In[51]:


with open('./entropy.pickle', 'wb') as pickle_file:
    pickle.dump(entropy,  pickle_file)


# In[54]:


with open('./histogram.pickle', 'wb') as pickle_file:
    pickle.dump(histogram,  pickle_file)


# In[55]:


with open('./velocity.pickle', 'wb') as pickle_file:
    pickle.dump(velocity,  pickle_file)


# In[56]:


with open('./E_traj.pickle', 'wb') as pickle_file:
    pickle.dump(E_traj,  pickle_file)


# In[57]:


with open('./train_loss_sum.pickle', 'wb') as pickle_file:
    pickle.dump(train_loss_sum,  pickle_file)


# In[58]:


with open('./test_loss_sum.pickle', 'wb') as pickle_file:
    pickle.dump(test_loss_sum,  pickle_file)


# In[60]:


torch.save(net.state_dict(), './net.pth')


# In[ ]:
