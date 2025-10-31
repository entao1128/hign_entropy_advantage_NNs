#!/usr/bin/env python
# coding: utf-8

# In[61]:
import numpy as np
import pandas as pd
from sklearn import datasets
#import seaborn as sns
import matplotlib.pyplot as plt
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

width = 10

if (len(sys.argv) != 4):
    print('Usage: average_entropy.py start_epoch epoch_step process_id')
    sys.exit()

start_epoch = int(sys.argv[1])
epoch_step = int(sys.argv[2])
process_id = int(sys.argv[3])


# load and process the data
train_df = pd.read_csv('../General_training/Language_model/train_embedding_df.csv')
test_df = pd.read_csv('../General_training/Language_model/test_embedding_df.csv')

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

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


# WLMD parameters
bin_begin=-20
bin_end=20
bin_width=0.02
gaussianSigma=25*bin_width
gaussianCutoff=4*gaussianSigma
margin=0  #number of bins in a margin. No need for that if bin_begin and bin_end are sufficiently large

kT = 1.0
frictionCoefficient = 0.01 #used in Langevin thermostat

#for regions with training loss larger than a threshold, we don't want to waste time studying
#use an initial value of entropy to bias the study toward the lower-train-loss region
maxLogTrainLossToStudy = 0.0
maxLogTestLossToStudy = 5.0
forbiddenRegionEntropyIncrease = 100000

#Wang-Landau factor, it first grows linearly, after reaching the highest value, it decays as 1/t
maxWLFactor=20
maxWLFactorStep=1e6

time_step=3e-5


bin_num = int((bin_end-bin_begin)/bin_width+2*margin+2)

binEdges = (np.array(range(bin_num))-margin)*bin_width+bin_begin
xx, yy = np.meshgrid(binEdges, binEdges)
if process_id == 0:
    np.savez_compressed("../XAndY_optimized.npz", x=xx, y=yy)

#for regions with training loss larger than a threshold, we don't want to waste time studying
#use an initial value of entropy to bias the study toward the lower-train-loss region
entropyBias = forbiddenRegionEntropyIncrease*(yy>maxLogTrainLossToStudy)*(yy-maxLogTrainLossToStudy)*(yy-maxLogTrainLossToStudy)
entropyBias += forbiddenRegionEntropyIncrease*(xx>maxLogTestLossToStudy)*(xx-maxLogTestLossToStudy)*(xx-maxLogTestLossToStudy)


# if there is pre-saved entropy data, load it, otherwise initialize it with 0
# note here we are reading the averaged entropy in the previous folder (the folder that we run the bash script)
if os.path.isfile('../ave_entropy.pickle'):
    with open('../ave_entropy.pickle', 'rb') as pickled_file:
        entropy = pickle.load(pickled_file)
else:
    entropy = torch.zeros((bin_num, bin_num))


# same as entropy, but for velocity, we do not need to initialize it manually
if os.path.isfile('velocity.pickle'):
    with open('./velocity.pickle', 'rb') as pickled_file:
        velocity = pickle.load(pickled_file)
else:
    velocity = None

if os.path.isfile('E_traj.pickle'):
    with open('E_traj.pickle', 'rb') as pickled_file:
        E_hist = pickle.load(pickled_file)
else:
    E_hist = []


# we always want to reset histogram when run a new iteration
histogram = np.zeros((bin_num, bin_num))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        #nn.init.normal_(m.weight, std=0.01)


# for the network, we always need to specifiy the structure (class)
net = nn.Sequential(nn.Linear(768, width), nn.ReLU(), nn.Linear(width, 1))

nParameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total number of parameters is: ' + str(nParameters))
langevinForceStddev = np.sqrt(frictionCoefficient * (2-frictionCoefficient) * kT)

# if there is previous saved net, we load it, otherwise we initialize one
if os.path.isfile('net.pth'):
    net.load_state_dict(torch.load('./net.pth'))
else:
    net.to(torch.float32)
    net.apply(init_weights)

net = net.cuda()


loss = nn.MSELoss()

'''
X_train_WL = X_train[:]
y_train_WL = y_train[:]
'''

X_train_WL = (X_train, X_test)
y_train_WL = (y_train, y_test)


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

    train_X = X_train_WL[0]
    train_y = y_train_WL[0]
    
    val_X = X_train_WL[1]
    val_y = y_train_WL[1]

    '''
    data_num = len(X_train_WL)
    cut = data_num // 2

    train_X = X_train_WL[:cut]
    train_y = y_train_WL[:cut]

    val_X = X_train_WL[cut:cut*2]
    val_y = y_train_WL[cut:cut*2]
    '''

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

    # do fft on GPU
    #gau_dist = cupy.array(gau_dist)

    # initially, we set all velocity to be zero
    if velocity is None:
        velocity = []
        for param in net.parameters():
            velocity.append(0 * param.data)

        for vi in range(len(velocity)):
            velocity[vi] = torch.normal(0, np.sqrt(kT), velocity[vi].shape)
            velocity[vi] = velocity[vi].cuda()
        #print(velocity)

    # set the bound for parametres
    para_bound = []
    for name, param in net.named_parameters():
        num_neuron = param.size()[0]
        para_bound.append(3 / np.sqrt(num_neuron))
        #print(para_bound)
        #break

    # set the learning rate
    if time_step is None:
        print("Error: time_step is not set. This version of program no longer has a default time_step.")
        exit(1)

    #scale_factor = 0.1

    def SL_grad(S, ln_loss_train, ln_loss_val, bin_width, bin_num):
        bin_id_train, bin_id_val = bin_id_cal_meta(ln_loss_train, ln_loss_val, bin_width, bin_num)
        train_grad = (-1*S[bin_id_train-1][bin_id_val] +1*S[bin_id_train+1][bin_id_val]) / bin_width / 2
        val_grad = (-1*S[bin_id_train][bin_id_val-1]+1*S[bin_id_train][bin_id_val+1]) / bin_width / 2
        #return train_grad, val_grad
        return torch.as_tensor(train_grad, device='cuda:0'), torch.as_tensor(val_grad, device='cuda:0')

    def bin_id_cal_meta(ln_loss_train, ln_loss_val, bin_width, bin_num):
        bin_id_train = torch.div(ln_loss_train-bin_begin, bin_width, rounding_mode='floor')+margin
        bin_id_val = torch.div(ln_loss_val-bin_begin, bin_width, rounding_mode='floor')+margin
        #bin_id_train = max(bin_id_train, 0)  # do not need for now, but in the future, we may set a low limit for loss
        #bin_id_train = min(bin_id_train, bin_num - 2*margin-1)  # reserved for the outer regime
        #bin_id_val = max(bin_id_val, 0)  # do not need for now, but in the future, we may set a low limit for loss
        #bin_id_val = min(bin_id_val, bin_num - 2*margin-1)
        return int(bin_id_train), int(bin_id_val)

    def histogram_updates(histogram, ln_loss_train, ln_loss_val, bin_width, bin_num):
        bin_id_train, bin_id_val = bin_id_cal_meta(ln_loss_train, ln_loss_val, bin_width, bin_num)
        histogram[bin_id_train][bin_id_val] += 1
        return histogram

    def entropy_updates_meta(entropy, ln_loss_train, ln_loss_val, gau_dist, scale_factor, bin_width, bin_num):
        width = gau_dist.shape[0] // 2
        bin_id_train, bin_id_val = bin_id_cal_meta(ln_loss_train, ln_loss_val, bin_width, bin_num)
        entropy[bin_id_train-width:bin_id_train+width+1, bin_id_val-width:bin_id_val+width+1] += scale_factor * gau_dist
        return entropy

    #these two variables keep track of the range of kinetic energy between two calls of the "print" function for monitoring.
    minKineticEnergy = 1e30
    maxKineticEnergy = 0

    # loop over epochs
    for epoch in range(start_epoch, start_epoch+numepochs):

        E_total = []
        for vel in velocity:  ## might be able to accerlerate
            E_total.append((vel**2).sum())
        kineticEnergy=0.5*sum(E_total)
        minKineticEnergy = min(kineticEnergy, minKineticEnergy)
        maxKineticEnergy = max(kineticEnergy, maxKineticEnergy)

        y_hat_train = net(train_X)
        ln_loss_train = torch.log(loss(y_hat_train, train_y.reshape(y_hat_train.shape)))

        y_hat_val = net(val_X)
        ln_loss_val = torch.log(loss(y_hat_val, val_y.reshape(y_hat_val.shape)))

        train_grad, val_grad = SL_grad(entropy, ln_loss_train, ln_loss_val, bin_width, bin_num)

        # zero the net gradients
        net.zero_grad()
        # get the train loss gradient, dL/dw
        ln_loss_train.backward()
        S_grad = -1 * train_grad
        if S_grad == 0:
            S_grad = 0.5
        with torch.no_grad():
            m = 0 # order id of the parameter group
            for param in net.parameters():
                velocity[m] += param.grad * S_grad * time_step
                m += 1

        # get the val loss gradient, dL/dw
        net.zero_grad()
        ln_loss_val.backward()
        S_grad = -1 * val_grad
        if S_grad == 0:
            S_grad = 0.5
        with torch.no_grad():
            m = 0
            for param in net.parameters():
                velocity[m] += param.grad * S_grad * time_step
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



        if epoch < maxWLFactorStep:
            scale_factor = maxWLFactor / maxWLFactorStep * epoch
        else:                       # factor should scale as 1/t
            scale_factor = maxWLFactor * maxWLFactorStep / epoch

        histogram = histogram_updates(histogram, ln_loss_train, ln_loss_val, bin_width, bin_num)
        entropy = entropy_updates_meta(entropy, ln_loss_train, ln_loss_val, gau_dist, scale_factor, bin_width, bin_num)

        if epoch % 100 == 0:
            E_traj.append(kineticEnergy.cpu().item())
            losses_train.append(ln_loss_train.item())
            losses_test.append(ln_loss_val.item())
            if epoch % 10000 == 0:
                print(f"epoch {epoch+1}: ln(train loss) {ln_loss_train.item():g}, ln(test loss) {ln_loss_val.item():g}, kinetic energy {minKineticEnergy:g}-{maxKineticEnergy:g}, scale factor {scale_factor:g}, train_grad {train_grad:g}, val_grad {val_grad:g}")
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

E_hist = E_hist + E_traj
with open('./E_traj.pickle', 'wb') as pickle_file:
    pickle.dump(E_hist,  pickle_file)


# In[57]:


with open('./train_loss_sum.pickle', 'wb') as pickle_file:
    pickle.dump(train_loss_sum,  pickle_file)


# In[58]:


with open('./test_loss_sum.pickle', 'wb') as pickle_file:
    pickle.dump(test_loss_sum,  pickle_file)


# In[60]:


torch.save(net.state_dict(), './net.pth')


# In[ ]:
