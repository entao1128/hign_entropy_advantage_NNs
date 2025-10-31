#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
#rom sklearn.model_selection import train_test_split
import os
import math

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

filename=input()
seed=int(input())

#current version uses 'binary_crossentropy', so only works with two classes
inputPath=os.path.expanduser("~/Synced/neuralNetworkWLMC/try10-moreSystematic/inputs/"+filename+".txt")
ifile=open(inputPath)
line=ifile.readline().split(" ")
nDataPerClass=int(line[0])
nClass=int(line[1])
line=ifile.readline().split(" ")
rMin=float(line[0])
rMax=float(line[1])
dThetaDR=float(line[2])
noise=float(line[3])
line=ifile.readline().split(" ")
line=ifile.readline().split(" ")
channels=list(map(int, line))[0:-1]

np.random.seed(seed)
tf.random.set_seed(2*seed)

def spirals():
    numData=nDataPerClass*nClass
    X=np.zeros(shape=[numData, 2])
    y=[]
    for i in range(nClass):
        y.append(np.ones(shape=[nDataPerClass])*i)
    y=np.concatenate(y)
    
    r=np.random.rand(numData)*(rMax-rMin)+rMin
    theta=dThetaDR*r+(2*math.pi*y)/nClass
    X[:,0]=r*np.cos(theta)+np.random.normal(loc=0.0, scale=noise, size=[numData])
    X[:,1]=r*np.sin(theta)+np.random.normal(loc=0.0, scale=noise, size=[numData])
    
    y=tf.keras.utils.to_categorical(y)
    return X, y

X_train, y_train = spirals()
X_test, y_test = spirals()


training_epochs = 5000 # Total number of training epochs
learning_rate = 0.001 # The learning rate
def create_model():
    model = tf.keras.Sequential()
    # Input layer
    #model.add(tf.keras.layers.Dense(2, input_dim=2, activation='relu'))
    model.add(tf.keras.Input(shape=(2,)))
    for c in channels[0:-1]:
        model.add(tf.keras.layers.Dense(c, activation='relu'))
    model.add(tf.keras.layers.Dense(channels[-1], activation='softmax'))

    # Compile a model    
    model.compile(loss='binary_crossentropy', 
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                metrics=['accuracy'])
    return model

model = create_model()

model.summary()

results = model.fit(
 X_train, y_train,
 epochs= training_epochs,
 validation_data = (X_test, y_test),
 verbose = 0
)

(loss, accuracy) = model.evaluate(X_train, y_train, verbose=0)
print("train loss={:.4f}, train accuracy: {:.4f}%".format(loss,accuracy * 100))

(loss, accuracy) = model.evaluate(X_test, y_test, verbose=0)
print("test loss={:.4f}, test accuracy: {:.4f}%".format(loss,accuracy * 100))

plt.figure()
plt.plot(results.history['acc'])
plt.plot(results.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig("accuracies.pdf")

plt.figure()
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig("loss.pdf")

plt.figure()
plotRange=1.5*rMax
xx = np.linspace(-plotRange, plotRange, 400)
yy = np.linspace(-plotRange, plotRange, 400)
gx, gy = np.meshgrid(xx, yy)
Z = model.predict(np.c_[gx.ravel(), gy.ravel()])
Z = Z[:,0].reshape(gx.shape)
plt.contourf(gx, gy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

axes = plt.gca()
axes.set_xlim([-plotRange, plotRange])
axes.set_ylim([-plotRange, plotRange])
plt.grid('off')
plt.axis('off')

prediction_values = model.predict_classes(X_test)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test[:,0], marker='o', cmap=cm.coolwarm)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train[:,0], marker='X', cmap=cm.coolwarm)
plt.title('Model predictions, train (x), and test (o)')
plt.savefig("predictions.pdf")
