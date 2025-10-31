import numpy as np

#binEdges=np.linspace(-4.0, 0.0, 41)
#sums=np.zeros(binEdges.shape)
#counts=np.zeros(binEdges.shape)

binBegin=-7.0
binEnd=1.0
binSize=0.1
binCount=int(np.ceil( (binEnd-binBegin)/binSize ))

sums=np.zeros((binCount,))
counts=np.zeros((binCount,))

sumLastAcc=0.0
countLastAcc=0.0
for i in range(1000):
    filename=str(i)+"/metrics.npz"
    print("reading "+filename)
    data=np.load(filename)
    ltl=np.log(data['train_loss'])
    lvl=data['test_acc']
    sumLastAcc+=lvl[-1]
    countLastAcc+=1.0
    for j in range(len(ltl)):
        if ltl[j]>binBegin and ltl[j]<binEnd:
            binId=int(np.floor((ltl[j]-binBegin)/binSize))
            sums[binId]+=lvl[j]
            counts[binId]+=1.0

data=[]
for i in range(binCount):
    if counts[i]>0:
        binCenter=binBegin+(i+0.5)*binSize
        binAverage=sums[i]/counts[i]
        print("count={}, ln train loss={}, average test acc={}".format(counts[i], binCenter, binAverage))
        data.append([binCenter, binAverage])

data=np.array(data)
np.savez_compressed("binnedLoss.npz", lnTrainLoss=data[:,0], valAcc=data[:,1])
print("average ending test accuracy={}".format(sumLastAcc/countLastAcc))

