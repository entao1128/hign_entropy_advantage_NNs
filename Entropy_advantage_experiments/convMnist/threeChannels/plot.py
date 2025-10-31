import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np
import pickle
import sys
from matplotlib.ticker import FuncFormatter

if len(sys.argv)!=2 and len(sys.argv)!=1:
    print("Usage: python3 plot.py (npz file)")
    print("Or: python3 plot.py #this will plot ave_entropy.pickle")
    quit()

if len(sys.argv)==2:
    data=np.load(sys.argv[1])
    entropy=data['entropy']
else:
    with open('ave_entropy.pickle', 'rb') as f:
        entropy=np.array(pickle.load(f))


xy=np.load("XAndY_optimized.npz")
plotRange_xlb=650
plotRange_xub=1101
plotRange_ylb=500
plotRange_yub=1501

fig=plt.figure(figsize=(6,6))
ax = fig.subplots()
print("sum entropy={:.2e}, num of bins non-zero={:.2e}".format(np.nansum(entropy), np.sum(entropy>1e-4)))
print("sum entropy in interested region={:.2e}, num of bins non-zero={:.2e}".format(np.nansum(entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]), np.sum(entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]>1e-4)))
entropy[entropy<1e-4]=float('nan')
entropy=entropy-np.nanmax(entropy)
heatmap=ax.pcolormesh(xy['x'][plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub], (xy['y'][plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]), entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub], cmap=plt.cm.jet)
fig.colorbar(heatmap)

plt.xlabel('ln(train loss)')
plt.ylabel("val accuracy")


#find the highest-entropy state at each train loss, plot with magenta line
xx=[]
yy=[]
for i in range(plotRange_xlb, plotRange_xub):
    temp=entropy[:,i]
    if not np.isnan(temp).all():
        maxEntropyIndex=np.nanargmax(entropy[plotRange_ylb+100:plotRange_yub-100, i])+plotRange_ylb+100
        xx.append(xy['x'][0, i])
        yy.append((xy['y'][maxEntropyIndex, 0]))
ax.plot(xx, yy, marker=None, color='magenta')
np.savetxt('magentaLine.txt', np.array([xx, yy]).transpose())

epsilon=1e-4
def valAcc2Y(acc):
    return -np.log(1/(epsilon+(1-2*epsilon)*acc)-1)
data2=np.load("optimizedSGD/binnedLoss.npz")
y=valAcc2Y(data2['valAcc'])
yerrUp=valAcc2Y(data2['valAcc']+data2['valAccErrorBar'])-y
yerrDown=y-valAcc2Y(data2['valAcc']-data2['valAccErrorBar'])
yerr=np.maximum(yerrUp, yerrDown)

ax.errorbar(data2['lnTrainLoss'], y, yerr, capsize=3, color='black', marker="none", linestyle='none')

prefix=sys.argv[0].split('/')[-1].split('.')
if len(sys.argv)==2:
    filename=sys.argv[1]+prefix[0]+"_entropy.png"
else:
    filename=prefix[0]+"_entropy.png"

ytick_accuracy=np.array([1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999])
ytick_location=-np.log(1/(epsilon+(1-2*epsilon)*ytick_accuracy)-1)
ytick_labels=[str(x) for x in ytick_accuracy]
ax.set_yticks(ytick_location)
ax.set_yticklabels(ytick_labels)
plt.grid()


plt.savefig(filename)

