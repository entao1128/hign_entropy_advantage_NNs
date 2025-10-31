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
plotRange_xub=1251
plotRange_ylb=650
plotRange_yub=1251

fig=plt.figure(figsize=(6,6))
ax = fig.subplots()
print("sum entropy={:.2e}, num of bins non-zero={:.2e}".format(np.nansum(entropy), np.sum(entropy>1e-4)))
print("sum entropy in interested region={:.2e}, num of bins non-zero={:.2e}".format(np.nansum(entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]), np.sum(entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]>1e-4)))
entropy[entropy<1e-4]=float('nan')
entropy=entropy-np.nanmax(entropy)
heatmap=ax.pcolormesh(xy['x'][plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub], (xy['y'][plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub]), entropy[plotRange_ylb:plotRange_yub,plotRange_xlb:plotRange_xub], cmap=plt.cm.jet)
fig.colorbar(heatmap)

plt.xlabel('ln(train loss)')
plt.ylabel("ln(val loss)")


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

data2=np.load("optimizedAdam/binnedLoss.npz")
ax.plot(data2['lnTrainLoss'], data2['lnTestLoss'], color='black')
ax.set_xlim([-5, -0.5])

prefix=sys.argv[0].split('/')[-1].split('.')
if len(sys.argv)==2:
    filename=sys.argv[1]+prefix[0]+"_entropy.png"
else:
    filename=prefix[0]+"_entropy.png"

plt.savefig(filename)

