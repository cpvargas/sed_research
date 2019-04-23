#small script to plot all filter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import os

directory = os.getcwd()
filters = []

for filename in os.listdir(directory):
    if filename.endswith(".dat"): 
        #print(os.path.join(directory, filename))
        filters.append(filename)
        continue
    else:
        continue

fig = figure(num=None, figsize=(24, 4), dpi=72, facecolor='w', edgecolor='k')

for F in filters:
    x,y = np.loadtxt(F,usecols=(0,1),unpack=True)
    x = x*0.0001
    plt.semilogx(x,y,label = F)

plt.xlabel("Wavelength ($\mu$m)",labelpad=-5)
plt.ylabel("Transmission")
plt.xlim(0.34,650)
#plt.legend()
plt.savefig("Filters.png",dpi=160)
