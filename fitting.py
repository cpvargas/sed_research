import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline

fname = "ARP_220_S_Uv-MIr_bms2014.txt"
lam,f =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

F = lam*f*1.e10

Lam = lam*1.e-4 #wavelength on microns

fname = "ARP_220_S_FIR_b2008.txt"

FIR_lam,FIR_f =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

FIR_F = FIR_lam*FIR_f*10000000*1.e10

#extrapolation of both sides (initial and final)
FIR_lam_i = FIR_lam[0]-(FIR_lam[2]-FIR_lam[1])
FIR_F_i = FIR_F[0:5].mean()
L = len(FIR_lam)
FIR_lam_f = FIR_lam[L-1]+(FIR_lam[L-2]-FIR_lam[L-3])
FIR_F_f = FIR_F[L-5:L-1].mean()

FIR_F = np.concatenate((FIR_F_i,FIR_F,FIR_F_f),axis=None)
FIR_lam = np.concatenate((FIR_lam_i,FIR_lam,FIR_lam_f),axis=None)

c = 299792458 #speed of light m s^-1

fname = "ARP_220_phot_NED.txt"

nu_phot,f_phot =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

F_phot = f_phot*1.e-23*nu_phot*1.e10
lam_phot = (c/nu_phot)*1.e6 

mask = lam_phot>200.

F_phot = F_phot[mask]
lam_phot = lam_phot[mask]

F_phot_s = []
lam_phot_s = []

bins = np.logspace(2.2,6,30)
for i in range(len(bins)-1):
    mask0 = lam_phot > bins[i]
    mask1 = lam_phot < bins[i+1]
    mean_bin = (bins[i]+bins[i+1])/2.
    mask = mask0*mask1
    if len(F_phot[mask])>1:
        F_phot_s.append(F_phot[mask].mean())
        lam_phot_s.append(mean_bin)
    if len(F_phot[mask])==1:
        F_phot_s.append(F_phot[mask])
        lam_phot_s.append(mean_bin)
    if len(F_phot[mask]==0):
        pass

F_phot_s = np.asarray(F_phot_s)
lam_phot_s = np.asarray(lam_phot_s)

xs = np.logspace(2.3,6,500)
interp = np.interp(xs,lam_phot_s,F_phot_s)

logx = np.log(lam_phot_s)
logy = np.log(F_phot_s)
interp = UnivariateSpline(logx, logy)
interp.set_smoothing_factor(0.5)
yfit = lambda x: np.exp(interp(np.log(x)))

plt.loglog(lam_phot_s,F_phot_s,"*",markersize=5.0)
plt.loglog(lam_phot,F_phot,"ro",markersize=0.5)
#plt.loglog(xs,spl(xs),"--")
#plt.loglog(xs,interp,"--")
plt.loglog(xs,yfit(xs),"--",color="gray")

plt.loglog(FIR_lam,FIR_F,"co",markersize=0.5)

plt.loglog(Lam,F,"go",markersize=0.5)

plt.show()





