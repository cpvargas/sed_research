import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import Planck15
c = 299792458 #speed of light m s^-1

def fit_SED(SUv_MIr,FIR,PHOT,z):
    lam,f =np.loadtxt(SUv_MIr,comments="#", usecols=(2,1),unpack=True)
    F = lam*f*1.e10
    Lam = lam*1.e-4 #wavelength on microns
    lam,f =np.loadtxt(FIR,comments="#", usecols=(2,1),unpack=True)
    FIR_lam,FIR_f =np.loadtxt(FIR,comments="#", usecols=(0,1),unpack=True)
    #passing the lambda to emitted, since it is in rest frame, i.e. redshifting
    FIR_lam = FIR_lam*(1.+z)
    FIR_F = FIR_lam*FIR_f*10000000*1.e10
    #extrapolation of both sides (initial and final)
    FIR_lam_i = FIR_lam[0]-(FIR_lam[2]-FIR_lam[1])
    FIR_F_i = FIR_F[0:5].mean()
    L = len(FIR_lam)
    FIR_lam_f = FIR_lam[L-1]+(FIR_lam[L-2]-FIR_lam[L-3])
    FIR_F_f = FIR_F[L-5:L-1].mean()
    FIR_F = np.concatenate((FIR_F_i,FIR_F,FIR_F_f),axis=None)
    FIR_lam = np.concatenate((FIR_lam_i,FIR_lam,FIR_lam_f),axis=None)
    nu_phot,f_phot =np.loadtxt(PHOT,comments="#", usecols=(0,1),unpack=True) 
    F_phot = f_phot*1.e-23*nu_phot*1.e10
    lam_phot = (c/nu_phot)*1.e6
    #fit of the mm+ tail
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
            F_phot_s.append(F_phot[mask][0])
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
    lam_tot = np.concatenate((Lam,FIR_lam,xs),axis=None)
    F_tot = np.concatenate((F,FIR_F,yfit(xs)),axis=None)
    return lam_tot,F_tot

ARP_220_lam,ARP_220_F = fit_SED("ARP_220_S_Uv-MIr_bms2014.txt","ARP_220_S_FIR_b2008.txt","ARP_220_phot_NED.txt",0.01813)
MRK_331_lam,MRK_331_F = fit_SED("MRK_0331_S_Uv-MIr_bms2014.txt","MRK_0331_S_FIR_b2008.txt","MRK_0331_phot_NED.txt",0.01848)


plt.loglog(ARP_220_lam,ARP_220_F,"ro",markersize=0.5,label="ARP 220")
plt.loglog(MRK_331_lam,MRK_331_F,"bo",markersize=0.5,label="MRK 0331")
plt.ylabel("$\lambda F_\lambda$ ($10^{-10}$ $ergs$ $s^{-1}$ $cm^{-2}$)")
plt.xlabel("$\lambda$ ($\mu$m)")
plt.legend()
plt.show()





