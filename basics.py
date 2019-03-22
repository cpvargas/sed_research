#The basics i.e. plotting the SEDs

import numpy as np
import matplotlib.pyplot as plt

#UV to Mid_IR

fname = "ARP_220_S_Uv-MIr_bms2014.txt"

#restframe lambda is col0 (Angstrom)
#F/lambda is col1 (ergs/s/cm^2/Angstrom)
lam,f =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

F = lam*f*1.e10

Lam = lam*1.e-4 #wavelength on microns

plt.loglog(Lam,F,"g-")

#Far infrared portion
fname = "ARP_220_S_FIR_b2008.txt"

FIR_lam,FIR_f =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

#lambda in microns
#f is F_lambda in W cm^-2 um^-1
#we get lambda F_lambda = nu F_nu multiplying by lambda
#and we pass Watts to ergs, W = 10000000 ergs s^-1

FIR_F = FIR_lam*FIR_f*10000000*1.e10

plt.loglog(FIR_lam,FIR_F,color="orange")


c = 299792458 #speed of light m s^-1

#Known photometry
fname = "ARP_220_phot_NED.txt"

nu_phot,f_phot =np.loadtxt(fname,comments="#", usecols=(0,1),unpack=True)

#changing units...
#f is in Jy = 10^-23 erg cm^-2

F_phot = f_phot*1.e-23*nu_phot*1.e10
lam_phot = (c/nu_phot)*1.e6 


plt.loglog(lam_phot,F_phot,"ro",markersize=0.5)

plt.ylabel("$\lambda F_\lambda$ ($10^{-10}$ $ergs$ $s^{-1}$ $cm^{-2}$)")
plt.xlabel("$\lambda$ ($\mu$m)")
plt.xlim(0.1,1e6)
plt.ylim(5*1.e-6,10*1e1)
plt.title("Arp 220 SED")
#plt.savefig("Arp_220_SED.png",dpi=300)
#plt.clf()
plt.show()


