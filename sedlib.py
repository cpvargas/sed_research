import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15
from datetime import datetime
from decimal import Decimal
from scipy import stats
from datetime import datetime
import subprocess
c = 299792458 #speed of light m s^-1
import extinction

from os.path import expanduser
home = expanduser("~")
libdir = home + '/sed_research'

def read_params(paramfile):
    """
    Reads the filter paramfile
    """
    params = {}
    params["Filters"] = {}
    with open(paramfile) as f:
        for line in f:
            #print line.split()
            if line[0] != "#":
                ls = line.split()
                params["Filters"][ls[0]] = {}
                params["Filters"][ls[0]]["lam_c"] = np.float(ls[1])
                params["Filters"][ls[0]]["delt_lam"] = np.float(ls[2])
                params["Filters"][ls[0]]["5sigma"] = np.float(ls[3])
                params["Filters"][ls[0]]["filename"] = libdir + "/Filters/" + ls[4]
                params["Filters"][ls[0]]["eazy_tag"] = np.float(ls[5])
                params["Filters"][ls[0]]["flux_err"] = np.float(ls[6])
    return params

class sed(object):
    "Spectral Energy Distribution Object"
    def __init__(self,paramfile,templatefile):
        params = read_params(paramfile)
        self.filters = params.get("Filters")
        template = templatefile
        self.loglam,self.lognuLnu = np.loadtxt(template,usecols=(0,1), delimiter=" ", unpack=True)
        self.nuLnu = np.power(10.,self.lognuLnu)
        self.lam = np.power(10.,self.loglam)
        self.template = self.nuLnu/self.lam #propto Flam at rest
        #Loading filters
        for Filter in self.filters:
            lam,trans = np.loadtxt(self.filters[Filter]["filename"],usecols =(0,1),delimiter=" ",unpack=True)
            #Pass wavelength from Angstrom to microns
            self.filters[Filter]["lam"] = lam*1e-4
            self.filters[Filter]["trans"] = trans
    
    def extinction(self,Av=1.0,Rv=5.0,applytotemp=True):
        """
        Applies calzetti00 extinction to nuLnu and
        it gives the option for the template
        """
        wave = self.lam*1e4
        ext = extinction.calzetti00(wave,Av,Rv)
        
        #Code is giving negative values below 0.001 
        #these values are set to 0
        mask = ext<0.001
        mask = np.array(~mask,dtype=int)
        ext = ext*mask
        
        self.maxval = np.sum(mask)
        self.ext = ext
        
        #Applying the extinction directly to nuLnu
        #since it's proportional to Fnu and Flam
        self.nuLnu = self.nuLnu*np.exp(-ext)
        
        #If we want to apply it to the template
        if applytotemp:
            self.template = self.template*np.exp(-ext)
        
    def plot_extinction(self):
        """
        Plots the actual extinction 
        """
        plt.xlabel(r"$\lambda$ $[\mu m]$")
        plt.ylabel(r"Extinction")
        plt.plot(self.lam[0:self.maxval],self.ext[0:self.maxval])
        
    def redshift(self,z):
        """
        Applies the redshift to the template
        """
        self.z = z
        #dividing nuLnu by D_L(z) in Mpc gives nuFnu in erg cm^-2 s^-1 
        #In loglog space the shape of nuFnu is not changed by redshifting only displaced 
        #to dimmer fluxes
        self.nuFnu = (self.nuLnu/(4.*np.pi*Planck15.luminosity_distance(z).value**2))
        #Same for the wavelength the redshift in loglog space only displace it to the right
        #To redshift the wavelength we only have to multiply by (1+z)
        self.lam_new = self.lam*(1.+z)
        #Fnu is nuFnu/nu = c*nuFnu/lam
        #Special care has to be taken with the units, lam is in microns and the speed of
        #light on m/s, there is also a conversion from the template to 1e-10 and from
        #erg cm^-2 s^-1 to Jy by a factor of 1e23
        self.Fnu = self.nuFnu*(self.lam_new*1e-6/c)*1e-10*1e23 #Fnu in Jy
        #Flam = c*Fnu/lam**2 we use units of erg cm^-2 s^-1 um^-1
        self.Flam = 1e6*c*(self.Fnu*1e-23)/self.lam_new**2
        #interpolation
        self.inuFnu = interp1d(self.lam_new,self.nuFnu)
        self.iFnu = interp1d(self.lam_new,self.Fnu)
        self.iFlam = interp1d(self.lam_new,self.Flam)
        
    def photometry(self):
        """
        Photometry of the SED using the filters available, gives the flux in fnu
        """
        self.SNR = []
        self.phot_err = []
        self.phot_c = []
        self.phot = []
        self.lam_eff = []
        self.lam_c = []
        self.lam_err = []
        self.tag = []
        self.eazytag = []
        self.det = [] #Detections of upper_limits
        #Gets the photometry of all the filters
        for Filter in self.filters:
            lam = self.filters[Filter]["lam"]
            trans = self.filters[Filter]["trans"]
            Fnu = self.iFnu(lam)
            Flam = self.iFlam(lam)
            
            #First we get the photometry in flam
            phot_lam = np.trapz(Flam*trans,lam)/np.trapz(trans,lam)
            
            #Then we get the lam_pivot
            lam_pivot = np.sqrt(np.trapz(trans,lam)/np.trapz(trans/lam**2,lam))
            
            #And finally the photometry in fnu
            #fnu = (lam_pivot**2)*flam/c
            phot_c = 1e17*(lam_pivot**2)*phot_lam/c
    
            self.phot_c.append(phot_c)
        
            #Calculating sigma_flux from mag to Jy    
            #fivesigma_mag = self.filters[Filter]["5sigma"]
            #sigma_flux = (10**(-(fivesigma_mag + 48.6)/2.5)*1e23)/5.
            
            #Flux error is in Jansky
            sigma_flux = self.filters[Filter]["flux_err"]
            
            #Random realization
            phot = np.random.normal(loc=phot_c,scale=sigma_flux)
            
            if phot<0:
                phot=0.
            
            #Calculating SNR
            SNR = phot/sigma_flux
            self.SNR.append(SNR)
            
            if SNR>3.:
                self.det.append("det")
                self.phot_err.append(sigma_flux)
                self.phot.append(phot)
            #For non detections
            else:
                self.det.append("up_lim")
                
                #Adding the treatment of 3/2
                phot = (3./2.)*sigma_flux
                phot_err = (3./2.)*sigma_flux
                
                #This value is not randomized
                self.phot.append(phot)
                self.phot_err.append(phot_err)
                
            self.lam_eff.append(lam_pivot)
            self.lam_c.append(self.filters[Filter]["lam_c"])
            self.lam_err.append(self.filters[Filter]["delt_lam"]/2.)
            self.tag.append(Filter)
            self.eazytag.append(self.filters[Filter]["eazy_tag"])
            
    def plot_sed(self,scale="normal",color="blue",title="y",linewidth=0.5):
        """
        Plots the current SED
        """ 
        plt.ylabel(r"$F_{\nu}$[Jy]")
        plt.xlabel("$\lambda$ [$\mu$m]")
        if scale=="normal":
            plt.plot(self.lam_new,self.iFnu(self.lam_new),color=color,linewidth=linewidth,label="{}".format(self.z))
        if scale=="loglog":
            plt.loglog(self.lam_new,self.iFnu(self.lam_new),color=color,label="{}".format(self.z))
        if title=="y":
            plt.title("z={}".format(self.z))
            
    def plot_filters(self,norm):
        """
        Plots the transmission filters 
        """
        for Filter in self.filters:
            lam = mysed.filters[Filter]["lam"]
            trans = mysed.filters[Filter]["trans"]*norm
            plt.plot(lam,trans,label = Filter)
            
    def export_phot(self,program="EAZY"):
        """
        Returns: photometry, photometry error and the 
        corresponding tags and eazytags of the filters
        """
        idx = np.argsort(np.asarray(self.lam_c))
        phot = np.asarray(self.phot)[idx]
        phot_err = np.asarray(self.phot_err)[idx]
        tag = np.asarray(self.tag)[idx]
        eazytag = np.asarray(self.eazytag)[idx]
        if program=="EAZY":
            return phot, phot_err, tag, eazytag
        
    def export_template(self,filename,folder):
        """
        Exports the template on EAZY format
        """
        with open(folder+filename,"w") as f:
            for i in range(len(self.lam)-1):
                f.write("{} {}\n".format(1e4*self.lam[i],self.template[i]))

