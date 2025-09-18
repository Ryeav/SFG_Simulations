import numpy as np
import matplotlib.pyplot as plt
import math as m
from tqdm import tqdm
import scipy.integrate as integrate
import scipy.constants as const
import time
import multiprocessing as mp
import itertools
import matplotlib as matplot
import matplotlib.colors as mcolors

n_sig = 2.149
n_pump = 2.118
n_SFG = 2.20057

wavelength_s = 1.52
wavelength_p = 0.82
wavelength_UC =(wavelength_s*wavelength_p)/(wavelength_p + wavelength_s)

Wavelengths = {'Pump': wavelength_p, 'Signal': wavelength_s,'SFG': wavelength_UC}

waves = [wavelength_p,wavelength_s,wavelength_UC]

# Sellmeier values taken from the Paper: 'Temperature-Dependent Sellmeier Equation for Refractive Index of 1.0 mol % Mg-Doped Stoichiometric Lithium Tantalate', DOI:10.7567/JJAP.52.032601
# Webaddress: https://iopscience.iop.org/article/10.7567/JJAP.52.032601

def Sellmeier(wave,T):
         A = 4.54773
         B = 0.0774167
         C = 0.22025
         D = -0.0226143
         E = 2.39494
         F = 7.45352
         b_t =  4.23526 *1e-8  * (T + 273.15)**2  #Celsius
         c_t = -6.53227  *1e-8 * (T + 273.15)**2  #Celsius
         return np.sqrt(A + (B + b_t )/(wave**2 -(C + c_t)**2) + E/(wave**2 - F**2) + D * wave**2)


def PhaseMismatch(T,wlp,wls,wlsfg):
        n_p = Sellmeier(wlp,T)
        n_s = Sellmeier(wls,T)
        #wl_SFG = (wavelength_s/n_s*wavelength_p/n_p)/(wavelength_p/n_p + wavelength_s/n_s)
        n_sfg = Sellmeier(wlsfg,T)
        
        
        KPump= 2*m.pi * n_p/wlp *1e6
        KSig = 2*m.pi * n_s/wls *1e6
        KSFG = 2*m.pi* n_sfg/wlsfg *1e6
       
        Delta_K = KPump + KSig - KSFG
        DK = Delta_K - 2*m.pi/((8.55e-6))
        return Delta_K


def GroupVelocity(wave,T):
         A = 4.54773
         B = 0.0774167
         C = 0.22025
         D = -0.0226143
         E = 2.39494
         F = 7.45352
         b_t =  4.23526 *1e-8 * (T + 273.15)**2  #Celsius
         c_t = -6.53227  *1e-8 * (T + 273.15)**2  #Celsius
         w = 2*np.pi*const.c/wave 
         dn_dlambda = 1/2 * (Sellmeier(wave,T))**(-1) * (-(B + b_t )/(wave**2 -(C + c_t)**2)**2*2*wave- E/(wave**2 - F**2)**2*2*wave+2*D*wave)
         return const.c *(Sellmeier(wave,T)-wave*dn_dlambda)**(-1)


def GVD(wave,T):
         """
         Returns (beta_2) factor for GVD, wavelength (wave) in micro meters, temperature in celsius. 

         """

         A = 4.54773
         B = 0.0774167
         C = 0.22025
         D = -0.0226143
         E = 2.39494
         F = 7.45352
         b_t =  4.23526 *1e-8 * (T + 273.15)**2  #Celsius
         c_t = -6.53227  *1e-8 * (T + 273.15)**2  #Celsius
         w = 2*np.pi*const.c/wave 
         
         dfn_lambda = (-(B + b_t )/(wave**2 -(C + c_t)**2)**2*2*wave- E/(wave**2 - F**2)**2*2*wave+2*D*wave)
        
         
         d2fn_lambda2 = (2*(B+b_t)/(wave**2 +(C+c_t)**2)**3*4*wave**2-2*(B+b_t)/(wave**2 +(C+c_t)**2)**2+2*E/(wave**2-F**2)**3*4*wave**2-2*E/(wave**2-F**2)**2+2*D)#-2*wave*(-2*(B+b_t)/(wave**2 + (C+c_t)**2)**3*2*wave-2*E/(wave**2-F**2)**3*2*wave)#(2*(B + b_t )/(wave**2 -(C + c_t)**2)**3*2*wave  -(B + b_t )/(wave**2 -(C + c_t)**2)**2*2  + 2*  E/(wave**2 - F**2)**3*2*wave- E/(wave**2 - F**2)**2*2+2*D) 
         d2n_dlambda2 = -1/4 * Sellmeier(wave,T)**(-3) *dfn_lambda**2 + 1/2 * (Sellmeier(wave,T))**(-1) * d2fn_lambda2
         
         
         return ((wave *1e-6)**3/(2*np.pi*(const.c)**2))*d2n_dlambda2*1e12#1/const.c * (2* dn_dlambda * dw_lambda + w* (d2n_dlambda2*dw_lambda**2 + dn_dlambda*dw2_lambda)) 


def print_params(T):
        GroupVelocities = {}
        RefractiveIndices = {}
        dk = PhaseMismatch(T,*waves)
        
        for source, wave in Wavelengths.items():
                GroupVelocities[source] =float(GroupVelocity(wave,T))
                RefractiveIndices[source] = float(Sellmeier(wave,T))

        print(f'Phasemismatch: {dk} at Temp. {T}'+f', reference: Dk-k_0 = {dk -np.sign(dk)*2*np.pi/(8.55e-6)}',
              f'Group velocities: {GroupVelocities}; ', f'Refractive indices: {RefractiveIndices}')


def get_params(T,waves = [0.82,1.52,(1.52*0.82)/(0.82 + 1.52)]):
        """
        Input: Temp, [wl_p,wl_s,wl_sfg] --> List in micrometers!
        Returns Phasemismatch, Group Velocities (dictionary), Refractive Indices (dictionary), GVD (beta_2) factors (dictionary) --> in order of: Pump, Signal, SFG

        """
        GroupVelocities = {}
        RefractiveIndices = {}
        GVDs = {}
        
        dk = PhaseMismatch(T,*waves)
        Wavelengths = {'Pump': waves[0], 'Signal': waves[1],'SFG': waves[2]}
        for source, wave in Wavelengths.items():
                GroupVelocities[source] = GroupVelocity(wave,T)
                RefractiveIndices[source] = Sellmeier(wave,T)
                GVDs[source] = GVD(wave,T)
        
        return dk, GroupVelocities, RefractiveIndices, GVDs


