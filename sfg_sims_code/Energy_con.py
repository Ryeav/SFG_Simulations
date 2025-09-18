import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.colors as mcolors
import scipy.constants as const
import getpass
import matplotlib as matplot
from scipy.fft import fft,ifft,fftfreq
from scipy.integrate import solve_ivp
from SellmeierParameters.Sellmeier_GroupVelocity import get_params,print_params
import datetime
from numba import njit,jit
from numba import prange
import math
import os
import gc

user = getpass.getuser()
max_cpus = os.cpu_count()

wavelength_s = 1.52
wavelength_p = 0.82
wavelength_UC =(wavelength_s*wavelength_p)/(wavelength_p + wavelength_s)
waves = [wavelength_p,wavelength_s,wavelength_UC]

     

def BR(w0, wavelength, z):                                               #Beamradius
    return w0 * np.sqrt( 1+ ( wavelength * z / ( np.pi * (w0**2 )) ) **2 )



#------------------------------------------------------------------------#


def Phi(wavelength, w0, z):                                              #Gouyphase
    return -np.arctan(wavelength*z/(np.pi * (w0**2)))  



#------------------------------------------------------------------------#

def R(w0, wavelength, z):                                                #Wavefront Curvature
    return z * ( 1 + ( (np.pi*w0**2) / (wavelength*z) )**2 )   


#------------------------------------------------------------------------#
def tenv_gauss(FWHM,tdelay,t):
     sigma = FWHM / (2 * np.sqrt(2*np.log(2)))  
     return 1* np.exp(-(t-tdelay)**2 / ( 4* (sigma**2) ) )

def tenv_sech(FWHM,tdelay,t):
        tau = FWHM / (2* np.log(1+np.sqrt(2)))#1.7627 # FWHM / ( 2 * ln(1+sqrt(2)) )
        
        
        return np.cosh((t-tdelay)/tau)**(-1)
#                                                                                       # Length Delay mapped to time domain == time delay  Ld = td * c                                                                                      



def AmpFunc( amp,k, wz, R, Phi, r):                        #Gaussian Amplitude Function for Signal and Pulse Beam
    return (amp/wz* np.exp(-(r**2)/wz**2 )) * (np.exp(1j*k*(r**2)/(2*R)))  * (np.exp(1j * Phi)) #*tenv(FWHM,tdelay,t)



#------------------------------------------------------------------------#
@njit(cache = True)
def nonlin_op(Delta_k, A,z,r_points,t_points):
    Lambda = 8.55e-6 
    pi = 3.14159265359
    phase_term = np.exp(-1j * (Delta_k) * z)
    poling_term =np.sign(np.sin(pi * 2*z / Lambda))
   
    len_Amp = r_points * t_points
    A_pump = np.reshape(A[:len_Amp],shape=(r_points,t_points))
    A_signal = np.reshape(A[len_Amp:2*len_Amp],shape=(r_points,t_points))
    A_SFG = np.reshape(A[2*len_Amp:],shape=(r_points,t_points))
    
    
    dA_pump =   np.conj(A_signal) * A_SFG * phase_term * poling_term 
         
   
    dA_signal =    np.conj(A_pump) * A_SFG * phase_term * poling_term 
        
   
    dA_SFG =    A_pump * A_signal * (phase_term)**(-1) * poling_term

    return dA_pump,dA_signal,dA_SFG



#@njit
def Grid(td,w0comb,L,T):
     
     groupvelos = get_params(T,waves)[1]
     vg_p,vg_s,vg_sfg = groupvelos.values()
     n_p,n_s,n_sfg  = get_params(T,waves)[2].values()
     wl_p = waves[0]*1e-6#0.82e-6
     wl_s = waves[1]*1e-6
     wl_sfg = (wl_s*wl_p)/((wl_p + wl_s))
     #L = 3e-3
     P_fwhm = 200e-15/1.76274717404
     S_fwhm = 200e-15/2.35482004503
     td_max = 0.5e-12
     w0_s,w0_p = w0comb
     #vg_p,vg_s,vg_sfg = 1,1,1
     w0_max = np.max(w0comb)
     w0_min = np.min(w0comb)
     z_r = np.max([np.pi*w0_p**2*n_p/wl_p,np.pi*w0_s**2*n_s/wl_s])
     z0 = L/2 - z_r
     ze = L/2 + z_r
     r_end =np.max([BR(w0_p,wl_p,-L/2 ),BR(w0_p,wl_p,-L/2 )])#w0_max * 8
     r_start =1e-7#5e-7
     dr = 5e-7#w0_min/15#w0_min/10
     r_points =int((r_end-r_start)/dr )+1#int(15)
     r_r = np.linspace(r_start,r_end,r_points)
     gd = (1/vg_p-1/vg_s)
     enve_p = 5 * P_fwhm
     enve_s = 5 * S_fwhm
     sigma_sum = P_fwhm + S_fwhm
     pad = 7* sigma_sum 
     #t_conv = #12*(P_fwhm+S_fwhm)+L*(204e-15)  -3    *(P_fwhm+S_fwhm)                    #in seconds
     dt0 = 0#z0/vg_p
     t0 = 0#-dt0
     t_start =-np.abs(L*gd) + np.abs(td) - pad  #td   - pad
     t_end =  np.abs(td) + pad#L*gd  + pad 
     t_conv = t_end - t_start
     dt =25e-15
     t_points =int((t_conv)/dt)+2
     t_r = np.linspace(t_start,t_end,t_points)
     t_r = np.ascontiguousarray(t_r)
     r_r = np.ascontiguousarray(r_r)
     #print(t_points)
     return r_r ,t_r, L   

def const_params(td,w0comb,L,T):
     '''
     Order : deff, dk, vg, n, wavelengths, c_0 Lightspeed, pi ~3.14...,r_step,t_step,dr,dt,r_range

     '''
     wl_p = waves[0]*1e-6#0.82e-6
     wl_s = waves[1]*1e-6
     wl_sfg = (wl_s*wl_p)/((wl_p + wl_s))
     Temp = T
     d_eff = 8e-12
     Delta_k = get_params(Temp,waves)[0]
     groupvelos = get_params(Temp,waves)[1]
     refrac = get_params(Temp,waves)[2]
     vg_p,vg_s, vg_sfg = groupvelos.values()
     n_p,n_s,n_sfg = refrac.values()
     gvd_p,gvd_s,gvd_sfg = get_params(Temp,waves)[3].values()
     r_range,t_range,L = Grid(td,w0comb,L,T)
     dr = r_range[1]-r_range[0]
     dt = t_range[1]-t_range[0]
     r_points, t_points= len(r_range), len(t_range)
     e0 = float(const.epsilon_0)
     c0 = float(const.c)
     pi = float(np.pi)
     #DrA = np.zeros((r_points,t_points),dtype='complex128')
     #DrA = np.ascontiguousarray(DrA)
     r_range = np.ascontiguousarray(r_range)
     
     return d_eff,Delta_k, vg_p,vg_s, vg_sfg,n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,t_points,dr,dt,r_range,t_range,gvd_p,gvd_s,gvd_sfg

def outside_params():
     Temp = 40
     P_fwhm = 200e-15#/1.76274717404
     S_fwhm = 200e-15#/2.35482004503
     return Temp,P_fwhm,S_fwhm


@njit
def loop_dt(A,r_points,t_points,dt):
     DtA = np.zeros_like(A)
     
     for i in range(r_points):
         DtA[i,-1] = (A[i,-1]-A[i,-2])/dt
         DtA[i,0] = (A[i,1]-A[i,0])/dt 
         for j in range(1,t_points-1):
              DtA[i,j] = (A[i,j+1]-A[i,j-1])/(2*dt)
     return DtA 
@njit
def vec_loop_dt(A,r_points, t_points,dt):
     DtA = np.zeros_like(A)
     for i in range(r_points):
         DtA[i*t_points] = (A[i*t_points+1]-A[i*t_points])/dt
         DtA[i*t_points-1] = (A[i*t_points-1]-A[i*t_points-2])/dt 
         for j in range(1,t_points-1):
              DtA[i*t_points+j] = (A[i*t_points+j+1]-A[i*t_points+j-1])/(2*dt)
         
     return DtA
@njit
def d_t(A,dt):
         
     DtA = np.zeros_like(A)
          
     DtA[:,1:-1] =  (A[:,2:] -A[:,:-2])/(2*dt)
     DtA[:,0]  = (A[:,1]-A[:,0])/dt
     DtA[:,-1]  = (A[:,-1]-A[:,-2])/dt
     return DtA



@njit
def looped_d2_r(A,r_points,t_points,dr,r_range):
     """
     Calculates Radial Dependence of Laplace Operator in cylindrical coordinates through central difference of second order,
     including boundary conditions  dr|rmin =0 --> von Neumann | Laplacian Term: ∇²A = ∂²A/∂r² + (1/r) * ∂A/∂r

     """
     
     DrA = np.zeros_like(A)
     
     for j in range(t_points):
          
          DrA[0, j] = 2 * (A[1, j] - A[0, j]) / dr**2           #      --> von Neumann boundary conditions
          #DrA[-1,j] =  (A[-1,j] - 2* A[-2,j] + A[-3,j])/(dr**2) + (1/r_range[-1]) * (A[-1,j]-A[-2,j])/(dr)  --> A|rmax = 0 for all times
          for i in range(1,r_points-1):
               inv_r = 1/r_range[i]
               DrA[i,j] = (A[i+1,j] - 2* A[i,j] + A[i-1,j])/(dr**2) + (inv_r) * (A[i+1,j]-A[i-1,j])/(2*dr) 
          
     return DrA

@njit(cache=True)
def looped_d2_r1(A,r_points,t_points,dr,r_range):
     """
     Calculates Radial Dependence of Laplace Operator in cylindrical coordinates through central difference of second order,
     including boundary conditions 

     """
     
     DrA = np.zeros_like(A)
    
     for j in range(t_points):
          DrA[0,j] =  (A[2,j] - 2* A[1,j] + A[0,j])/(dr**2) + (1/r_range[0]) * (A[1,j]-A[0,j])/(dr)
          DrA[-1,j] =  (A[-1,j] - 2* A[-2,j] + A[-3,j])/(dr**2) + (1/r_range[-1]) * (A[-1,j]-A[-2,j])/(dr)
     DrA[1:-1,:] =  ((A[2:,:] -2* A[1:-1,:]+ A[:-2,:])/(dr**2)
                            + (1/r_range[1:-1, None])*(A[2:,:]-A[:-2,:])/(2*dr)) 
          
     return DrA
@njit
def d_r(A,dr,r_range):
     """
     Calculates Radial Dependence of Laplace Operator in cylindrical coordinates through central difference of second order,
     including boundary conditions Dirichlet | radial Laplacian Term: ∇²A = ∂²A/∂r² + (1/r) * ∂A/∂r

     """
     DrA = np.zeros_like(A)
     DrA[1:-1,:] =  ((A[2:,:] -2* A[1:-1,:]+ A[:-2,:])/(dr**2)
                    + (1/r_range[1:-1, None])*(A[2:,:]-A[:-2,:])/(2*dr))
     DrA[0,:]  = (A[2,:]-2*A[1,:]+A[0,:])/dr**2 + (1/r_range[0])*(A[0,:]-A[1,:])/(2*dr)#2*(A[1,:]+A[0,:])/dr**2#
     #DrA[-1,:] = (A[-1,:]-2*A[-2,:]+A[-3,:])/dr**2 +  (1/r_range[-1])*(A[-1,:]-A[-2,:])/(2*dr)
     return DrA
@njit
def d2_t(A,dt):
     """
     Calculates second order temporal derivative for GVD

     """
     DtA = np.zeros_like(A)
     DtA[:,1:-1] =  (A[:,2:] -2* A[:,1:-1]+ A[:,:-2])/(dt**2)
               
     DtA[:,0]  = (A[:,2]-2*A[:,1]+A[:,0])/dt**2 
     DtA[:,-1] = (A[:,-1]-2*A[:,-2]+A[:,-3])/dt**2   
     return DtA


@njit
def DFsys(z,A,params):
     d_eff,Delta_k, vg_p,vg_s, vg_sfg,n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,t_points,dr,dt,r_range,t_range,gvd_p,gvd_s,gvd_sfg=params 
     
     len_Amp = r_points * t_points
     A_pump = np.reshape(A[:len_Amp],(r_points,t_points))
     A_signal = np.reshape(A[len_Amp:2*len_Amp],(r_points,t_points))
     A_SFG = np.reshape(A[2*len_Amp:],(r_points,t_points))
     
     kpump = 2 * pi / wl_p# *n_p
     ksig = 2 * pi / wl_s #*n_s
     ksfg = 2 * pi / wl_sfg #* n_sfg
     wsig = 2*pi*c0/wl_s
     wpump = 2*pi*c0/wl_p
     wsfg = 2*pi*c0/wl_sfg
     pre_pump = 2*wpump/(n_p*c0)#wpump/(2*const.c**2*kpump*n_p*const.epsilon_0)
     pre_sig = 2*wsig/(n_s*c0)#wsig/(2*const.c**2*ksig*n_s*const.epsilon_0)
     pre_sfg = 2*wsfg/(n_sfg*c0)#wsfg/(2*const.c**2*ksfg*n_sfg*const.epsilon_0)
     
     nonlin_pump,nonlin_sig,nonlin_SFG = nonlin_op(Delta_k,A,z, r_points,t_points)
     GD_pump = 0
     GD_signal = (1/vg_p - 1/vg_s)  * d_t(A_signal,dt)
     GD_SFG =    (1/vg_p- 1/vg_sfg) * d_t(A_SFG,dt)
     GVD_pump = gvd_p/2*d2_t(A_pump,dt)
     GVD_signal = gvd_s/2*d2_t(A_signal,dt)
     GVD_SFG = gvd_sfg/2*d2_t(A_SFG,dt)

     dA_pump =             +  1j*d_eff*pre_pump*nonlin_pump  + 1*1j/(2*kpump*n_p)  *d_r(A_pump,dr,r_range)   +1j*GVD_pump#* looped_d2_r(A_pump, r_points,t_points,dr,r_range)           
     dA_signal=  GD_signal +  1j*d_eff*pre_sig*nonlin_sig    + 1*1j/(2*ksig*n_s)   *d_r(A_signal,dr,r_range) +1j*GVD_signal#* looped_d2_r(A_signal, r_points,t_points,dr,r_range)  
     dA_SFG =    GD_SFG    +  1j*d_eff*pre_sfg*nonlin_SFG    + 1*1j/(2*ksfg*n_sfg) *d_r(A_SFG,dr,r_range)    +1j*GVD_SFG#* looped_d2_r(A_SFG, r_points,t_points,dr,r_range) 
     
     
     dA =np.zeros_like(A)
     
     for i in range(len_Amp):
          dA[i]  = dA_pump[i//t_points,i%t_points]
          dA[i+len_Amp] = dA_signal[i//t_points,i%t_points]
          dA[i+2*len_Amp] = dA_SFG[i//t_points,i%t_points]
     
     return dA

def init_grid(z0,r_range,t_range,tdelay,L,w0comb,T):
    w0_s,w0_p = w0comb
    params = const_params(tdelay,w0comb,L,T)
    r_range,t_range, L = Grid(tdelay,w0comb,L,T)
    Temp,P_fwhm,S_fwhm = outside_params()
    
    d_eff,Delta_k, vg_p,vg_s, vg_sfg,n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,t_points,dr,dt,r_range,t_range,gvd_p,gvd_s,gvd_sfg = params
    z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
    E_photon_s = const.Planck * c0/wl_s
    avg_power_pump = 4e-1
    pump_rep_rate = 7.64e7    
    P0_s = E_photon_s/ (S_fwhm/(2 * np.sqrt(2*np.log(2))) )#692.6e-9#7.6e6 * (c0/wl_s) * const.Planck #1e-12
    pump_sech = 1 # is pump tenv sech shaped?
    if pump_sech == 1:
     P0_p = avg_power_pump/pump_rep_rate /(P_fwhm/(2* np.log(1+np.sqrt(2))))
    else:
       P0_p = avg_power_pump/pump_rep_rate /(S_fwhm/(2 * np.sqrt(2*np.log(2))) ) 
    AMP0_s = np.sqrt(2*P0_s/(n_s*np.pi*e0 *c0*w0_s**2))### intital beam amplitudes 
    AMP0_p = np.sqrt(2*P0_p/(n_p*np.pi*e0 *c0*w0_p**2))
    
          
            #set up r x t grid and calculate initial conditions + boundary conditions dr|r==0 = 0, A_j|r== rmax = 0
            #initial field amplitudes
    KSig = 2*pi/wl_s
    KPump = 2*pi/wl_p

    #z0 = -L/2
    gd =(1/vg_p-1/vg_s) 
    A_SFG = np.array([[complex(0,0) for _ in t_range] for _ in r_range])#np.zeros_like(r_range)#[A_SFG* np.exp(-r**2/w0_sfg**2) for r in r_range] #SFG at a given and r 
    #w0_s*AMP0_s, w0_p*AMP0_p
    A_s0 = np.array([[AmpFunc(AMP0_s*w0_s, KSig*n_s,  BR(w0_s, wl_s/n_s, z0), R(w0_s, wl_s/n_s, z0), Phi(wl_s/n_s, w0_s , z0), r) * tenv_gauss(S_fwhm,-tdelay,t) for t in t_range] for r in r_range])
    A_p0 = np.array([[AmpFunc(AMP0_p*w0_p, KPump*n_p,  BR(w0_p, wl_p/n_s, z0), R(w0_p, wl_p/n_p, z0), Phi(wl_p/n_p, w0_p, z0), r) * tenv_sech(P_fwhm,0,t) for t in t_range ] for r in r_range])
    A_signal = np.array([[complex(np.real(A_s0[i][j]),np.imag(A_s0[i][j])) for j in range(len(t_range))] for i  in range(len(r_range))])
    A_signal[-1,:] = complex(0,0) 
    A_pump = np.array([[complex(np.real(A_p0[i][j]),np.imag(A_p0[i][j])) for j in range(len(t_range))] for i  in range(len(r_range))])
    A_pump[-1,:] = complex(0,0) 
    
    Y0 = np.stack([A_pump,A_signal,A_SFG], axis = 0).flatten()
    
    Y0  = np.ascontiguousarray(Y0)
    return Y0

def energy_cons():
     return 0

def SomewhereIBelong(args):
    w0_s,w0_p,tdelay,L,T = args
    w0comb = [w0_s,w0_p] 
    params = const_params(tdelay,w0comb,L,T)
    r_range,t_range, L = Grid(tdelay,w0comb,L,T)
    Temp,P_fwhm,S_fwhm = outside_params()
    
    d_eff,Delta_k, vg_p,vg_s, vg_sfg,n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,t_points,dr,dt,r_range,t_range,gvd_p,gvd_s,gvd_sfg=params
    
    #z_range = np.linspace(-L/2,L/2,z_points+1)
    z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
    z0 = 0
    Y0= init_grid(z0-L/2,*Grid(tdelay,w0comb,L,T)[:-1],tdelay,L,w0comb,T)
###################################################################################################################################################################################################################
    ze = L
    dr_avg = dr#5e-7
    max_step =np.min([pi/wl_s *n_s *dr_avg**2,pi/wl_p *n_p *dr_avg**2])# pi*np.min([1/wl_s*n_s*(0.5*w0_s/10)**2,1/wl_p*n_p*(0.5*w0_p/10)**2])*1e-2
    steps =  math.ceil(L/max_step)
    
    slices = 35
    
    L_slice = (L/slices)
    
    int_0 = z0#-z_r
    int_e = ze
    L_int = int_e - int_0
    
    L_int_slice = L_int/slices
    #amp_evo = Y0
    for slice in range(slices):
          amp_evo = solve_ivp(DFsys, [int_0+ slice*L_int_slice, (slice+1)*L_int_slice], Y0, args = (params,),max_step = max_step, method ='RK23', t_eval= None,  dense_output= False).y[:,-1]#DOP853#BDF#LSODA#Radau#RK45#RK23  #max_step = 1e-1*dr**2*(2*np.pi/wl_s)
          Y0 = amp_evo
          del amp_evo
          gc.collect()
    
    
    
    sol_pump = []
    sol_signal = []
    sol_SFG = []
    
    
    linlen = len(r_range)*len(t_range)
    sol_pump = Y0[:linlen]
    sol_signal=Y0[linlen:2*linlen]
    sol_SFG=Y0[2*linlen:]

    A_pump = np.reshape(sol_pump, (r_points,t_points))
    A_signal =np.reshape(sol_signal, (r_points,t_points))
    A_SFG = np.reshape(sol_SFG, (r_points,t_points))
    Y01 = init_grid(z0-L/2,*Grid(tdelay,w0comb,L,T)[:-1],tdelay,L,w0comb,T)  
    A0_pump= np.reshape(Y01[:linlen],(r_points,t_points))
    A0_signal= np.reshape(Y01[linlen:2*linlen],(r_points,t_points))
    A0_SFG= np.reshape(Y01[2*linlen:],(r_points,t_points))
###################################################################################################################################################################################################################    
   
 
    #dr = r_end/r_points
    P_sfg = np.zeros(len(t_range))
    P_pump = np.zeros(len(t_range))
    P_signal = np.zeros(len(t_range))
    P0_sfg = np.zeros(len(t_range))
    P0_pump = np.zeros(len(t_range))
    P0_signal = np.zeros(len(t_range))
    E_sfg = 0
    E_pump = 0
    E_signal = 0
    E0_sfg = 0
    E0_pump = 0
    E0_signal = 0#np.zeros(z_points)
    E_sfg_z = 0
   
    Y01= init_grid(z0-L/2,*Grid(tdelay,w0comb,L,T)[:-1],tdelay,L,w0comb,T)    
    for e,t in enumerate(t_range):
            for n, r in enumerate(r_range):
                P_sfg[e] += 2*np.pi*abs(A_SFG[n][e])**2 *dr *r #(2*n+1)*dr**2 #*
                P_pump[e] += 2*np.pi*abs(A_pump[n][e])**2 *dr *r
                P_signal[e] += 2*np.pi*abs(A_signal[n][e])**2 *dr *r
                P0_sfg[e] += 2*np.pi*abs(A0_SFG[n][e])**2 *dr *r #(2*n+1)*dr**2 #*
                P0_pump[e] += 2*np.pi*abs(A0_pump[n][e])**2 *dr *r
                P0_signal[e] += 2*np.pi*abs(A0_signal[n][e])**2 *dr *r
            E_sfg+= P_sfg[e]*dt
            E_pump+= P_pump[e]*dt
            E_signal+= P_signal[e]*dt 
            E0_sfg+= P0_sfg[e]*dt
            E0_pump+= P0_pump[e]*dt
            E0_signal+= P0_signal[e]*dt  
    P0_s = 1e-12#7.6e6 * (c0/wl_s) * const.Planck #1e-12
    P0_p = 3e-1
    AMP0_s = 1 * np.sqrt(P0_s/(n_s*np.pi*e0 *c0* w0_s**2))### intital beam amplitudes 
    AMP0_p = 1 * np.sqrt(P0_p/(n_p*np.pi*e0 *c0* w0_p**2))
    P_pump_max = np.average(P_pump)
    P_signal_max = np.average(P_signal)
    P_SFG_max = np.average(P_sfg)
    P_total = P_pump_max + P_signal_max + P_SFG_max
    E0 = E0_sfg *n_sfg + E0_pump *n_p + E0_signal * n_s
    E_end = E_sfg * n_sfg + E_pump*n_p +E_signal * n_s 
    E0_s = E0_signal
    E_end_s = E_signal
    
    return E_end_s/E0_s#E_end/E0
#((P0_p + P0_s)*(t_range[-1]-t_range[0]) - e0*c0*(E_sfg *n_sfg + E_pump * n_p + E_signal * n_s))/((P0_p + P0_s)*(t_range[-1]-t_range[0]))#(P0_p+P0_s)-P_total#,[A_signal,A_pump,A_SFG], [A_s_prog, A_p_prog, A_sfg_prog]    

#def exec(td_range,w0_comb,L):

def exec(w0combs,td, L,Temp):
     cpus = int(max_cpus/2 )
     vg_p,vg_s,vg_sfg = get_params(Temp,waves)[1].values()
     td =-(1/vg_p - 1/vg_s) * L/2 * 1
     num_combs = len(w0combs)
     acc = int(np.sqrt(len(w0combs)))
     #w0comb = [40e-6,50e-6] #sig, pump
     #td_range = np.linspace(-1.2,0.5,acc)*1e-12#-L/(2*vg_p)
     params = [[*w0comb,td, L , Temp] for w0comb in w0combs]
     with Pool(cpus) as pool:
            with tqdm(
                total=num_combs, desc="Calculating E_sfg", ncols=100
            ) as bar:
                resultlin = []
                #refrac_dk.append( [Delta_K(T, waves) for T in T_range])
                
                for param in params:
                     result = pool.apply_async(SomewhereIBelong, [param], callback = lambda _ : bar.update() ) #parallel computing, asynchronously submits each combination individually    
                     resultlin.append(result)
                resultlin =[r.get() for r in resultlin]
                results = np.zeros((acc,acc))
                for j in range(num_combs):
                    results[j//acc][j%acc] = resultlin[j]


     return results
if __name__ == "__main__":
        matplot.rcParams.update({"font.size": 15})
        start_time = str(datetime.datetime.now())
        print(f'Starting Sim at {start_time[11:19]}')
        
        Temp,P_fwhm,S_fwhm = outside_params()
        acc =1
        w0p_list = np.linspace(5.5,5.5 , acc) *1e-6  # Range of w0_p values
        w0s_list = np.linspace(7.5,7.5 , acc) *1e-6 # Range of w0_s values
        
        w0combs = [[w0_s, w0_p] for w0_s in w0s_list for w0_p in w0p_list]
        Temp = 73
        L = 2e-3
        
        
        
        
        cost_params = const_params(0,w0combs[0],L,Temp)
        
        vg_p,vg_s,vg_sfg = get_params(Temp,waves)[1].values()
        td = -(1/vg_p-1/vg_s)*L/2
        print(td)
        r_range,t_range, L = Grid(td,w0combs[0],L,Temp)
        params = const_params(td,w0combs[0],L,Temp)
        d_eff,Delta_k, vg_p,vg_s, vg_sfg,n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,t_points,dr,dt,r_range,t_range,gvd_p,gvd_s,gvd_sfg=params
        print(f'time points {len(t_range)}, radius points {len(r_range)}, crystal Length {int(L*1e3)} mm')
        print_params(Temp)
        #z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
        #z0 = L/2-z_r
        results = exec(w0combs, td, L, Temp)
        print(results)
        #######
        np.save(rf'C:\\Users\\{user}\\SFG Simulation\\Sim_Data\\Energy_Temp{Temp}_crylen_{int(L*1e3)}_acc{int(acc)}_w0range{float(w0combs[0][1]),float(w0combs[0][1])}_{float(w0combs[-1][1]),float(w0combs[-1][1])}', results)
        
        
        
        
        
        

       
