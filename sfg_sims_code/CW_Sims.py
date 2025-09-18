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
from SFG_Sim.Sellmeier_GroupVelocity import get_params,print_params 
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
#                                                                                       # Length Delay mapped to time domain == time delay  Ld = td * c                                                                                      



def AmpFunc( amp,k, wz, R, Phi, r):                        #Gaussian Amplitude Function for Signal and Pulse Beam
    return (amp/wz* np.exp(-(r**2)/wz**2 )) * (np.exp(1j*k*(r**2)/(2*R)))  * (np.exp(1j * Phi))



#------------------------------------------------------------------------#
@njit(cache = True)
def nonlin_op(Delta_k, A,z,r_points):
    Lambda = 8.55e-6 
    pi = 3.14159265359
    #Delta_k = 3.2 / 1e-3
    phase_term = np.exp(-1j * (Delta_k) * z)
    poling_term =np.sign(np.sin(pi * 2*z / Lambda))
   
    len_Amp = r_points 
    A_pump = np.reshape(A[:len_Amp],shape=(r_points))
    A_signal = np.reshape(A[len_Amp:2*len_Amp],shape=(r_points))
    A_SFG = np.reshape(A[2*len_Amp:],shape=(r_points))
    
    
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
     
     w0_s,w0_p = w0comb
     #vg_p,vg_s,vg_sfg = 1,1,1
     w0_max = np.max(w0comb)
     w0_min = np.min(w0comb)
     z_r = np.max([np.pi*w0_p**2*n_p/wl_p,np.pi*w0_s**2*n_s/wl_s])
     z0 = L/2 - z_r
     ze = L/2 + z_r
     #r_end =w0_max * 12
     r_end =np.max([BR(w0_s,wl_s,-L/2 ),BR(w0_p,wl_p,-L/2 )])*2
     r_start =1e-7
     dr = w0_min/10#w0_min/10
     r_points =int((r_end-r_start)/dr )+1#int(15)
     r_r = np.linspace(r_start,r_end,r_points)
    
     return r_r 

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
     r_range = Grid(td,w0comb,L,T)
     dr = r_range[1]-r_range[0]
     
     r_points= len(r_range)
     e0 = float(const.epsilon_0)
     c0 = float(const.c)
     pi = float(np.pi)
     #DrA = np.zeros((r_points,t_points),dtype='complex128')
     #DrA = np.ascontiguousarray(DrA)
     #r_range = np.ascontiguousarray(r_range)
     
     return d_eff,Delta_k, n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,dr,r_range

def outside_params():
     Temp = 73
     
     return Temp



@njit
def d_t(A,dt):
         
     DtA = np.zeros_like(A)
          
     DtA[:,1:-1] =  (A[:,2:] -A[:,:-2])/(2*dt)
     DtA[:,0]  = (A[:,1]-A[:,0])/dt
     DtA[:,-1]  = (A[:,-1]-A[:,-2])/dt
     return DtA



@njit
def d_r(A,dr,r_range):
     """
     Calculates Radial Dependence of Laplace Operator in cylindrical coordinates through central difference of second order,
     including boundary conditions Dirichlet | radial Laplacian Term: ∇²A = ∂²A/∂r² + (1/r) * ∂A/∂r

     """
     DrA = np.zeros_like(A)
     DrA[1:-1] =  ((A[2:] -2* A[1:-1]+ A[:-2])/(dr**2)
                    + (1/r_range[1:-1])*(A[2:]-A[:-2])/(2*dr))
     DrA[0]  = (A[2]-2*A[1]+A[0])/dr**2 + (1/r_range[0])*(A[0]-A[1])/(2*dr)
     #DrA[-1] = DrA[-1] = (A[-1]-2*A[-2]+A[-3])/dr**2 +  (1/r_range[-1])*(A[-1]-A[-2])/(2*dr)
     
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
     d_eff,Delta_k, n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,dr,r_range=params 
     
     len_Amp = r_points 
     A_pump = np.reshape(A[:len_Amp],(r_points))
     A_signal = np.reshape(A[len_Amp:2*len_Amp],(r_points))
     A_SFG = np.reshape(A[2*len_Amp:],(r_points))
     
     kpump = 2 * pi / wl_p# *n_p
     ksig = 2 * pi / wl_s #*n_s
     ksfg = 2 * pi / wl_sfg #* n_sfg
     wsig = 2*pi*c0/wl_s
     wpump = 2*pi*c0/wl_p
     wsfg = 2*pi*c0/wl_sfg
     pre_pump = 2*wpump/(n_p*c0)
     pre_sig = 2*wsig/(n_s*c0)
     pre_sfg = 2*wsfg/(n_sfg*c0)
     
     nonlin_pump,nonlin_sig,nonlin_SFG = nonlin_op(Delta_k,A,z, r_points)
    

     dA_pump =    1j*d_eff*pre_pump*nonlin_pump  + 1*1j/(2*kpump*n_p)  *d_r(A_pump,dr,r_range)               
     dA_signal=   1j*d_eff*pre_sig*nonlin_sig    + 1*1j/(2*ksig*n_s)   *d_r(A_signal,dr,r_range) 
     dA_SFG =     1j*d_eff*pre_sfg*nonlin_SFG    + 1*1j/(2*ksfg*n_sfg) *d_r(A_SFG,dr,r_range)       
     
     
     dA =np.zeros_like(A)
     
     for i in range(len_Amp):
          dA[i]  = dA_pump[i]
          dA[i+len_Amp] = dA_signal[i]
          dA[i+2*len_Amp] = dA_SFG[i]
     
     return dA

def init_grid(z0,r_range,tdelay,L,w0comb,T):
    w0_s,w0_p = w0comb
    params = const_params(tdelay,w0comb,L,T)
    r_range = Grid(tdelay,w0comb,L,T)
    
    
    d_eff,Delta_k, n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,dr,r_range = params
    z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
       
    P0_s = 1e-12#7.6e6 * (c0/wl_s) * const.Planck #1e-12
    P0_p = 3e-1
    AMP0_s = 1 * np.sqrt(2*P0_s/(n_s*np.pi*e0 *c0))### intital beam amplitudes 
    AMP0_p = 1 * np.sqrt(2*P0_p/(n_p*np.pi*e0 *c0))
    
          
    KSig = 2*pi/wl_s
    KPump = 2*pi/wl_p

    #z0 = -L/2
   
    A_SFG = np.array([complex(0,0) for _ in r_range])#np.zeros_like(r_range)#[A_SFG* np.exp(-r**2/w0_sfg**2) for r in r_range] #SFG at a given and r 
    #w0_s*AMP0_s, w0_p*AMP0_p
    A_s0 = np.array([AmpFunc(AMP0_s, KSig*n_s,  BR(w0_s, wl_s/n_s, z0), R(w0_s, wl_s/n_s, z0), Phi(wl_s/n_s, w0_s , z0), r)  for r in r_range])
    A_p0 = np.array([AmpFunc(AMP0_p, KPump*n_p,  BR(w0_p, wl_p/n_p, z0), R(w0_p, wl_p/n_p, z0), Phi(wl_p/n_p, w0_p, z0), r)  for r in r_range])
    A_signal = np.array([complex(np.real(A_s0[i]),np.imag(A_s0[i])) for i  in range(len(r_range))])
    A_signal[-1] = complex(0,0) 
    A_pump = np.array([complex(np.real(A_p0[i]),np.imag(A_p0[i]))  for i  in range(len(r_range))])
    A_pump[-1] = complex(0,0) 
    
    Y0 = np.stack([A_pump,A_signal,A_SFG], axis = 0).flatten()
    
    Y0  = np.ascontiguousarray(Y0)
    return Y0

def energy_cons():
     return 0

def SomewhereIBelong(args):
    w0_s,w0_p,tdelay,L,T = args
    w0comb = [w0_s,w0_p] 
    params = const_params(tdelay,w0comb,L,T)
    r_range = Grid(tdelay,w0comb,L,T)
    
    
    d_eff,Delta_k, n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,dr,r_range=params
    
    #z_range = np.linspace(-L/2,L/2,z_points+1)
    z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
    z0 = 0
    Y0= init_grid(-L/2,Grid(tdelay,w0comb,L,T),tdelay,L,w0comb,T)
###################################################################################################################################################################################################################
    ze = L
    dr_avg = dr#5e-7
    max_step =np.min([pi/wl_s *n_s *dr_avg**2,pi/wl_p *n_p *dr_avg**2])# pi*np.min([1/wl_s*n_s*(0.5*w0_s/10)**2,1/wl_p*n_p*(0.5*w0_p/10)**2])*1e-2
    steps =  math.ceil(L/max_step)
    
    slices = 25
    
    L_slice = (L/slices)
    
    int_0 = z0#-z_r
    int_e = ze
    L_int = int_e - int_0
    
    L_int_slice = L_int/slices
    #amp_evo = Y0
    for slice in range(slices):
          amp_evo = solve_ivp(DFsys, [int_0+ slice*L_int_slice, int_0+(slice+1)*L_int_slice], Y0, args = (params,),max_step = max_step, method ='RK23', t_eval= None,  dense_output= False).y[:,-1]#DOP853#BDF#LSODA#Radau#RK45#RK23  #max_step = 1e-1*dr**2*(2*np.pi/wl_s)
          Y0 = amp_evo
          del amp_evo
          gc.collect()
    
    
    
    sol_pump = []
    sol_signal = []
    sol_SFG = []
    
    
    linlen = len(r_range)
    sol_pump = Y0[:linlen]
    sol_signal=Y0[linlen:2*linlen]
    sol_SFG=Y0[2*linlen:]

    A_pump = np.reshape(sol_pump, (r_points))
    A_signal =np.reshape(sol_signal, (r_points))
    A_SFG = np.reshape(sol_SFG, (r_points))  

###################################################################################################################################################################################################################    
   
 
    #dr = r_end/r_points
 
    P_sfg = 0#np.zeros((len(t_range)))
    E_sfg = 0
    
    for n, r in enumerate(r_range):
            P_sfg += 2*np.pi*np.abs(A_SFG[n])**2 *dr *r #(2*n+1)*dr**2 #*
            
    #P_max = np.max(P_sfg)    
    

    return P_sfg#,[A_signal,A_pump,A_SFG], [A_s_prog, A_p_prog, A_sfg_prog]    

#def exec(td_range,w0_comb,L):

def exec(w0combs,td, L,Temp):
     cpus = int(max_cpus/1 )
     
     td =0#-(1/vg_p - 1/vg_s) * L/2 
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
                #refrac_dk.append( [Delta_K(T, waves) )
                
                for param in params:
                     result = pool.apply_async(SomewhereIBelong, [param], callback = lambda _ : bar.update() ) #parallel computing, asynchronously submits each combination individually    
                     resultlin.append(result)
                resultlin =[r.get() for r in resultlin]
                results = np.zeros((acc,acc))
                for j in range(num_combs):
                    results[j//acc][j%acc] = resultlin[j]


     return results
if __name__ == "__main__":
        matplot.rcParams.update({"font.size": 17})
        start_time = str(datetime.datetime.now())
        print(f'Starting Sim at {start_time[11:19]}')
        
        
        acc =10
        w0p_list = np.linspace(5,15, acc)  *1e-6  # Range of w0_p values
        w0s_list = np.linspace(5,15, acc)  *1e-6 # Range of w0_s values
        #### SIM PARAMETERS ########################
        
        
        w0combs = [[w0_s, w0_p] for w0_s in w0s_list for w0_p in w0p_list]
        Temp = 73
        L = 1e-3
        
        
        
        
        cost_params = const_params(0,w0combs[0],L,Temp)
        vg_p,vg_s,vg_sfg = get_params(Temp,waves)[1].values()
        td = 0#-(1/vg_p-1/vg_s)*L/2
        print(td)
        r_range = Grid(td,w0combs[0],L,Temp)
        params = const_params(td,w0combs[0],L,Temp)
        d_eff,Delta_k, n_p,n_s,n_sfg ,wl_p,wl_s,wl_sfg, e0,c0,pi,r_points,dr,r_range=params
        print(f'radius points {len(r_range)}, crystal Length {int(L*1e3)} mm')
        print_params(Temp)
        #z_r = np.max([pi*w0_p**2*n_p/wl_p,pi*w0_s**2*n_s/wl_s])
        #z0 = L/2-z_r
        
        results = exec(w0combs, td, L, Temp)
        
        
        #############################################
        np.save(rf'C:\\Users\\{user}\\SFG Simulation\\Sim_Data\\Vox_Pixel_Temp{Temp}_crylen_{int(L*1e3)}_acc{int(acc)}_w0range{float(w0combs[0][1]),float(w0combs[0][1])}_{float(w0combs[-1][1]),float(w0combs[-1][1])}', results)
        pic = results/np.max(results)
        # Plotting the result as an intensity map
        fig, ax = plt.subplots(figsize = (8,8.5))
        image = ax.imshow(pic, extent=[
                w0p_list.min()*1e6,
                w0p_list.max()*1e6,
                w0s_list.min()*1e6,
                w0s_list.max()*1e6],
                origin="lower",
                aspect = 'equal',
                cmap="viridis"#,
                #norm = norm,
                #interpolation = 'bilinear'
        )
        fig.colorbar(image , label="SFG Energy norm. [A.U.]",shrink = 0.77)
        #ax.set_title(r"$\omega_{signal,pump}$ "+r"$itallic{combination}$") #, Res. = $\pm${round((w0p_list.max()-w0p_list.min())*1e6/acc,2)} $\mu$m"
        ax.set_xlabel(r"$\omega_{pump}$"+ " " + "["+ r"$\mu$"+"m"+"]")
        ax.set_ylabel(r"$\omega_{sig} $" + " " +"["+ r"$\mu$"+"m"+"]")
        xmax,ymax = np.argmax(results)//acc, np.argmax(results)%acc
        Optimum = str(r"Max. Energy at:  $\omega_{0,\,signal}$ = " + fr" {round(w0s_list[xmax]*1e6,2)} $\mu$m, and "+ r"$\omega_{0,\,pump}$ = " + fr"{round(w0p_list[ymax]*1e6,2)}  $\mu$m")
        Optimumfull = str(fr"max. to be found at: result({xmax},{ymax}), corresponding to w0_signal = {w0p_list[xmax]*1e6} um and w0_pump = {w0p_list[ymax]*1e6} um  max. UpCon value: {np.max(results)}")
        bbox = {'facecolor': 'lightgray',  'pad': 5}
        plt.figtext(0.05,-0.15, Optimum, transform = ax.transAxes, verticalalignment = 'bottom',fontsize = 12, bbox = bbox )
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        print(Optimumfull, sep = '\n' )
        #plt.savefig(fr"C:\\Users\\{user}\\Vox_acc{acc}.png")
        #ending_t = time.time()
        #print(f"The simulation took {ending_t - starting_t} seconds, {(ending_t - starting_t)/60} ,minutes")
        print(fr"File to copy: 'C:\\Users\\{user}\\SFG Simulation\\Sim_Data\\Vox_Pixel_Temp{Temp}_crylen_{int(L*1e3)}_acc{int(acc)}_w0range{float(w0combs[0][1]),float(w0combs[0][1])}_{float(w0combs[-1][1]),float(w0combs[-1][1])}")
        plt.savefig(fr"C:\\Users\\{user}\\Vox_acc{acc}_Temp{Temp}_Len{L*1e3}.png")
        plt.show()
        
        
        
        
        

       
