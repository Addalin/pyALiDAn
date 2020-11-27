# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:48:50 2019

@author: addalin
"""

from datetime import datetime
import os,inspect
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sprs
from scipy.linalg import block_diag
import pandas as pd
#import mpld3
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches

wDir = r'C:\Users\addalin\Dropbox\Lidar\code'
sys.path.append(wDir)
#sys.path.append(wDir+'\lidar_molecular\molecular\rayleigh_scattering')
os.chdir(wDir)
from generate_atmosphere import LidarProfile,RadiosondeProfile
import miscLidar as mscLid
from molecular import rayleigh_scattering  
from netCDF4 import Dataset



## %% Constants 
from global_settings import * #eps, c, h_plank, n_chan, min_height, CHANELS, LAMBDA

n = 500  # number of measurments
top_meas_height = 10 # [km]
heights = np.linspace(min_height, top_meas_height, num=n)  # Measurments heights [Km]

## %% Read Lidar measurments from .nc file
#'''Read lidar measuements from .nc files'''
#nc_path="C:\\Users\\hofer\\Documents\\haifa_170422\\2017_04_22_Sat_TROPOS_00_00_01.nc"
#ncfile = Dataset(nc_path,'r')
#raw_signal355 = ncfile.variables['raw_signal'][:,:,0]
#raw_signal532 = ncfile.variables['raw_signal'][:,:,4]
#raw_signal1064 = ncfile.variables['raw_signal'][:,:,7]


## %% Read Radiosonde measuements file and generate bacscatter and extiction molecular profiles.
'''Read Temprature and pressue from radiosounde file'''
soundePath = 'H:/data_haifa/Radiosonden/Haifa/2017/40179_20170422_12.txt'  # directory\stationNo_YYYYMMDD_hh.dat#
sounde_profile = RadiosondeProfile(soundePath)
temps = sounde_profile.interpolateKmKelvin(heights)     # Temprature [K]
pres = sounde_profile.interpolateKMPres(heights)        # Pressures [hPa]
relhs = sounde_profile.interpolateKMRLH(heights)        # Atmospheric relative humidity [%]
lambda_um = LAMBDA_nm.G  # Changing wavelength to [nm] units; these are the input requirments of "molecular" module.
sigma_mol = rayleigh_scattering.alpha_rayleigh(wavelength = lambda_um, pressure=pres, temperature=temps, C=385., rh=relhs)*1e+3 # converting from [1/m] to [1/km] 
beta_mol = rayleigh_scattering.beta_pi_rayleigh(wavelength = lambda_um, pressure=pres, temperature=temps, C=385., rh=relhs)*1e+3 # converting from [1/m] to [1/km sr] 

'''Visualize molecular frofiles''' 
vis_molecular = False 
if vis_molecular:
    lData  = {'x':heights,'y':sigma_mol,'lableX': 'Heights [km]','lableY': r"$\boldsymbol\sigma\quad[\rm 1/km\cdot sr]$"}
    rData  = {'x':heights,'y':beta_mol,'lableX': 'Heights [km]','lableY': r'$\boldsymbol\beta\quad[\rm{{1}/{km\cdot sr}]}$'}
    [fig, axes] = mscLid.visCurve(lData,rData)
    #fig.suptitle('Lidar readings from 22/4/17')
    
    ax = axes.ravel()
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[0].set_title(r'$\boldsymbol\sigma_{{\rm mol}}^{{\lambda={0:.0f}}}$'.format(lambda_um),fontsize=16,fontweight='bold')
    ax[1].set_title(r'$\boldsymbol\beta_{{\rm mol}}^{{\lambda={0:.0f}}}$'.format(lambda_um),fontsize=16,fontweight='bold')
    ax[0].set_xlim((0,heights.max()))
    ax[1].set_xlim((0,heights.max()))
    ax[0].set_ylim((0,sigma_mol.max()*1.1))
    ax[1].set_ylim((0,beta_mol.max()*1.1))
    plt.tight_layout()
    fig.canvas.draw()

## %% Read Lidar retrievals of Juilian, and regenarate lidar measuements (if .nc file is not exsist)
''' Load Lidar profiles from 22/4/17'''
retrivals_path = "./2017_04_22/*41smooth.txt"
lidar_paths = sorted(glob.glob(retrivals_path))[:4] 
profiles_sigma = []
profiles_beta = []
for path in lidar_paths:
    lidar_profile = LidarProfile(path)
    #min_height = lidar_profile._lidar_height
    #heights = np.linspace(min_height, top_meas_height, num = n) # num =1334 # num=100
    cur_profiles_0 = lidar_profile.interpolate(heights)[0]
    cur_profiles_1 = lidar_profile.interpolate(heights)[1]
    profiles_sigma.append(cur_profiles_0)
    profiles_beta.append(cur_profiles_1)

''' Total profiles values of beta and sigma : molecular + aerosol '''
beta_aer_orig = profiles_beta[0]  # [1/km sr] #  [0] is for choosing hour 8:00-9:00 
sigma_aer_orig = profiles_sigma[0] # [1/km]
betaT_orig = beta_aer_orig  + beta_mol 
sigmaT_orig = sigma_aer_orig + sigma_mol 
B_0 = 55    # [sr] - Lidar ratio . The value is taken from -20170422_0800-0900-41smooth-info.txt the lidar ratio for UV,G,IR is 55

# %%
'''Lidar constant. Note that A,P0,tau here are estimated. 
The lidar_const value should be be calculated following Rayleigh fit'''
tau = 50e-9 # [sec] temporal pulse width of the lidar
A = 1       # [km] - TODO : ask for this value 
P0 = 1e+15   # TODO : ask for this value 
#lidar_const = 0.5*P0*c*tau*A #1e+13
lidar_const= 5e+13 #lidar_const_optimized #5e+13  #1e+13 #lidar_const_optimized #1e+13# lidar_const_optimized#1e+13 #lidar_const_optimized #1e+13 #lidar_const_optimized
'''Calculate the power and the Logaritmic range adjusted power'''
P = mscLid.generate_P(P0,c,A,tau,heights,sigmaT_orig,betaT_orig)
P = P.round()
P[P<np.finfo(np.float).eps] = np.finfo(np.float).eps
S = mscLid.calc_S(heights,P)

'''Visualize the power and the Logaritmic range adjusted power'''
vis_power=False
if vis_power:
    lData  = {'x':heights,'y':P,'lableX': 'Heights [km]','lableY': r'$\boldsymbol{\rm P}\quad[\rm A.U.]$'}
    rData  = {'x':heights,'y':S,'lableX': 'Heights [km]','lableY': r'$\boldsymbol{\rm S}\quad[\rm A.U.]$'}
    [fig, axes] = mscLid.visCurve(lData,rData)
    #fig.suptitle('Power and the logaritmic range adjusted power based on readings from 22/4/17')
    ax = axes.ravel()
    P_max = 1.1*np.max(P[1:-1])
    P_min = -0.2e-3#-1.1*np.min(P[1:-1])
    ax[0].set_ylim((P_min,P_max)) 
    ax[0].set_xlim((0,np.max(heights))) 
    ax[1].set_ylim((S.min(),1.1*S.max()))
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    fig.canvas.draw()
## %%
'''KLETTT '''
#Calculate and show the inversion of the extinction and backscatter coefficients
#ind_m = 49#49#10 #49   # How to set the height rm and the values Sm ?? 46-49 doesn't work
n = len(heights)
ind_m = np.int(50.*n/100.0) # np.int(0.49*1334)
sigma_r = mscLid.calc_extiction_klett(S, heights, sigma_mol, ind_m)
k=1  #k is a fitting coefficient 0.67<k<1 (this is unknown in reality)

beta_r = (1./B_0)*(sigma_r-sigma_mol) + beta_mol #B_0*np.power(sigma_r,k)

'''Visualize Klett'''
vis_Klett=False
if vis_Klett:
    fnt_size = 12
    y_labels=[r'$\boldsymbol\sigma\;{\rm [1/km]}$',r'$\boldsymbol\beta\;{\rm [1/km \cdot sr]}$']
    x_labels = ['Heights [Km]','Heights [Km]']
    titles=[r'$\boldsymbol\sigma$',r'$\boldsymbol\beta$']
    labels = [[r'$\boldsymbol\sigma_{\rm Klett}$', r'$\boldsymbol\sigma_{\rm orig}$'],[r'$\boldsymbol\sigma_{\rm Klett}$',r'$\boldsymbol\beta_{\rm orig}$']]
    y = [[sigma_r,sigmaT_orig],[beta_r,betaT_orig]]
    x =  [[heights,heights],[heights,heights]]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax = axes.ravel()
    for (ax_i,y_i,x_i,label_i,title_i,y_label_i,x_label_i) in zip(axes,y,x,labels,titles,y_labels,x_labels):
        bby_i = [[],[]]
        bbx_i = [[],[]]
        for (y_j,x_j,label_j) in zip(y_i,x_i,label_i):
            ax_i.plot(x_j,y_j,label=label_j)    
            bby_i[0].append(min(y_j))
            bby_i[1].append(max(y_j))
            bbx_i[0].append(min(x_j))
            bbx_i[1].append(max(x_j))
        ax_i.set_ylabel(y_label_i)
        ax_i.set_ylabel(x_label_i)
        ax_i.set_xlim((min(bbx_i[0]),max(bbx_i[1])))
        ax_i.set_ylim((min(bby_i[0]),max(bby_i[1])))
        ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
        ax_i.legend()
        ax_i.legend()
        ax_i.set_title(title_i,fontsize=16,fontweight='bold')
        

## %% Create model matrices and functions
    
dr = np.mean(heights[1:]-heights[0:-1]) # dr for integration
D = dr*np.tril(np.ones((n,n)),0)
D[:,0] = heights[0]
D2 = -2*D
D2_inv = np.linalg.inv(D2)
rr =  np.power(heights,2) 
RR = np.diagflat(rr)
Q = np.linalg.inv(RR)

phi = lambda U_vec,D_mat:  np.exp(np.matmul(D_mat,U_vec))

## %% Solve regularized iterative LS 

'''set optimization flags and parameters'''
add_noise = False
WLS = False
add_Reg = True #False
lambda_reg = 0.125*0.125*0.125#2/2000.0
th_df = 10*eps    # The treshold value of l2 error differences (df) bewtween iterations 
max_iters = 1200

''' set input measurments, and initial solution'''
ind_m = np.int(50*n/100.0) #+128

if add_noise:
    # add noise to p
    stdPowerNoise = np.sqrt(P)
    PNoise = np.random.normal(loc=P, scale=stdPowerNoise, size=n)
    PNoise= PNoise.round()
    PNoise[PNoise<eps] = eps
    #std_read_noise = 0
    #poissonNoise = np.random.poisson(P).astype(float)
    #PNoise =P +  stdPowerNoise*np.random.randn(n)
    
    # set normalize mesurments - y
    Y = PNoise/lidar_const
    
    # set initial solution sigma_0
    SNoise = mscLid.calc_S(heights,PNoise)
    sigmaT_0 = mscLid.calc_extiction_klett(SNoise, heights, sigma_mol, ind_m) # TODO - the value of sigma_m should be actualy taken from the molecular profile
    #sigma_0 = 0.1*np.ones_like(Y)
    if WLS: 
        var_noise = P/(lidar_const**2)
        W = np.diag(1/var_noise)
        Ws = np.sqrt(W)
        
else:
    # set normalize mesurments - y
    Y = P/lidar_const
    # set initial solution sigma_0
    S = mscLid.calc_S(heights,P)
    #sigmaT_0 = mscLid.calc_extiction_klett(S, heights, sigmaT_orig, ind_m) 
    #sigma_0= U_sol1
    sigmaT_0 = sigma_mol #0.1*np.ones_like(Y)
    #sigma_0 = U_sol_interp

'''set initial solution beta_0, sigma_0 , V_0, and g_model at iteration 0'''
v_orig = phi(sigmaT_0,D2)

#beta_aer_0 =  B_0*(sigmaT_0 - sigma_mol)  # beta = beta_mol + beta_aer
#betaT_0 = beta_aer_0 + beta_mol
betaT_0 = beta_mol
V_0 = phi(sigmaT_0, D2)
Q_v =  np.matmul(np.diag(V_0),Q)

if add_Reg:
    Ld = mscLid.laplacian_operator(n,1,1)
    L = sprs.csr_matrix(Ld)
    Q_reg =  np.sqrt(lambda_reg)*L.todense()
    Q_v = np.concatenate((Q_v,Q_reg),axis=0)
    Y = np.asarray(np.concatenate([Y,np.zeros_like(Y)])).reshape((2*n,)) 
    
    model =  np.asarray(np.matmul(Q_v,betaT_0)[0,0:n]).reshape((n,))
    meas = Y[0:n].reshape((n,))
    g_model = meas - model 
else:
     g_model = Y - np.matmul(Q_v,betaT_0)
g_model = np.array(g_model)


if WLS: 
    g_model = np.matmul(Ws,g_model) 


f1_ls = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]

'''start iterations of LS'''
it = 0
df = f1_ls - 0
print np.abs(g_model).max()
print ('epoch {}, loss Ls1 {}'.format(it,f1_ls))   

'''open empty results figure'''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.ravel()
ax[0].plot(heights,sigmaT_0, label = r'$\boldsymbol\sigma_{{\rm total}}$ - {}'.format(it))
ax[1].plot(heights,betaT_0, label = r'$\boldsymbol\beta_{{\rm total}}$ - {}'.format(it))

while (np.abs(df)> th_df and it<=max_iters):
    if WLS: 
        Qv_t_Qv = np.matmul(np.matmul(Q_v.transpose(),W),Q_v)
        Qv_t = np.matmul(Q_v.transpose(),W)
    else:
        Qv_t_Qv =  np.matmul(Q_v.transpose(),Q_v)
        Qv_t = Q_v.transpose()
    
    '''LS solution '''    
    Q_v_inv = np.matmul(np.linalg.inv(Qv_t_Qv),Qv_t) 
    betaT_sol =  np.matmul(Q_v_inv,Y).reshape((n,))
    betaT_sol = np.asarray(betaT_sol).reshape((n,))
    '''Keeping equality and inequality constraints'''
    inds_negative = np.where(betaT_sol<beta_mol)
    betaT_sol[inds_negative]  = beta_mol[inds_negative]# non-negativity    
    beta_aer_sol =  betaT_sol - beta_mol 
    sigmaT_sol = (B_0)*beta_aer_sol + sigma_mol
    V_sol = phi(sigmaT_sol,D2)
    
    Q_v = np.diag(V_sol)*Q
    if add_Reg:
        Q_v = np.concatenate((Q_v,Q_reg),axis=0)
        model =  np.asarray(np.matmul(Q_v,betaT_sol)[0,0:n]).reshape((n,))
        g_model = meas  - model
    else:
        g_model = Y - np.matmul(Q_v,betaT_sol)
    g_model = np.array(g_model)
    if WLS: 
        g_model = np.matmul(Ws,g_model)

    f_ls_new = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]

    df = f1_ls- f_ls_new   # Differences of l2 error betwen adjacent iterations
    f1_ls = f_ls_new
    
    '''print and plot results - durint procces'''
    it+=1
    #print np.abs(g_model).max()
    #print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))   
    if (it==2):#(it % 200 == 0):
        
        ax[0].plot(heights,sigmaT_sol, label =  r'$\boldsymbol\sigma_{{\rm total}}$ - {}'.format(it))
        ax[1].plot(heights,betaT_sol, label =  r'$\boldsymbol\beta_{{\rm total}}$ - {}'.format(it))
        print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))
        

'''print and plot ls results'''            
if it<max_iters:
    ax[0].plot(heights,sigmaT_sol, label =  r'$\boldsymbol\sigma_{{\rm total}}$ - {}'.format(it))
    ax[1].plot(heights,betaT_sol, label =  r'$\boldsymbol\beta_{{\rm total}}$ - {}'.format(it))

 
## %%
fnt_size = 12
y_labels=[r'$\boldsymbol\sigma\;{\rm [1/km]}$',r'$\boldsymbol\beta\;{\rm [1/km \cdot sr]}$']
x_labels = ['Heights [Km]','Heights [Km]']
titles = [r'LS solutions of $\boldsymbol{\sigma_{\rm total}}$',r'LS solutions of $\boldsymbol{\beta_{\rm total}}$']
ax[0].plot(heights,sigmaT_orig, 'b--', label = r'$\boldsymbol\sigma_{\rm total}$ - original')
ax[1].plot(heights,betaT_orig, 'b--', label = r'$\boldsymbol\beta_{\rm total} $ - original')
ax[0].plot(heights,sigma_r, 'r--', label = r'$\boldsymbol\sigma_{\rm total}$ - Klett')
ax[1].plot(heights,beta_r, 'r--', label = r'$\boldsymbol\beta_{\rm total} $ - Klett')

for i,ax_i in enumerate(ax):
    ax_i.set_ylabel(y_labels[i], fontsize=fnt_size,fontweight='bold')
    ax_i.set_xlabel(x_labels[i], fontsize=fnt_size,fontweight='bold')
    ax_i.set_title(titles[i], fontsize=16,fontweight='bold')
    ax_i.set_xlim((0,heights.max()))
    ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax_i.legend()
    
(alpha_min,alpha_max) = (min(sigmaT_orig.min(),sigmaT_sol.min(),sigma_r.min()),max(sigmaT_orig.max(),sigmaT_sol.max(),sigma_r.max()))
(beta_min,beta_max) = (min(betaT_orig.min(),betaT_sol.min(),beta_r.min()),max(betaT_orig.max(),betaT_sol.max(),beta_r.max()))

ax[0].set_ylim((alpha_min,alpha_max*1.01))
ax[1].set_ylim((beta_min,beta_max*1.01))
plt.tight_layout()
fig.canvas.draw()
plt.show()

# %% find r_m
sigmas_heights = np.zeros((len(heights),n))
diff_sigma = np.zeros(n)
#diff_sigma2 = np.zeros(n)

for ind_m,h in enumerate(heights):
    sigma_m  = mscLid.calc_extiction_klett(S, heights, sigma_mol, ind_m)
    sigmas_heights[ind_m,:] = sigma_m
    diff = np.abs(sigma_m-sigmaT_sol).mean()
    diff_sigma[ind_m] = diff # 0.5*np.matmul(g_model.transpose(),g_model)
plt.figure()
plt.plot(heights,diff_sigma)
plt.plot(heights[1:],diff_sigma[0:-1]-diff_sigma[1:])
plt.yscale('log')
lowest3 = np.argsort(diff_sigma)[:3]
#prev_guess = [10,49,50]
#for ind in prev_guess: 
    #plt.axvline(x=heights[ind],color='0.5',linestyle='--',alpha=0.5)
#    plt.plot(heights[ind],diff_sigma[ind],'or')
plt.plot(heights[lowest3[0]],diff_sigma[lowest3[0]],'o',color='limegreen')
plt.xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
plt.grid(which='both',axis='x',linestyle='--')
plt.grid(which='major',axis='y',linestyle='--')
plt.xticks(np.arange(0, 11, step=1))
plt.xlim((0,10))
plt.ylabel(r'$\boldsymbol{\frac{1}{n}|\sigma_{\rm LS}-\sigma_{\rm Klett}|}$', fontsize=fnt_size,fontweight='bold')
plt.show()
