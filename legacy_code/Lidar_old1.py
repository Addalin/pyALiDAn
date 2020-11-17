# -*- coding: utf-8 -*-

from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse as sprs
from scipy.linalg import block_diag
import os
import pandas as pd
from generate_atmosphere import LidarProfile
#import mpld3
import miscLidar as mscLid
from constsLidar import *
import matplotlib.patches as mpatches
eps = np.finfo(np.float).eps
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from matplotlib.ticker import FormatStrFormatter

# %%
''' Load readings of Lidar from 22/4/17'''

lidar_paths = sorted(glob.glob("./2017_04_22/*41smooth.txt"))[:4] 
profiles_alpha = []
profiles_beta = []

for path in lidar_paths:
    lidar_profile = LidarProfile(path)
    min_height = lidar_profile._lidar_height
    heights = np.linspace(min_height, 10,num=100) # num =1334 # num=100
    cur_profiles_0= lidar_profile.interpolate(heights)[0]
    cur_profiles_1= lidar_profile.interpolate(heights)[1]

    profiles_alpha.append(cur_profiles_0)
    profiles_beta.append(cur_profiles_1)

beta = profiles_beta[0] # choosing hour 8:00-9:00 
alpha = profiles_alpha[0] 

'''Visualize lidar readings''' 
vis_lidar = True 
if vis_lidar:
    lData  = {'x':heights,'y':alpha,'lableX': 'Heights [km]','lableY': r'$\boldsymbol\sigma\quad[\rm{{1}/{km\cdot sr}]}$'}
    rData  = {'x':heights,'y':beta,'lableX': 'Heights [km]','lableY': r'$\boldsymbol\beta\quad[\rm{{1}/{km\cdot sr}]}$'}
    [fig, axes] = mscLid.visCurve(lData,rData)
    #fig.suptitle('Lidar readings from 22/4/17')
    
    ax = axes.ravel()
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[0].set_title(r'$\boldsymbol\sigma_{\rm Original}$',fontsize=16,fontweight='bold')
    ax[1].set_title(r'$\boldsymbol\beta_{\rm Original}$',fontsize=16,fontweight='bold')
    ax[0].set_xlim((0,heights.max()))
    ax[1].set_xlim((0,heights.max()))
    ax[0].set_ylim((0,alpha.max()*1.1))
    ax[1].set_ylim((0,beta.max()*1.1))
    fig.canvas.draw()
    #fig.savefig('orig_coeffs.svg',bbox_inches='tight')
# %%
'''Constants and settings'''
A = 1   # TODO : ask for this value 
P0 = 100*200000  # TODO : ask for this value 

lambda_uv = 355e-9
lambda_g = 532e-9 # wavelength [nm]
lambda_ir = 1064e-9
J_photon = h_plank*c/lambda_uv

beta[beta<eps]= eps
alpha[alpha<eps] = eps
ind = np.where(alpha>10*eps)
B_0 =  (beta/alpha)[ind].mean()
# %%
'''Calculate the power and the Logaritmic range adjusted power'''
tau = 50e-9 # [sec]
P = mscLid.generate_P(P0,c,A,tau,heights,alpha,beta)
#P[P<eps] = eps
P= P.round()
P[P<eps] = eps
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

# %%
'''KLETTT '''
#Calculate and show the inversion of the extinction and backscatter coefficients
#ind_m = 49#49#10 #49   # How to set the height rm and the values Sm ?? 46-49 doesn't work
n = len(heights)
ind_m = np.int(50.*n/100.0) # np.int(0.49*1334)
sigma_r = mscLid.calc_extiction_klett(S, heights, alpha, ind_m)
k=1
beta_r = B_0*np.power(sigma_r,k)

'''Visualize Klett'''
vis_Klett=True
if vis_Klett:
    lData  = {'x':heights,'y':sigma_r,'lableX': 'Heights [km]','lableY': r'$\boldsymbol\sigma\quad[\rm{{1}/{km}]}$'}
    rData  = {'x':heights,'y':beta_r,'lableX': 'Heights [km]','lableY': r'$\boldsymbol\beta\quad[\rm{{1}/{km\cdot sr}]}$'}
    [fig, axes] = mscLid.visCurve(lData,rData)
    #fig.suptitle('Klett''s retrievals')
    
    ax = axes.ravel()
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[0].set_title(r'$\boldsymbol\sigma_{\rm Klett}$',fontsize=16,fontweight='bold')
    ax[1].set_title(r'$\boldsymbol\beta_{\rm Klett}$',fontsize=16,fontweight='bold')
    ax[0].set_xlim((0,heights.max()))
    ax[1].set_xlim((0,heights.max()))
    ax[0].set_ylim((0,sigma_r.max()*1.1))
    ax[1].set_ylim((0,beta_r.max()*1.1))
    fig.canvas.draw()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8))
ax = axes.ravel()
fnt_size = 14
ax[0].plot(heights,alpha, 'b--', label = r'Original $\boldsymbol\sigma$')
ax[1].plot(heights,beta, 'b--', label = r'$Original \boldsymbol\beta$')

ax[0].plot(heights,sigma_r, label = r'$\boldsymbol\sigma_{Klett}$ at $r_m=1.22[km]$')
ax[1].plot(heights,beta_r, label = r'$\boldsymbol\beta_{Klett}_{Klett}$ at $r_m=1.22[km]$')


ax[0].plot(heights,sigma_r,  label = r'$\boldsymbol\sigma_{Klett}$ at $r_m=5.06[km]$')
ax[1].plot(heights,beta_r, label = r'$\boldsymbol\beta_{Klett}_{Klett}$ at $r_m=5.06[km]$')

ax[0].plot(heights,sigma_r,  label = r'$\boldsymbol\sigma_{Klett}$ at $r_m=5.16[km]$')
ax[1].plot(heights,beta_r, label = r'$\boldsymbol\beta_{Klett}_{Klett}$ at $r_m=5.16[km]$')

ax[0].legend()
ax[1].legend()

# %% Model matrices and functions
    
n = len(heights)
dr = np.mean(heights[1:]-heights[0:-1]) # dr for integration
tau = dr*2/c  #[sec]  Is this resonable ? How to set the real value 
lidar_const = 0.5*P0*c*tau*A

'''Model matrices '''
D = dr*np.tril(np.ones((n,n)),0)
D[:,0]=heights[0]
D2 = -2*D
D2_inv = np.linalg.inv(D2)
rr =  np.power(heights,2) # rr = r^2 #one_v = np.ones((n,1)) #  np.array(np.dot(D,one_v)) 
RR = np.diagflat(rr)
Q = np.linalg.inv(RR)

'''model functions'''
#def phi(X,D=np.nan):
#    #if np.isnan(D):
#    #    D =  np.eye(np.size(X))
#    # model function
#    phi_f = np.exp(np.matmul(D,X))
#    grad_phi = np.matmul(np.diag(phi_f),D)
#    return [phi_f,grad_phi]
phi = lambda U_vec,D_mat:  np.exp(np.matmul(D_mat,U_vec))
phi_inv = lambda  V_vec,D_mat: np.matmul(np.linalg.inv(D_mat),np.log(V_vec))

grad_phi = lambda U_vec,D_mat:  np.matmul(np.diag(U_vec),D_mat)

def g3_U(U_vec,V_vec,Y,Q_mat,D_mat):  
    # model function
    #V = phi(U,D)[0] # V const
    diag_V = np.diag(V_vec)
    VQ = np.matmul(diag_V,Q_mat)
    #QV = np.matmul(Q,V)
    g = Y - np.matmul(VQ,U_vec)
    #g = Y - np.matmul(U.transpose(),QV)
    grad_g = -VQ 
    #grad_g = -QV   
    return [g,grad_g]

def g3_V(U_vec,V_vec,Y,Q_mat,D_mat):  
    # model function
    #[V,grad_V] = phi(U,-2*D)
    diag_U = np.diag(U_vec)
    UQ = np.matmul(diag_U,Q_mat)
    g = Y - np.matmul(UQ,V_vec)
    #QU = np.matmul(Q,U)
    g = Y - np.matmul(UQ,V_vec)    
    #g = Y - np.matmul(V.transpose(),QU)
    grad_V = grad_phi(V_vec,D_mat)
    grad_g = -np.matmul(grad_V,UQ)
    #grad_g = -np.matmul(grad_V,QU)
    return [g,grad_g]

def cost3(U,V,Y,Q,D,model):
    [g,grad_g] = model(U,V,Y,Q,D)
    f = 0.5*(g**2).mean()
    grad_f = np.matmul(grad_g,g)/len(U)
    return [f,grad_f]

'''optimization parameters '''
verbose =  True
rate =2e+1 #2E+1
minb = eps #eps - less then 1e-7 the GD is not converting
maxb = 1
epochs = 300000

# %% Alternating GD
'''setting input data'''

#stdPowerNoise = np.sqrt(P)
#PNoise =np.random.normal(loc=P, scale=stdPowerNoise, size=n) # normal(P,P)=normal(P,std=sqrt(P))

#Y = PNoise/(lidar_const*B_0)
#covmat = np.diag(np.power(stdPowerNoise/(lidar_const*B_0),2))
#Y = P_awgn/(lidar_const*B_0)

Y = P/(lidar_const*B_0)


''' initialization options of U:
    1. np.copy(sigma_r) - Klett's recovey
    2. np.copy(sigma_r_awgn) - Klett's recovey for noisy power (Adaptive White Gaussian Noise)
    3. 0.1*np.ones_like(Y)  - constant value
    4. U_sol1 -or-  U__sol_LS - solution of LS
    5. # 10*eps+ 0.5 + 0.4*np.sin(6*heights) - testing some crazy signal
    '''
# initialization U_0 using Klett
ind_m = 51 #49#10 #49   # referece range height
sigma_r = mscLid.calc_extiction_klett(S, heights, alpha, ind_m)
U = np.copy(sigma_r)  
V = phi(U,D2) 
U[U<minb] = minb


'''Setting output structures'''
loss1_epochs = np.zeros(epochs)
loss2_epochs = np.zeros(epochs)
temporal_epochs = np.array([0,10000,100000,300000])#,300000])
U_temp = np.zeros((len(temporal_epochs),n))
V_temp = np.zeros((len(temporal_epochs),n))
U_temp[0][:]=U
V_temp[0][:]=V

'''Optimize alternating GD'''
for epoch in range(epochs):
     
    # Update step of U :
    [f1,grad_f1] = cost3(U,V,Y,Q,D2,g3_U)
    U = U - rate*grad_f1
    
    # Inequality & equality constraints
    U[U<minb] = minb      
    V = phi(U,D2)
    
    # Update step of V:
    [f2,grad_f2] = cost3(U,V,Y,Q,D2,g3_V)
    V = V - rate*grad_f2
    
   # Save function values - for debegging & visualizing
    loss1_epochs[epoch] = f1 
    loss2_epochs[epoch] = f2

    if np.any(temporal_epochs==epoch+1):       
        pos = np.argwhere(temporal_epochs == epoch+1)[0][0]
        U_temp[pos][:]=U
        V_temp[pos][:]=V

    if verbose and ((epoch+1) % 50000 == 0):
        print ('epoch {}, loss f1 {},loss f2 {},norm f1 {}, norm_f2 {}'.format(epoch,f1,f2,np.linalg.norm(grad_f1,2),np.linalg.norm(grad_f2,2)))   
        
# %% 
'''Visualize alternating GD '''

Epochs = np.arange(1, epochs + 1 , 1)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,8))
ax = axes.ravel()
fnt_size = 14
ax[0].plot(Epochs, loss1_epochs)
if np.log(loss1_epochs[0]/loss1_epochs[1])>0:
    ax[0].set_yscale('log')
ax[0].set_title(r'Loss - $f(\boldsymbol{ \sigma, \tilde \nu})$', fontsize=16,fontweight='bold')#,y=0.85)

ax[1].plot(Epochs, loss2_epochs)
if np.log(loss2_epochs[0]/loss2_epochs[-1])>0:
    ax[1].set_yscale('log')
ax[1].set_title(r'Loss - $f(\boldsymbol{ \tilde \sigma, \nu})$', fontsize=16,fontweight='bold')#,y=0.85)

for ax_i in ax[0:2]:
    ax_i.set_ylabel('Loss', fontsize=fnt_size,fontweight='bold')
    ax_i.set_xlabel(r'Epochs $\#$', fontsize=fnt_size,fontweight='bold')
    ax_i.set_xlabel(r'Epochs $\#$', fontsize=fnt_size,fontweight='bold')
    ax_i.set_xlim((0,epochs))  
    
for i,epoch in enumerate(temporal_epochs):    
    ax[2].plot(heights,U_temp[i][:], label = 'epoch {}'.format(epoch) )

ax[2].plot(heights,alpha, 'b--', label = r'Original $\boldsymbol\sigma$')
ax[2].set_ylabel(r'$\boldsymbol\sigma$', fontsize=fnt_size,fontweight='bold')
ax[2].set_xlabel('Heights [Km]', fontsize=12,fontweight='bold')
ax[2].set_title(r'Retrival over time of $\boldsymbol\sigma$', fontsize=16,fontweight='bold')
ax[2].set_ylim((0, U_temp.max()*1.05))

ax[2].text(1,0.02, r'Klett''s solution \n for $r_m={:.2f}[$Km$]$'.format(heights[ind_m]),fontsize=16)
ax[2].texts[0].set_position((1,0.001))
arrow1 = mpatches.FancyArrowPatch((2,0.02), (2, 0.04),mutation_scale=15)
ax[2].add_patch(arrow1)

for i,epoch in enumerate(temporal_epochs):    
    ax[3].plot(heights,V_temp[i][:], label = 'epoch {}'.format(epoch) )

V_orig = phi(alpha,D2)
ax[3].plot(heights,V_orig, 'b--', label = r'Original $\boldsymbol\nu$ ')
ax[3].set_ylabel(r'$\boldsymbol\nu$', fontsize=fnt_size,fontweight='bold')
ax[3].set_xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
ax[3].set_title(r'Retrival over time of $\boldsymbol{\nu=\exp^{-2D\sigma}}$', fontsize=16,fontweight='bold')
ax[3].set_ylim((V_temp.min()-np.abs(0.05*V_temp.min()),V_temp.max()*1.05))

ax[3].text(4,0.85, r'Klett''s solution \n for $r_m={:.2f}[$Km$]$'.format(heights[ind_m]),fontsize=16)
ax[3].texts[0].set_position((4.25,0.75))
arrow2 = mpatches.FancyArrowPatch((6,0.75), (6, 0.68),mutation_scale=15)
ax[3].add_patch(arrow2)

for ax_i in ax[2:4]:
    ax_i.set_xlim((0,heights.max()))
    ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax_i.legend()

fig.canvas.draw()
plt.tight_layout()
fig.show()
# %% Iterative LS - for U
add_noise = False
WLS = False
add_Reg = False
lambda_reg = 0.125*0.125#2/2000.0


''' set constants, input measurments, and initial solution'''
ConstP = lidar_const*B_0
ind_m = np.int(10.0*n/100.0) #+128
th_df = 10*eps

if add_noise:
    # add noise to p
    stdPowerNoise = np.sqrt(P)
    PNoise =np.random.normal(loc=P, scale=stdPowerNoise, size=n)
    PNoise= PNoise.round()
    PNoise[PNoise<eps] = eps
    #std_read_noise = 0
    #poissonNoise = np.random.poisson(P).astype(float)
    #PNoise =P +  stdPowerNoise*np.random.randn(n)
    
    # set normalize mesurments - y
    Y = PNoise/ConstP
    
    # set initial solution sigma_0
    SNoise = mscLid.calc_S(heights,PNoise)
    sigma_0 = mscLid.calc_extiction_klett(SNoise, heights, alpha, ind_m)
    #sigma_0 = 0.1*np.ones_like(Y)
    if WLS: 
        var_noise = P/(ConstP**2)
        #covmat =  np.diag(var_noise)
        #Sigma_inv =  np.linalg.inv(covmat)
        #Sigma_inv = np.diag((ConstP**2)/P)
        W = np.diag(1/var_noise)
        Ws = np.sqrt(W)
        
else:
    # set normalize mesurments - y
    Y = P/ConstP
    # set initial solution sigma_0
    S = mscLid.calc_S(heights,P)
    sigma_0 = mscLid.calc_extiction_klett(S, heights, alpha, ind_m) #U_sol_interp
    #sigma_0= U_sol1
    #sigma_0 = 0.1*np.ones_like(Y)
    sigma_0 = U_sol_interp

# %%
    
'''original solution'''
v_orig = phi(alpha,D2)
Q_v_orig = np.diag(v_orig)*Q
g_model_orig = Y - np.matmul(Q_v_orig,alpha)    
    
    
# set initial solution U_0, V_0
U_0 =  np.array(np.copy(sigma_0)).reshape(n,1)
V_0 = phi(U_0,D2)
#U_0 = 10*eps+ 0.5 + 0.05*np.sin(6*heights) #0.1*np.ones_like(U_0) # 0.1*np.ones_like(U_0) #np.abs(np.sin(heights))#0.1*np.ones_like(U_0)
Q_v =  np.matmul(np.diag(V_0[:,0]),Q)
if add_Reg:
    Ld = mscLid.laplacian_operator(n,1,1)
#    Ld=np.zeros((n,n))
#    i,j = np.indices(Ld.shape)
#    Ld[i==j]=K0
#    Ld[i==j-1]=K1
#    Ld[i==j+1]=K1
#    Ld[i==j+2]=K2
#    Ld[i==j-2]=K2
#    Ld+= laplacian_operator(n,1,1)
#    Ld[i==j+3]=K3
#    Ld[i==j-3]=K3
#    Ld[i==j+4]=K4
#    Ld[i==j-4]=K4
#    Ld[i==j+5]=K5
#    Ld[i==j-5]=K5
    L = sprs.csr_matrix(Ld)
    Q_reg =  np.sqrt(lambda_reg)*L.todense()
    Q_v = np.concatenate((Q_v,Q_reg),axis=0)
    Y=np.concatenate([Y,np.zeros_like(Y)]).reshape((2*n,1)) 
    
# iteration - 0
if add_Reg:
    g_model = Y[0:n,0].reshape((n,1)) - np.matmul(Q_v,U_0)[0:n,0].reshape((n,1))  
else:
     g_model = Y - np.matmul(Q_v,U_0)[:,0]
g_model = np.array(g_model)


if WLS: 
    #Q_v = np.matmul(np.sqrt(Sigma_inv),Q_v)
    #Y= np.matmul(np.sqrt(Sigma_inv),Y)
    g_model = np.matmul(Ws,g_model) 


f1_ls = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]
#if add_Reg:
#    g_reg = np.matmul(np.matmul(U_0.transpose(),L.todense()),U_0)[0,0]
#    f1_ls += lambda_reg*g_reg

'''start iterate'''
it = 0
df = f1_ls - 0
print np.abs(g_model).max()
print ('epoch {}, loss Ls1 {}'.format(it,f1_ls))   



'''open empty results figure'''
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.ravel()

ax[0].plot(heights,U_0, label = r'$\boldsymbol\sigma$ - {}'.format(it))
ax[1].plot(heights,V_0, label =  r'$\boldsymbol\nu$ - {}'.format(it))
#ax[2].plot(heights,g_model[0:n],label=r'$g(\sigma,\nu)$ - {}'.format(it))
(Umin,Umax) = (U_0.min(),U_0.max())
(Vmin,Vmax)= (V_0.min(),V_0.max())
max_iters = 1200

while (np.abs(df)> th_df and it<=max_iters):
    # LS U
    if WLS: 
        Qv_t_Qv = np.matmul(np.matmul(Q_v.transpose(),W),Q_v)
        Qv_t = np.matmul(Q_v.transpose(),W)
        #Q_v_inv = np.matmul(np.matmul(np.linalg.inv(Qv_t_Qv),Q_v.transpose()),Sigma_inv)
    else:
        Qv_t_Qv =  np.matmul(Q_v.transpose(),Q_v)
        Qv_t = Q_v.transpose()
        #Q_v_inv = np.matmul(np.linalg.inv(Qv_t_Qv),Q_v.transpose())
    #if add_Reg:
    #    Qv_t_Qv += lambda_reg*(0.75*L.todense() + 0.25*L2.todense())   
        
    Q_v_inv = np.matmul(np.linalg.inv(Qv_t_Qv),Qv_t) 

    U_sol1 =  np.matmul(Q_v_inv,Y).reshape((n,1))
    U_sol1[U_sol1<minb]  = minb
    V_sol1 = phi(U_sol1,D2)
    
    Q_v = np.diag(V_sol1[:,0])*Q
    if add_Reg:
        Q_v = np.concatenate((Q_v,Q_reg),axis=0)
    
    #g_model = Y - np.matmul(Q_v,U_sol1)[:,0] 
    if add_Reg:
        g_model = Y[0:n,0].reshape((n,1))  - np.matmul(Q_v,U_sol1)[0:n,0].reshape((n,1))
    else:
        g_model = Y - np.matmul(Q_v,U_sol1)[:,0]
    g_model = np.array(g_model)
    if WLS: 
        g_model = np.matmul(Ws,g_model)

    f_ls_new = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]

    df = f1_ls- f_ls_new 
    print np.abs(g_model).max()
    f1_ls = f_ls_new
    
    # print and plot results
    it+=1
    #print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))   
    if (it==2):#(it % 200 == 0):
    
        ax[0].plot(heights,U_sol1, label =  r'$\boldsymbol\sigma$ - {}'.format(it))
        ax[1].plot(heights,V_sol1,label =    r'$\boldsymbol\nu$ - {}'.format(it))
        #ax[2].plot(heights,g_model[0:n],'r',label=r'$g(\sigma,\nu)$ - {}'.format(it))
        (Umin,Umax) = (min(U_sol1.min(),Umin),max(U_sol1.max(),Umax))
        (Vmin,Vmax) = (min(V_sol1.min(),Vmin),max(V_sol1.max(),Vmax))
        print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))
            
if it<max_iters:
    ax[0].plot(heights,U_sol1,'limegreen', label =  r'$\boldsymbol\sigma$ - {}'.format(it))
    ax[1].plot(heights,V_sol1,'limegreen', label =    r'$\boldsymbol\nu$ - {}'.format(it))
    #ax[2].plot(heights,g_model,label=r'$g(\sigma,\nu)$ - {}'.format(it))
(Umin,Umax) = (min(alpha.min(),Umin),max(alpha.max(),Umax))
(Vmin,Vmax) = (min(phi(alpha,D2).min(),Vmin),max(phi(alpha,D2).max(),Vmax))
(Umin,Umax) = (min(U_sol1.min(),Umin),max(U_sol1.max(),Umax))
(Vmin,Vmax) = (min(V_sol1.min(),Vmin),max(V_sol1.max(),Vmax))
# %%
fnt_size = 12
ax[0].set_ylabel(r'$\boldsymbol\sigma$', fontsize=fnt_size,fontweight='bold')
ax[0].set_xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
ax[0].set_title(r'$\boldsymbol{\sigma_{\rm LS}}$', fontsize=16,fontweight='bold')
ax[1].set_ylabel(r'$\boldsymbol\nu$', fontsize=fnt_size,fontweight='bold')
ax[1].set_xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
ax[1].set_title(r'$\boldsymbol{\nu_{LS}=\phi\big(-2D\sigma_{\rm LS}\big)}$', fontsize=16,fontweight='bold') 
ax[0].plot(heights,alpha, 'b--', label = r'Original $\boldsymbol\sigma$')
ax[1].plot(heights,v_orig, 'b--', label = r'Original $\boldsymbol\nu$')
#ax[2].plot(heights,g_model_orig, 'b--', label = r'Original $\boldsymbol g(\sigma,\nu)$')
for ax_i in ax:
    ax_i.set_xlim((0,heights.max()))
    ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax_i.legend()
ax[0].set_ylim((Umin,Umax*1.01))
ax[1].set_ylim((Vmin,Vmax*1.01))
plt.tight_layout()
fig.canvas.draw()
# %% find r_m
sigmas_heights = np.zeros((len(heights),n))
diff_sigma = np.zeros(n)
for ind_m,h in enumerate(heights):
    sigma_m  = mscLid.calc_extiction_klett(S, heights, alpha, ind_m)
    sigmas_heights[ind_m,:] = sigma_m
    diff = np.abs(sigma_m-U_sol1).mean()
    diff_sigma[ind_m] = diff # 0.5*np.matmul(g_model.transpose(),g_model)
plt.figure()
plt.plot(heights,diff_sigma)
plt.yscale('log')
lowest3 = np.argsort(diff_sigma)[:3]
prev_guess = [10,49,50]
for ind in prev_guess: 
    #plt.axvline(x=heights[ind],color='0.5',linestyle='--',alpha=0.5)
    plt.plot(heights[ind],diff_sigma[ind],'or')
plt.plot(heights[lowest3[0]],diff_sigma[lowest3[0]],'o',color='limegreen')
plt.xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
plt.grid(which='both',axis='x',linestyle='--')
plt.grid(which='major',axis='y',linestyle='--')
plt.xticks(np.arange(0, 11, step=1))
plt.xlim((0,10))
plt.ylabel(r'$\boldsymbol{\frac{1}{n}|\sigma_{\rm LS}-\sigma_{\rm Klett}|}$', fontsize=fnt_size,fontweight='bold')
plt.show()
# %% LS - Additive white Gaussian noise
# Adding noise using target SNR

# Set a target SNR
target_snr_db = 70
# Calculate signal power and convert to dB 
sig_avg_watts = np.mean(P)
sig_avg_db = 10 * np.log10(sig_avg_watts)
# Calculate noise according to [2] then convert to watts
noise_avg_db = sig_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
# Generate an sample of white noise
mean_noise = 0
std_noise = np.sqrt(noise_avg_watts)
noise_volts = np.random.normal(mean_noise,std_noise , len(P))
# Noise up the original signal
P_awgn = P + noise_volts
P_awgn[P_awgn<eps] = eps
S_awgn = mscLid.calc_S (heights,P_awgn)
# %%

lData  = {'x':heights,'y':P,'lableX': 'Heights [km]','lableY': 'Lidar Power'}
rData  = {'x':heights,'y':S,'lableX': 'Heights [km]','lableY': 'Logaritmic range adjusted power'}
[fig, axes] = mscLid.visCurve(lData,rData)
fig.suptitle('Power and the logaritmic range adjusted power based on readings from 22/4/17')
ax = axes.ravel()
ax[0].lines[0].set_label('P')
ax[1].lines[0].set_label('S')
ax[0].plot(heights,P_awgn,label='P + AWGN')
ax[1].plot(heights,S_awgn,label='S + AWGN')
ax[0].legend()
ax[1].legend()
P_max = 1.1*np.max(P[1:-1])
P_min = -0.2e-3#-1.1*np.min(P[1:-1])
ax[0].set_ylim((P_min,P_max)) 
ax[0].set_xlim((0,np.max(heights))) 
plt.tight_layout()

fig.canvas.draw()
# %%
'''Visualize Klett'''
lData  = {'x':heights,'y':sigma_r,'lableX': 'Heights [km]','lableY': 'sigma retrival'}
rData  = {'x':heights,'y':beta_r,'lableX': 'Heights [km]','lableY': 'beta retrival'}
[fig, axes] = mscLid.visCurve(lData,rData)
fig.suptitle('Klett''s retrievals')

ax = axes.ravel()
ax[0].set_xlim((0,heights.max()))
ax[1].set_xlim((0,heights.max()))
ax[0].set_ylim((0,sigma_r.max()*1.1))
ax[1].set_ylim((0,beta_r.max()*1.1))
fig.canvas.draw()

sigma_r_awgn = mscLid.calc_extiction_klett(S_awgn, heights, alpha, ind_m)
beta_r_awgn = B_0*np.power(sigma_r_awgn,k)
ax[0].lines[0].set_label(r'$\sigma$')
ax[1].lines[0].set_label(r'$\beta$')
ax[0].plot(heights,sigma_r_awgn,label=r'$\sigma$ + AWGN')
ax[1].plot(heights,beta_r_awgn,label=r'$\beta$ + AWGN')
ax[0].legend()
ax[1].legend()


#%%# Plot in dB
y_watts = y_volts ** 2
y_db = 10 * np.log10(y_watts)
plt.subplot(1,3,3)
plt.plot(heights, 10* np.log10(y_volts**2))
plt.title('Signal with noise (dB)')
plt.ylabel('Power (dB)')
plt.xlabel('Time (s)')
plt.show()

# %%

# %%    coarse-to-fine 
''' save first the original data - coarse '''
n_orig = np.copy(n) # 1334
P_orig=np.copy(P)
heights_orig = heights
alpha_orig = np.copy(alpha)
beta_orig = np.copy(beta)
''' refine the signal sollution, to serve as an initial solution to a finner resolution'''
#heights = np.linspace(min_height, 10, 100) #n_orig/(np.power(2,0)))
#n = len(heights)
#P= np.interp(heights,heights_orig,P_orig)

heights = np.linspace(min_height, 10, 1334)
alpha=np.interp(heights,heights_orig,alpha_orig)
beta=np.interp(heights,heights_orig,beta_orig)

P = mscLid.generate_P(P0,c,A,tau,heights,alpha,beta)

n = len(heights)
U_sol_interp = np.interp(heights,heights_orig,np.squeeze(np.asarray(U_sol1)))  # U_sol_interp - is the coarse solution

# %% Load matlab data from Birgit 
import scipy.io as spio
#mat = spio.loadmat('data_1800_13_Sep_2017.mat')

mat = spio.loadmat('../data_1200_22_Apr_2017.mat')
'''Channels: 0: 355[nm] UV
             2: 387[nm] blue (ramman)
             4: 532[nm] green
             6: 607[nm] red (ramman)
             7: 1064[nm] IR'''
chan = 4 # green
#top_bin = mat['rbins'][0][0]
sind = 0# 0 Km
eind = 1301 # 10Km
heights1 = mat['alt'][0][sind:eind]/1000.0 #converts [m] to [Km]

n = len(heights1)
dr = np.mean(heights1[1:]-heights1[0:-1])
tau = dr*2/c
B_0 = 55
#P = mat['bg_corr'][chan][sind:eind]
#P[P<=0]=eps
sigma_mol = mat['alpha_mol'][chan][sind:eind]
beta_mol = mat['beta_mol'][chan][sind:eind]
#S = mscLid.calc_S(heights1,P)
#sigma_0 = mscLid.calc_extiction_klett(S, heights, sigma_mol, ind_m)
#Y =  np.copy(P)


# solutions:
ind_m = np.int(mat['RefBin'][0][chan]) #ref_bin = mat['RefBin'][0][chan]  # the solution of reference range
sigma_sol = mat['alpha_aerosol'][chan][sind:eind]
sigma_sol_sm = mat['alpha_aerosol_sm'][chan][sind:eind]
beta_sol = mat['beta_aerosol'][chan][sind:eind]
beta_sol_sm = mat['beta_aerosol_sm'][chan][sind:eind]



'''Calculate the power and the Logaritmic range adjusted power'''
alpha = sigma_sol_sm
beta = beta_sol_sm

lidar_const = 25909075299601.637000 #0.5*P0*c*tau*A
ConstP=lidar_const*B_0
tot_alpha = alpha+sigma_mol
tot_beta=beta+beta_sol
P1 = mscLid.generate_P(P0,c,A,tau,heights,alpha,beta)
P1[np.isnan(P1)]=0
#P[P<eps] = eps
#P1= P1.round()
P1[P1<eps] = eps
P=P1
S = mscLid.calc_S(heights,P1)
sigma_0 = mscLid.calc_extiction_klett(S, heights, tot_alpha, ind_m)

Y = P1/ConstP 

vis_TROPOS_RES = True
if vis_TROPOS_RES:

    fnt_size = 12
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax = axes.ravel()
    ax = axes.ravel()
    # plot profiles
    ax[0].plot(heights,sigma_sol,label = r'$\sigma$')
    ax[1].plot(heights,beta_sol,label = r'$\beta$')
    # plot smooth profiles
    ax[0].plot(heights,sigma_sol_sm,label = r'$\sigma_{sm}$')
    ax[1].plot(heights,beta_sol_sm,label = r'$\beta_{sm}$')
    
    ax[0].plot(heights,sigma_0,label = r'$\sigma_{sm}$')

    ax[0].set_title(r'$\boldsymbol\sigma_{\rm TROPOS} \quad \rm for\quad \lambda = 532[nm]$',fontsize=16,fontweight='bold')
    ax[1].set_title(r'$\boldsymbol\beta_{\rm TROPOS} \quad \rm for\quad \lambda = 532[nm]$',fontsize=16,fontweight='bold')
    ax[0].set_ylabel(r'$\boldsymbol\sigma\quad[\rm{{1}/{km\cdot sr}]}$',fontsize=fnt_size,fontweight='bold')
    ax[1].set_ylabel(r'$\boldsymbol\beta\quad[\rm{{1}/{km\cdot sr}]}$',fontsize=fnt_size,fontweight='bold')

    for ax_i in ax:
        ax_i.set_xlim((0,heights[-1]))
        ax_i.set_xlabel('heights[km]',fontsize=fnt_size,fontweight='bold')
        ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
        ax_i.legend()
    ax[0].set_ylim((0,sigma_sol[sind+2:eind].max()*1.1))
    ax[1].set_ylim((0,beta_sol[sind+2:eind].max()*1.1))
    plt.tight_layout()
    fig.canvas.draw()

    lData  = {'x':heights,'y':P,'lableX': 'Heights [km]','lableY': r'$\boldsymbol{\rm P}\quad[\rm A.U.]$'}
    rData  = {'x':heights,'y':S,'lableX': 'Heights [km]','lableY': r'$\boldsymbol{\rm S}\quad[\rm A.U.]$'}
    [fig, axes] = mscLid.visCurve(lData,rData)
    ax = axes.ravel()
    P_max = 1.1*np.max(P[1:-1])
    P_min = -0.2e-3#-1.1*np.min(P[1:-1])
    ax[0].set_ylim((P_min,P_max)) 
    ax[0].set_xlim((0,np.max(heights))) 
    ax[1].set_ylim((S.min(),1.1*S.max()))
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    fig.canvas.draw()

# %%
    
    
'''original solution'''
v_orig = phi(alpha,D2)
Q_v_orig = np.diag(v_orig)*Q
g_model_orig = Y - np.matmul(Q_v_orig,alpha)    
    
    
# set initial solution U_0, V_0
U_0 =  np.array(np.copy(sigma_0)).reshape(n,1)
V_0 = phi(U_0,D2)
Q_v =  np.matmul(np.diag(V_0[:,0]),Q)
if add_Reg:
    Ld = mscLid.laplacian_operator(n,1,1)
    L = sprs.csr_matrix(Ld)
    #Q_reg =  np.sqrt(lambda_reg)*L.todense()
    #Q_v = np.concatenate((Q_v,Q_reg),axis=0)
    #Y=np.concatenate([Y,np.zeros_like(Y)]).reshape((2*n,1)) 
    
# iteration - 0
g_model = Y - np.matmul(Q_v,U_0)[:,0]
g_model = np.array(g_model)


if WLS: 
    #Q_v = np.matmul(np.sqrt(Sigma_inv),Q_v)
    #Y= np.matmul(np.sqrt(Sigma_inv),Y)
    g_model = np.matmul(Ws,g_model) 


f1_ls = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]


'''start iterate'''
it = 0
df = f1_ls - 0
print 1-df/f1_ls
t =  1-df/f1_ls
print ('epoch {}, loss Ls1 {}'.format(it,f1_ls))   



'''open empty results figure'''
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))
ax = axes.ravel()
ax[0].plot(heights,alpha, 'b--', label = r'Original $\boldsymbol\sigma$')
ax[1].plot(heights,v_orig, 'b--', label = r'Original $\boldsymbol\nu$')
ax[2].plot(heights,g_model_orig, 'b--', label = r'$Original g(\sigma,\nu)$')

ax[0].plot(heights,U_0, label = r'$\boldsymbol\sigma$ - {}'.format(it))
ax[1].plot(heights,V_0, label =  r'$\boldsymbol\nu$ - {}'.format(it))
ax[2].plot(heights,g_model[0:n],label=r'$g(\sigma,\nu)$ - {}'.format(it))
(Umin,Umax) = (U_0.min(),U_0.max())
(Vmin,Vmax)= (V_0.min(),V_0.max())
max_iters = 1200

while (np.abs(df)> th_df and it<=max_iters):
    # LS U
    if WLS: 
        Qv_t_Qv = np.matmul(np.matmul(Q_v.transpose(),W),Q_v)
        Qv_t = np.matmul(Q_v.transpose(),W)
        #Q_v_inv = np.matmul(np.matmul(np.linalg.inv(Qv_t_Qv),Q_v.transpose()),Sigma_inv)
    else:
        Qv_t_Qv =  np.matmul(Q_v.transpose(),Q_v)
        Qv_t = Q_v.transpose()
        #Q_v_inv = np.matmul(np.linalg.inv(Qv_t_Qv),Q_v.transpose())
    #if add_Reg:
    #    Qv_t_Qv += lambda_reg*(0.75*L.todense() + 0.25*L2.todense())   
        
    Q_v_inv = np.matmul(np.linalg.inv(Qv_t_Qv),Qv_t) 

    U_sol1 =  np.matmul(Q_v_inv,Y).reshape((n,1))
    U_sol1[U_sol1<minb]  = minb
    V_sol1 = phi(U_sol1,D2)
    
    Q_v = np.diag(V_sol1[:,0])*Q
    if add_Reg:
        Q_reg =  np.sqrt(t)*L.todense()
        Q_v = np.concatenate((Q_v,Q_reg),axis=0)
    
    #g_model = Y - np.matmul(Q_v,U_sol1)[:,0] 
    if add_Reg:
        g_model = Y[0:n,0].reshape((n,1))  - np.matmul(Q_v,U_sol1)[0:n,0].reshape((n,1))
    else:
        g_model = Y - np.matmul(Q_v,U_sol1)[:,0]
    g_model = np.array(g_model)
    if WLS: 
        g_model = np.matmul(Ws,g_model)
    #Q_v =  Q_v_ if wo_reg else np.concatenate((Q_v_,L.todense()),axis=0)
    
    #.reshape(len(Y),1)
    #if WLS: 
    #    g_model = np.matmul(np.sqrt(Sigma_inv),g_model)    
    f_ls_new = 0.5*np.matmul(g_model.transpose(),g_model).reshape((1,1))[0,0]
    #if add_Reg:
        #W = np.diag(np.sqrt(np.abs(1-g_model/np.max(g_model))))
        #WW = np.matmul(np.matmul(W,L.todense()),W)
        #g_reg = np.matmul(np.matmul(U_sol1.transpose(),WW),U_sol1)[0,0]
        #g_reg1 = np.matmul(np.matmul(U_sol1.transpose(),L.todense()),U_sol1)[0,0]
        #g_reg2 = np.matmul(np.matmul(U_sol1.transpose(),L2.todense()),U_sol1)[0,0]
        #f_ls_new += lambda_reg*(0.75*g_reg1+0.25*g_reg2)
    #f_ls_new = 0.5*np.matmul(g_model.transpose(), np.matmul(Sigma_inv,g_model)) if WLS else 0.5*np.matmul(g_model.transpose(),g_model) 
    #g_model = np.matmul(np.sqrt(Sigma_inv),(Y - np.matmul(Q_v,U_sol1)))
    #f_ls_new = 0.5*(g_model**2).mean()
    df = f1_ls- f_ls_new 
    print 1-df/f1_ls
    t = 1-df/f1_ls
    f1_ls = f_ls_new
    
    # print and plot results
    it+=1
    #print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))   
    if (it % 200 == 0):
    
        ax[0].plot(heights,U_sol1, label =  r'$\boldsymbol\sigma$ - {}'.format(it))
        ax[1].plot(heights,V_sol1, label =    r'$\boldsymbol\nu$ - {}'.format(it))
        ax[2].plot(heights,g_model[0:n],label=r'$g(\sigma,\nu)$ - {}'.format(it))
        (Umin,Umax) = (min(U_sol1.min(),Umin),max(U_sol1.max(),Umax))
        (Vmin,Vmax) = (min(V_sol1.min(),Vmin),max(V_sol1.max(),Vmax))
        print ('epoch {}, loss LS1 {}, df {}'.format(it,f1_ls,df))
        
if it<max_iters:
    ax[0].plot(heights,U_sol1, label =  r'$\boldsymbol\sigma$ - {}'.format(it))
    ax[1].plot(heights,V_sol1, label =    r'$\boldsymbol\nu$ - {}'.format(it))
    ax[2].plot(heights,g_model,label=r'$g(\sigma,\nu)$ - {}'.format(it))
(Umin,Umax) = (min(alpha.min(),Umin),max(alpha.max(),Umax))
(Vmin,Vmax) = (min(phi(alpha,D2).min(),Vmin),max(phi(alpha,D2).max(),Vmax))
(Umin,Umax) = (min(U_sol1.min(),Umin),max(U_sol1.max(),Umax))
(Vmin,Vmax) = (min(V_sol1.min(),Vmin),max(V_sol1.max(),Vmax))
fnt_size = 12
ax[0].set_ylabel(r'$\boldsymbol\sigma$', fontsize=fnt_size,fontweight='bold')
ax[0].set_xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
ax[0].set_title(r'$\boldsymbol{\sigma_{\rm LS}}$', fontsize=16,fontweight='bold')
ax[1].set_ylabel(r'$\boldsymbol\nu$', fontsize=fnt_size,fontweight='bold')
ax[1].set_xlabel('Heights [Km]', fontsize=fnt_size,fontweight='bold')
ax[1].set_title(r'$\boldsymbol{\nu_{LS}=\phi\big(-2D\sigma_{\rm LS}\big)}$', fontsize=16,fontweight='bold') 

for ax_i in ax:
    ax_i.set_xlim((0,heights.max()))
    ax_i.ticklabel_format(axis='y', style='sci', scilimits=(-4,-4))
    ax_i.legend()
ax[0].set_ylim((Umin,Umax*1.01))
ax[1].set_ylim((Vmin,Vmax*1.01))
plt.tight_layout()
fig.canvas.draw()