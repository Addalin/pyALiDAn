"""
Created on Tue Jul 23 12:51:57 2019

@author: addalin

Lidar reading and visualizing for Aviad
"""

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
import matplotlib.patches as mpatches
eps = mscLid.eps 
print eps
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
from matplotlib.ticker import FormatStrFormatter

def laplacian_operator(nx,ny,nz):
    if nx>1 and ny==1 and nz==1:
        K=1
    elif nx==1 or ny==1 or nz==1:     
        K=2
    else:
        K=3

    ex = np.ones((nx,1))  
    data_x = np.concatenate((ex,-K*ex,ex),axis=1).transpose()
    diags_x = np.array([-1, 0, 1])
    Lx = sprs.spdiags(data_x,diags_x,nx,nx)
    
    ey = np.ones((ny,1))
    data_y = np.concatenate((ey,-K*ey,ey),axis=1).transpose()
    diags_y = np.array([-1, 0, 1])
    Ly = sprs.spdiags(data_y,diags_y,ny,ny)  

    Ix = sprs.eye(nx)  
    Iy = sprs.eye(ny) 
    L2 = sprs.kron(Iy,Lx) + sprs.kron(Ly,Ix) 
 
    N = nx*ny*nz 
    e = np.ones((N,1))
    data_z = np.concatenate((e,e),axis=1).transpose()
    diags_z = np.array([-nx*ny, nx*ny])
    L = sprs.spdiags(data_z,diags_z,N,N)  
    Iz = sprs.eye(nz)  
 
    A = sprs.kron(Iz,L2)+L
    return A

# %%
''' Load readings of Lidar from 22/4/17'''

lidar_paths = sorted(glob.glob("../Lidar/2017_04_22/*41smooth.txt"))[:4] 
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