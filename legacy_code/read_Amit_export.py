# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:05:04 2019

@author: addalin
"""
import os
import sys
import glob
import numpy as np
import cv2
print cv2.__version__
print cv2.__file__

import scipy as sp
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


# %%
# Lowding Amit's data 

#folder = r'H:\amitibo\exports\2017_09_04\2017_09_04_06_14_07'
folder = r'H:\amitibo\ec2_data\SHDOM\exports\2017_05_19\2017_05_19_10_35_00'
wDir = folder
sys.path.append(wDir)
os.chdir(wDir)

file_name = 'export_data.pkl'
file_path = folder + '\\' + file_name
with open(file_path, 'rb') as f:
    data = pickle.load(f)
    
c_names = data.keys()

cam = 0 
dat_cam = data[c_names[cam]]
mask_im = dat_cam['MASK']
chanels = ['R','G','B','PSI','PHI']
fnt_size = 12
nchan = len(chanels)
fig, axes = plt.subplots(nrows=2, ncols=(nchan+1)/2, figsize=(10,5))
ax = axes.ravel()
frame = fig.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
mask_im = dat_cam['MASK']
[w,h]=np.shape(mask_im)
for (i,chan) in enumerate(chanels):
    RGB_im = np.zeros((w,h,3))
    im = dat_cam[chan]
    masked_im = im # np.multiply(im,mask_im)
    if chan in chanels[0:3]:
        print chan
        masked_im= masked_im.astype('uint8')
        RGB_im[0:w,0:h,i] =masked_im
    ax[i].imshow(masked_im)
    ax[i].set_title(chan)
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
#RGB_im =Image.fromarray(RGB_im)
ax[5].imshow(RGB_im)
ax[5].set_title('RGB')
fig.tight_layout()
fig.show()

# Todo = show the RGB image : https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html
# %%
file_name = 'space_carve.pkl'
file_path = folder + '\\' + file_name
with open(file_path, 'rb') as f:
    data_space_carve = pickle.load(f)
    
    