from datetime import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from generate_atmosphere import LidarProfile
#%matplotlib inline
import mpld3
#mpld3.enable_notebook()
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('font', family='serif')


eps = np.finfo(np.float).eps
print eps
find_nearest = lambda array,value: (np.abs(array - value)).argmin()


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def generate_P (P0,c,A,heights,alpha,beta):
    dr = heights[1:]-heights[0:-1] # dr for integration
    last = dr[-1]
    dr = np.append(dr,last)
    tau = np.mean(dr)*2/c  #[sec]  Is this resonable ? How to set the real value 
    lidar_const = 0.5*P0*c*tau*A
    numerator =  lidar_const*beta*np.exp(-2*np.cumsum(alpha*dr))
    denominator = np.power(heights,2)+eps # epsilon is to avoid NaN
    z_pos =  np.where(numerator == 0)[0]
    exp_alpha = np.exp(-2*np.cumsum(alpha*dr))
    P = numerator/denominator
    return P


def calc_S (heights,P):
    '''Calculate the Logaritmic range adjusted power S(r) = ln(r^2*P(r))'''
    lanpow = lambda r,p: np.log(np.power(r,2)*p)#+eps) #if (h!=0 and p!=0) else 0
    S = np.array([lanpow(r,p) for (r,p) in zip(heights,P)])
    return S

def calc_extiction_klett(S, heights, alpha, ind_m):
    '''Calculate the inversion of the extinction coefficient'''
    k=1 
    S_m = S[ind_m]
    sigma_m = alpha[ind_m]
    
    dr =  heights[1]-heights[0]
    exp_S = np.exp((S-S_m)/k)
    denominator = (1/(sigma_m) + (2/k)*np.flip(np.cumsum(np.flip(exp_S,0)*dr),0))
    sigma = exp_S/(denominator)#+eps)
    return sigma



## Rread Lidar data from Amit repository
lidar_paths = sorted(glob.glob("../Lidar/2017_04_22/*41smooth.txt"))[:4]
profiles_alpha = []
profiles_beta = []

for path in lidar_paths[0:1]:
    lidar_profile = LidarProfile(path)
    min_height = lidar_profile._lidar_height
    heights = np.linspace(min_height, 10,num=100)
    cur_profiles_0= lidar_profile.interpolate(heights)[0]
    cur_profiles_1= lidar_profile.interpolate(heights)[1]

    profiles_alpha.append(cur_profiles_0)
    profiles_beta.append(cur_profiles_1)
    
lidar_df = pd.read_csv(
    path,
    delimiter="\t",
    index_col=0,
    skiprows=None
)
lidar_df[lidar_df<0] = 0
times = []
for path in lidar_paths:
    s = path.split("/")[-1].split('-')[1:3]
    times.append("{}:{}".format(s[0].split('_')[1], s[1]))
    
## Show beta and sigma
beta = profiles_beta[0]
alpha = profiles_alpha[0] 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax = axes.ravel()

ax[0].plot(heights,alpha)
ax[0].set_ylabel('sigma')#, fontsize=18,fontweight='bold')# [km^-1sr^1]$")
ax[0].set_xlabel('Heights [km]')#,fontsize=18,fontweight='bold')
#ax[0].set_title('Extinction coefficient from Lidar measurments',fontsize=20,fontweight='bold')
ax[1].plot(heights,beta)
ax[1].set_ylabel('beta')#,fontsize=18,fontweight='bold')# [km^-1sr^1]$")
ax[1].set_xlabel('Heights [km]')#,fontsize=18,fontweight='bold')
#ax[1].set_title('Backscatter coefficient from Lidar measurments',fontsize=20,fontweight='bold')

plt.show()

fig.savefig('orig_coeffs.svg',bbox_inches='tight')