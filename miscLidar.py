#!/usr/bin/env python
# MIT License
# Copyright (c) 2020  Adi Vainiger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Miscellaneous operations for lidar analysis.
"""

import os, sys
import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.sparse as sprs
from datetime import datetime, timedelta, time

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
eps = np.finfo(np.float).eps


# %%
def smooth(x, window_len=11, window='hanning'):
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
		raise ValueError("smooth only accepts 1 dimension arrays.")

	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")

	if window_len < 3:
		return x

	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

	s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
	# print(len(s))
	if window == 'flat':  # moving average
		w = np.ones(window_len, 'd')
	else:
		w = eval('np.' + window + '(window_len)')

	y = np.convolve(w / w.sum(), s, mode='valid')
	return y


# %%
class RadiosondeProfile(object):
	"""Temperature Profile Handling

	This object facilitates handling temperature and pressure profiles.

	TODO

	Args:
		radiosonde_path (str): path to radiosonde file.
	"""

	def __init__(self, radiosonde_path):
		import pandas as pd

		df = pd.read_csv(radiosonde_path, delimiter="\s+")
		self._heights = df['HGHT'].values  # Measurement heights [m]
		self._temps = df['TEMP'].values  # Atmospheric temperature [celcuis deg]
		self._pressures = df['PRES'].values  # Atmospheric pressure [hPa]
		self._RelativeHumidity = df['RELH'].values  # Atmospheric relative humidity [%]

	@property
	def temp_celsius(self):
		return self._temps

	@property
	def temp_kelvin(self):
		return self._temps + 273.15

	@property
	def height_meters(self):
		return self._heights

	@property
	def height_kilometers(self):
		return self._heights / 1000

	def interpolateKmKelvin(self, heights):
		"""Interpolate temperature values for a given height profile.

		Args:
			heights (array): 1D grid of heights in kilometers.
		"""

		temps = np.interp(
			heights,
			self.height_kilometers,
			self.temp_kelvin
		)

		return temps

	def interpolateKMPres(self, heights):
		"""Interpolate pressure values for a given height profile.

		Args:
			heights (array): 1D grid of heights in kilometers.
		"""

		pressures = np.interp(
			heights,
			self.height_kilometers,
			self._pressures
		)

		return pressures

	def interpolateKMRLH(self, heights):
		"""Interpolate relative humidity values for a given height profile.

		Args:
			heights (array): 1D grid of heights in kilometers.
		"""

		relhs = np.interp(
			heights,
			self.height_kilometers,
			self._RelativeHumidity
		)

		return relhs


# %%
class LidarProfile(object):
	"""Temperature Profile Handling

	This object facilitates handling Lidar profiles.

	Args:
		lidar_path (str): path to Lidar file.
	"""

	def __init__(self, lidar_path, alpha_column="alpha532 (Mm^-1 sr^-1):", beta_column="Beta532 (Mm^-1 sr^-1):",
	             beta_mol_column='BetaMOL 532 (m^-1 sr^-1)', lidar_height=0.229, skiprows=None):
		import pandas as pd

		lidar_df = pd.read_csv(lidar_path, delimiter="\t", sep=" ", encoding='unicode_escape',
		                       index_col=0, skiprows=None)
		lidar_df[lidar_df < 0] = 0

		self._lidar_height = lidar_height
		self._heights = lidar_df.index.values + lidar_height
		self._extinction = lidar_df[alpha_column].values / 1000  # convert from Mm^-1 to Km^-1
		self._backscatter = lidar_df[beta_column].values / 1000  # convert from Mm^-1 to Km^-1
		self._backscatter_mol = lidar_df[
			beta_mol_column].values  # TODO: check with Julin that the values are already 1/(km sr)

	def interpolate(self, heights):
		"""Interpolate lidar values for a given height profile.

		Args:
			heights (array): 1D grid of heights in kilometers.
		"""

		extinction = np.interp(
			heights,
			self._heights,
			self._extinction
		)
		backscatter = np.interp(
			heights,
			self._heights,
			self._backscatter
		)
		backscatter_mol = np.interp(
			heights,
			self._heights,
			self._backscatter_mol
		)

		return extinction, backscatter, backscatter_mol

	def __getitem__(self, key):
		print("Inside `__getitem__` method!")
		return self.__getitem__[key]


# %%
def calc_tau(sigma,heights):
	"""
	Calculate the the attenuated optical depth (tau is the optical depth), through a collumn of heights.
	:param sigma: ndarray (1-d or 2-d) of attenuation coefficents related to heights vector
	:param heigts: vector of heights.
	Important note: make sure the heights are biased to GROUND (above the lidar), othewise the integration is wrong!
	:return: vector of attenuated optoical depth : tau = integral( sigma(r) * dr)
	"""
	if sigma.ndim == 1:
		dr = heights[1:] - heights[0:-1]  # dr for integration
		dr = np.insert(dr.flatten(), 0, heights[0])
	else:
		dr = heights[1:,0] - heights[0:-1,0]  # dr for integration
		dr = np.insert(dr.flatten(), 0, heights[0,0])
		dr = np.tile(dr.reshape(dr.shape[0], 1), sigma.shape[1])

	tau = np.cumsum(sigma * dr, axis=0)

	return tau


def generate_P(P0, c, A, dt, heights, sigma, beta, lidar_const=1, add_photon_noise=True):
	""" Regenerate the lidar power readings according to atmospheric profile of backscatter (beta) and extinction
	coefficient(sigma) """
	if sigma.ndim > 1:
		heights = np.tile(heights.reshape(heights.shape[0], 1), sigma.shape[1])
	tau = calc_tau(sigma,heights)
	#dr = heights[1:] - heights[0:-1]  # dr for integration
	#dr = np.insert(dr.flatten(), 0, heights[0])

	#if sigma.ndim > 1:
	#	dr = np.tile(dr.reshape(dr.shape[0], 1), sigma.shape[1])
	#	heights = np.tile(heights.reshape(heights.shape[0], 1), sigma.shape[1])
	if lidar_const is None:
		lidar_const = 0.5 * P0 * c * dt * A

	#tau = np.cumsum(sigma * dr, axis=0)
	numerator = beta * np.exp(-2 * tau)  # add axis
	denominator = np.power(heights, 2) + eps  # epsilon is to avoid NaN
	P = lidar_const * numerator / denominator

	P[P < np.finfo(np.float).eps] = np.finfo(np.float).eps

	if lidar_const>1: # lidar_const = 1 for cases of calculating P_mol without constant
		if add_photon_noise:
			std_P = np.sqrt(P)
			rand_P = std_P * np.random.normal(loc=0, scale=1.0, size=P.shape)
			P = P + rand_P
		P = P.round() # converting P to photons counts

	P[P < np.finfo(np.float).eps] = np.finfo(np.float).eps

	return P


# %%
def calc_S(heights, P):
	"""Calculate the Logaritmic range adjusted power S(r) = ln(r^2*P(r))"""
	rr = np.diag(np.power(heights, 2))
	S = np.matmul(rr, P)
	return S


# %%
def calc_extiction_klett(S, heights, sigma_0, ind_m, k=1):
	"""
	Calculate the inversion of the extinction coefficient using Klett method
	"""
	S_m = S[ind_m]
	sigma_m = sigma_0[ind_m]

	dr = heights[1] - heights[0]
	exp_S = np.exp((S - S_m) / k)
	denominator = (1 / (sigma_m) + (2 / k) * np.flip(np.cumsum(np.flip(exp_S, 0) * dr), 0))
	sigma = exp_S / (denominator + eps)
	return sigma


# %%
def visCurve(lData, rData, stitle=""):
	'''Visualize 2 curves '''

	fnt_size = 18
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))
	ax = axes.ravel()

	for (x_j, y_j) in zip(lData['x'], lData['y']):
		ax[0].plot(x_j, y_j)
	if lData.__contains__('legend'):
		ax[0].legend(lData['legend'], fontsize=fnt_size - 6)
	ax[0].set_xlabel(lData['lableX'], fontsize=fnt_size, fontweight='bold')
	ax[0].set_ylabel(lData['lableY'], fontsize=fnt_size, fontweight='bold')
	ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	ax[0].set_title(lData['stitle'], fontsize=fnt_size, fontweight='bold')

	for (x_j, y_j) in zip(rData['x'], rData['y']):
		ax[1].plot(x_j, y_j)
	if rData.__contains__('legend'):
		ax[1].legend(rData['legend'], fontsize=fnt_size - 6)
	ax[1].set_xlabel(rData['lableX'], fontsize=fnt_size, fontweight='bold')
	ax[1].set_ylabel(rData['lableY'], fontsize=fnt_size, fontweight='bold')
	ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	ax[1].set_title(rData['stitle'], fontsize=fnt_size, fontweight='bold')

	fig.suptitle(stitle, fontsize=fnt_size + 4, va='top', fontweight='bold')
	fig.set_constrained_layout = True
	# fig.show()

	return [fig, axes]


# %%
def laplacian_operator(nx, ny, nz):
	if nx > 1 and ny == 1 and nz == 1:
		K = 1
	elif nx == 1 or ny == 1 or nz == 1:
		K = 2
	else:
		K = 3

	ex = np.ones((nx, 1))
	data_x = np.concatenate((ex, -K * ex, ex), axis=1).transpose()
	diags_x = np.array([-1, 0, 1])
	Lx = sprs.spdiags(data_x, diags_x, nx, nx)

	ey = np.ones((ny, 1))
	data_y = np.concatenate((ey, -K * ey, ey), axis=1).transpose()
	diags_y = np.array([-1, 0, 1])
	Ly = sprs.spdiags(data_y, diags_y, ny, ny)

	Ix = sprs.eye(nx)
	Iy = sprs.eye(ny)
	L2 = sprs.kron(Iy, Lx) + sprs.kron(Ly, Ix)

	N = nx * ny * nz
	e = np.ones((N, 1))
	data_z = np.concatenate((e, e), axis=1).transpose()
	diags_z = np.array([-nx * ny, nx * ny])
	L = sprs.spdiags(data_z, diags_z, N, N)
	Iz = sprs.eye(nz)

	A = sprs.kron(Iz, L2) + L
	return A


# %%
def calc_gauss_curve(t, A, H, t0, W):
	"""
	return gaussian approximation according the following parameters:
	:param t: Time parameter (or x-axis)
	:param A: Bias term (above y=0)
	:param H: The height of Gaussian curve
	:param t0: The center of the Gaussian lobe
	:param W: The width of Gaussian lobe
	:return: y = A+H*np.exp(-(t-t0)**2/(2*(W**2)))
	"""
	return A + H * np.exp(-(t - t0) ** 2 / (2 * (W ** 2)))


# %%
def generate_poisson_signal(mu, n):
	"""
	Generates a random value from the (discrete) Poisson
	distribution with parameter mu.
	:param mu: the parameter of Poisson distribution.
				Here : x~Poiss(mu):  mu = E(x) = Var(x)
	:param n: the number of randomized events
	:return x: the poisson signal
	:ref: https://people.smp.uq.edu.au/DirkKroese/mccourse.pdf, algorithm 3.4, p. 48
	"""
	x = np.zeros(n)
	for ii in range(n):
		k = 1
		a = 1
		a = a * np.random.rand()
		while a >= np.exp(-mu):
			a = a * np.random.rand()
			k = k + 1
		x[ii] = k - 1
	return x


# %%
def create_times_list(datatime_start, datatime_end, delta_time, type_time='seconds'):
	"""'
	Create time steps list for given start time end end time.

	datatime_start: 'datetime' of start time
	datatime_end: 'datetime' of end time
	delta_time: time resolution between times
	type_time: currently 'seconds' or 'days'. /TODO: add months later

	times : list of time in 'datetime' format

	required import: "from datetime import datetime, timedelta"
	"""
	if type_time == 'seconds':
		timestep = timedelta(seconds=delta_time)
		timeinterval = (datatime_end - datatime_start)
		totalsteps = np.int(timeinterval.total_seconds() / timestep.total_seconds()) + 1
	elif type_time == 'days':
		timestep = timedelta(days=delta_time)
		timeinterval = (datatime_end - datatime_start)
		totalsteps = np.int(timeinterval.days / timestep.days) + 1
	times = [datatime_start + timestep * step for step in range(0, totalsteps)]
	return times


# %%DEBUG -----------------------------
if __name__ == '__main__':
	delta_t = 30  # 100
	bins_per_hr = np.int(60 * 60 / delta_t)
	bins_per_day = 24 * bins_per_hr
	t = np.arange(0, bins_per_day)
	A = 0.03
	H = 1.12
	t0 = 1141.667
	W = 383.33
	print('gauss curve:', calc_gauss_curve(t, A, H, t0, W))
	print('This is miscellaneous functions file to Lidar')
