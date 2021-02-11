"""
Generate atmospheric test cases for debugging and evaluation purposes
"""
from __future__ import division
import logging
import numpy as np
import pandas as pd


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
		self._heights = df['HGHT'].values  # Meausrment heights [m]
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
			heights (array): 1D grid of heights in kilometers.(relative sea level)
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
			heights are relative to sea level
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
			heights are relative to sea level.
		"""

		relhs = np.interp(
			heights,
			self.height_kilometers,
			self._RelativeHumidity
		)

		return relhs

	def get_df_sonde(self):
		"""
		:return: DataFrame of ['TEMPS','PRES','RELHS'] of radiosonde (or gdas) measurements,
		interpolated according the height profile of the radiosonde file (1D grid in meters)
		heights are relative to sea level
		Temperature profiles is converted to Kelvin
		"""

		return pd.DataFrame(data={'TEMPS': self.temp_kelvin(), 'PRES': self._pressures,
		                          'RELHS': self._RelativeHumidity}, index = self._heights).astype('float64').fillna(0)

	def get_df_sonde(self, heights):
		"""
		:param heights: profile with the min_height, top_height and resolution set by the user
		:return: DataFrame of ['TEMPS','PRES','RELHS'] of radiosonde (or gdas) measurements interpolated according the input height
		Temperature profiles is converted to Kelvin
		"""
		return pd.DataFrame( data=
		                     {'TEMPS': self.interpolateKmKelvin(heights),
		                     'PRES': self.interpolateKMPres(heights),
		                     'RELHS': self.interpolateKMRLH(heights)}, index = heights ).astype('float64').fillna(0)

class LidarProfile(object):
	"""Temperature Profile Handling

	This object facilitates handling Lidar profiles.

	Args:
		lidar_path (str): path to Lidar file.
	"""

	def __init__(self, lidar_path, alpha_column="alpha532 (Mm^-1 sr^-1):", beta_column="Beta532 (Mm^-1 sr^-1):",
	             beta_mol_column='BetaMOL 532 (m^-1 sr^-1)', lidar_height=0.229, skiprows=None):
		import pandas as pd

		lidar_df = pd.read_csv(lidar_path, delimiter="\t",sep=" ", encoding='unicode_escape',
		                       index_col=0, skiprows=None)
		lidar_df[lidar_df < 0] = 0

		self._lidar_height = lidar_height
		self._heights = lidar_df.index.values + lidar_height
		self._extinction = lidar_df[alpha_column].values / 1000  # convert from Mm^-1 to Km^-1
		self._backscatter = lidar_df[beta_column].values / 1000  # convert from Mm^-1 to Km^-1
		self._backscatter_mol = lidar_df[beta_mol_column].values # TODO: check with Julin that the values are already 1/(km sr)

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


# --------------- Used for developing phase-function retrieval --------------#
# ---------------------------------------------------------------------------#

def generate_empty_atmosphere(size, npz, temp_profile):
	zlevels = np.linspace(0, size, npz)
	iparlev = np.zeros_like(zlevels)
	iparlev[5] = 1

	# Setup [3,3] matrices with the voxel in the middle padded with zeros
	dens_matrix, reff_matrix, veff_matrix = [np.zeros([3, 3, npz]) for i in range(3)]

	atm_params = {
		'density': dens_matrix,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'zlevels': zlevels,
		'temperature': temp_profile.interpolateKmKelvin(zlevels),
		'iparlev': iparlev,
		'dx': 1,
		'dy': 1,
		'origin': [0, 0, 0]
	}
	return atm_params


def pattern3D(X, Y):
	"""Create a 3D pattern.

	This creates an egg catron like pattern taken from the Mayavi
	surf example.
	"""

	w = np.sin(X + Y) + np.sin(X - Y) + np.cos(X + 1.5 * Y)

	w = w / (w.max() - w.min())
	w = w * 0.5 + 0.5

	return w


def generate_lidar_atmosphere(mre, mim, reff, veff, x, y, zlevels, temp_profile, lidar_profile=None,
                              scale_lidar=1.,
                              pattern_3d=None):
	"""Generate an atmosphere based on a lidar profile.

	Args:
		mre, mim, reff, veff (float): Particle parameters.
		x, y, zlevels (arrays): 1D arrays listing the atmosphere dims.
		temp_profile (Temperature object): Object handling temperature data.
		lidar_profile (Lidar object): Object handling lidar data.
		scale_lidar (float): Factor by which to scale atmosphere. Useful for
			simulating large extinication atmophsere.
		pattern_3d (optional, array): 3D pattern to overlay on the lidar profile.

	Returns:
		Dictionary holding atmosphere data.
	"""

	#
	# Create mesh grid.
	#
	X, Y, H = np.meshgrid(y, x, zlevels)

	#
	# SHDOM shit.
	#
	iparlev = np.ones_like(zlevels)
	iparlev[-1] = 0
	m = mre - mim * 1j

	#
	# Create layered density matrix according to the profile of
	# the Lidar.
	#
	if lidar_profile is None:
		density_matrix = np.zeros(
			shape=(len(x), len(y), len(zlevels)),
			order='F'
		)
	else:
		density_matrix = np.ones(
			shape=(len(x), len(y), len(zlevels)),
			order='F'
		) * lidar_profile.interpolate(zlevels).reshape(1, 1, -1)
		density_matrix *= scale_lidar

	profile_density = density_matrix.copy()

	#
	# Apply some 3D pattern.
	#
	if pattern_3d is not None:
		logging.info("Applying a 3D pattern to the Lidar atmosphere.")
		density_matrix = density_matrix * pattern_3d

	reff_matrix = np.ones_like(density_matrix) * reff
	veff_matrix = np.ones_like(density_matrix) * veff

	atm_params = {
		'density': density_matrix,
		'profile_density': profile_density,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'm': m,
		'zlevels': zlevels,
		'temperature': temp_profile.interpolateKmKelvin(zlevels),
		'iparlev': iparlev,
		'dx': x[1] - x[0],
		'dy': y[1] - y[0],
		'origin': [X.min(), Y.min(), zlevels.min()]
	}

	return atm_params


def generate_aerosol_single_voxel(density, refractive_index, reff, veff, size):
	#
	# Voxel parameters
	#
	Vdx, Vdy, Vdz = size, size, size
	temperature_profile_0_20km = [294.2, 289.7, 295.2, 279.2, 273.2, 287.2, 261.2, 254.7, 248.2, 241.7, \
	                              235.3, 228.8, 222.3, 215.8, 215.7, 215.7, 215.7, 215.7, 216.8, 217.9, 219.2]
	z_levels_0_20km = np.linspace(0, 20, len(temperature_profile_0_20km))
	zlevels = np.linspace(0, 2 * Vdz, 3)
	iparlev = np.array([0, 1, 0])

	#
	# Setup [3,3] matrices with the voxel in the middle padded with zeros
	#
	dens_matrix, reff_matrix, veff_matrix = [np.zeros([3, 3, 1]) for i in range(3)]
	dens_matrix[1, 1, 0], reff_matrix[1, 1, 0], veff_matrix[1, 1, 0] = density, reff, veff
	m_matrix = np.zeros([3, 3, 1], dtype=np.complex)
	m_matrix[1, 1, 0] = refractive_index

	atm_params = {
		'density': dens_matrix,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'm': m_matrix,
		'zlevels': zlevels,
		'temperature': np.interp(zlevels, z_levels_0_20km, temperature_profile_0_20km),
		'iparlev': iparlev,
		'dx': Vdx,
		'dy': Vdy,
		'origin': [0, 0, 0]
	}
	return atm_params


def generate_water_single_voxel(denstype='ext', density=100, reff=10, veff=0.1, size=0.02):
	# Voxel parameters
	Vdx, Vdy, Vdz = size, size, size
	temperature_profile_0_20km = [294.2, 289.7, 295.2, 279.2, 273.2, 287.2, 261.2, 254.7, 248.2, 241.7,\
	                              235.3, 228.8, 222.3, 215.8, 215.7, 215.7, 215.7, 215.7, 216.8, 217.9, 219.2]
	z_levels_0_20km = np.linspace(0, 20, len(temperature_profile_0_20km))
	zlevels = np.linspace(0, 2 * Vdz, 3)
	iparlev = np.array([0, 1, 0])

	# Setup [3,3] matrices with the voxel in the middle padded with zeros
	dens_matrix, reff_matrix, veff_matrix = [np.zeros([3, 3, 1]) for i in range(3)]
	dens_matrix[1, 1, 0], reff_matrix[1, 1, 0], veff_matrix[1, 1, 0] = density, reff, veff

	atm_params = {
		denstype: dens_matrix,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'zlevels': zlevels,
		'temperature': np.interp(zlevels, z_levels_0_20km, temperature_profile_0_20km),
		'iparlev': iparlev,
		'dx': Vdx,
		'dy': Vdy,
		'origin': [0, 0, 0]
	}
	return atm_params


# ------------------- Used for developing multi-scale RTE -------------------#
# ---------------------------------------------------------------------------#
def generate_blobs(distance, radius, sqrt_blob_number,
                   optical_depth=15, blob_height=3, rect_atmosphere='True'):
	# define atmospheric parameters
	domain_height = 20
	dx, dy, dz = .02, .02, .02
	temperature_profile_0_20km = [294.2, 289.7, 295.2, 279.2, 273.2, 287.2, 261.2, 254.7, 248.2, 241.7, \
	                              235.3, 228.8, 222.3, 215.8, 215.7, 215.7, 215.7, 215.7, 216.8, 217.9, 219.2]
	z_levels_0_20km = np.linspace(0, 20, len(temperature_profile_0_20km))
	zlevels = np.hstack([0, np.arange(blob_height - radius - dz, blob_height + radius + dz, dz), domain_height])

	ext = (optical_depth) / (2 * radius)
	ncells = int((2 * radius) / dz + 1)

	# Create a row of blobs
	blob = genBlob(ext, radius, ncells)
	air_east = np.zeros(shape=(ncells, int(distance / dy), ncells))
	row_extinction = blob
	for n in range(sqrt_blob_number - 1):
		row_extinction = np.concatenate((row_extinction, air_east, blob), axis=1)

	# Create a rectangular cloud field
	extinction = row_extinction
	if rect_atmosphere is True:
		air_north = np.zeros(shape=(int(distance / dx), row_extinction.shape[1], ncells))
		for n in range(sqrt_blob_number - 1):
			extinction = np.concatenate((extinction, air_north, row_extinction), axis=0)

	iparlev = np.concatenate((np.zeros(2),
	                          np.ones(extinction.shape[2]),
	                          np.zeros(2)))
	atm_params = {
		'ext': extinction,
		'reff': 10 * np.ones_like(extinction),
		'zlevels': zlevels,
		'temperature': np.interp(zlevels, z_levels_0_20km, temperature_profile_0_20km),
		'iparlev': iparlev,
		'dx': dx,
		'dy': dy,
		'origin': [0, 0, 0]
	}
	return atm_params


def genBlob(ext, radius, ncells):
	#
	# Create x and y indices
	#
	x = np.linspace(-1, 1, ncells)
	y = np.linspace(-1, 1, ncells)
	z = np.linspace(-1, 1, ncells)
	X, Y, Z = np.meshgrid(x, y, z)

	mask = X ** 2 + Y ** 2 + Z ** 2 <= radius

	extinction = ext * mask

	return extinction


def generate_grid_atmosphere(
		density,
		mre,
		mim,
		reff,
		veff,
		x, y, zlevels,
		temp_profile,
		cloud_coords=((0.5, 0.5, 1), (-0.5, -0.5, 1.5),),
		cloud_radiuses=((0.5, 0.5, .25), (0.5, 0.5, .25),),
):
	X, Y, H = np.meshgrid(y, x, zlevels)

	mask = np.zeros_like(Y)
	for coords, radiuses in zip(cloud_coords, cloud_radiuses):
		Z = ((Y - coords[0]) / radiuses[0]) ** 2 + ((X - coords[1]) / radiuses[1]) ** 2 + (
				(H - coords[2]) / radiuses[2]) ** 2
		mask[Z < 1] = 1

	dens_matrix = density * np.ones(
		shape=(len(x), len(y), len(zlevels)),
		order='F') * mask
	reff_matrix = np.ones_like(dens_matrix) * reff * mask
	veff_matrix = np.ones_like(dens_matrix) * veff * mask

	m = mre - mim * 1j

	iparlev = np.ones_like(zlevels)
	iparlev[-1] = 0

	atm_params = {
		'density': dens_matrix,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'm': m,
		'zlevels': zlevels,
		'temperature': temp_profile.interpolateKmKelvin(zlevels),
		'iparlev': iparlev,
		'dx': x[1] - x[0],
		'dy': y[1] - y[0],
		'origin': [X.min(), Y.min(), zlevels.min()]
	}

	return atm_params


def generate_uniform_atmosphere(
		density,
		mre,
		mim,
		reff,
		veff,
		x, y, zlevels,
		temp_profile,
):
	dens_matrix = density * np.ones(
		shape=(len(x), len(y), len(zlevels)),
		order='F')
	reff_matrix = np.ones_like(dens_matrix) * reff
	veff_matrix = np.ones_like(dens_matrix) * veff

	m = mre - mim * 1j

	iparlev = np.ones_like(zlevels)
	iparlev[-1] = 0

	atm_params = {
		'density': dens_matrix,
		'reff': reff_matrix,
		'veff': veff_matrix,
		'm': m,
		'zlevels': zlevels,
		'temperature': temp_profile.interpolateKmKelvin(zlevels),
		'iparlev': iparlev,
		'dx': x[1] - x[0],
		'dy': y[1] - y[0],
		'origin': [x.min(), y.min(), zlevels.min()]
	}

	return atm_params
