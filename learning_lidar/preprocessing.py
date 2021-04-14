# %% Imports

import warnings  # Ignore warnings

warnings.filterwarnings("ignore")
import pandas as pd
from pandas.api.types import is_numeric_dtype
import os
from datetime import datetime, timedelta, time, date
import glob
from molecular import rayleigh_scattering
import numpy as np
from generate_atmosphere import RadiosondeProfile
import re
from netCDF4 import Dataset
import fnmatch
import matplotlib.pyplot as plt
import global_settings as gs
import miscLidar as mscLid
from utils_.utils import create_and_configer_logger, write_row_to_csv
import logging
import torch, torchvision
import torch.utils.data
import xarray as xr
import matplotlib.dates as mdates
from pytictoc import TicToc
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


# %% General functions


def get_month_folder_name(parent_folder, day_date):
    month_folder = os.path.join(parent_folder, day_date.strftime("%Y"), day_date.strftime("%m"))
    return (month_folder)


def extract_date_time(path, format_filename, format_times):
    """
    Extracting datetime from file name using a formatter string.
    :param path: path to folder containing the files to observe
    :param format_filename:  a formatter string
    :param format_times: format of datetime in the file name
    :return: time_stamps - A list of datetime.datetime objects
    usage:  create a formatter string: format_filename=  r'-(.*)_(.*)-(.*)-.*.txt'
            Call the function: time_stamp = extract_date_time(soundePath,r'40179_(.*).txt',['%Y%m%d_%H'])

    """
    filename = os.path.basename(path)
    matchObj = re.match(format_filename, filename, re.M | re.I)
    time_stamps = []
    for fmt_time, grp in zip(format_times, matchObj.groups()):
        time_stamps.append(datetime.strptime(grp, fmt_time))
    return time_stamps


def save_dataset(dataset, folder_name, nc_name):
    """
    Save the input dataset to netcdf file
    :param dataset: array.Dataset()
    :param folder_name: folder name
    :param nc_name: netcdf file name
    :return: ncpath - full path to netcdf file created if succeeded, else none
    """
    logger = logging.getLogger()
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            logger.debug(f"Creating folder: {folder_name}")
        except Exception:
            logger.exception(f"Failed to create folder: {folder_name}")
            return None

    ncpath = os.path.join(folder_name, nc_name)
    try:
        dataset.to_netcdf(ncpath, mode='w', format='NETCDF4', engine='netcdf4')
        dataset.close()
        logger.debug(f"Saving dataset file: {ncpath}")
    except Exception:
        logger.exception(f"Failed to save dataset file: {ncpath}")
        ncpath = None
    return ncpath


def load_dataset(ncpath):
    """
    Load Dataset stored in the netcdf file path (ncpath)
	:param ncpath: a netcdf file path
	:return: xarray.Dataset, if fails return none
	"""
    logger = logging.getLogger()
    try:
        dataset = xr.open_dataset(ncpath, engine='netcdf4').expand_dims()
        dataset.close()
        logger.debug(f"Loading dataset file: {ncpath}")
    except Exception:
        logger.exception(f"Failed to load dataset file: {ncpath}")
        return None
    return dataset


# %% Functions to handle TROPOS datasets (netcdf) files


def get_daily_molecular_profiles(station, day_date, lambda_nm=532, height_units='km'):
    """
    Generating daily molecular profile from gdas txt file
    :param station: gs.station() object of the lidar station
    :param gdas_txt_paths: paths to gdas txt files , containing table with the columns
    "PRES	HGHT	TEMP	UWND	VWND	WWND	RELH	TPOT	WDIR	WSPD"
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm]
    :param height_units: units of height grid in 'km' (default) or 'm'
    this parameter gives dr= heights[1]-heights[0] =~ 7.5e-3[km] -> the height resolution of pollyXT lidar)
    :return: Returns backscatter and extinction profiles as pandas-dataframes, according to Rayleigh scattering.
    The resulted profile has a grid height above (in 'km' or 'm' - according to input), above sea level. start at min_height, end at top_height and extrapolated to have h_bins.
    """
    logger = logging.getLogger()

    '''Load GDAS measurements through 24 hrs of day_date'''
    _, gdas_curday_paths = get_daily_gdas_paths(station, day_date, 'txt')
    if not gdas_curday_paths:
        logger.debug(f"For {day_date.strftime('%Y/%m/%d')}, "
                     f"there are not the required GDAS '.txt' files. Starting conversion from '.gdas1'")
        gdas_curday_paths = convert_daily_gdas(station, day_date)

    next_day = day_date + timedelta(days=1)
    _, gdas_nxtday_paths = get_daily_gdas_paths(station, next_day, 'txt')
    if not gdas_nxtday_paths:
        logger.debug(f"For {day_date.strftime('%Y/%m/%d')}, "
                     f"there are not the required GDAS '.txt' files. Starting conversion from '.gdas1'")
        gdas_nxtday_paths = convert_daily_gdas(station, next_day)

    gdas_txt_paths = gdas_curday_paths
    gdas_txt_paths.append(gdas_nxtday_paths[0])
    timestamps = [get_gdas_timestamp(station, path) for path in gdas_txt_paths]

    '''Setting height vector above sea level (for interpolation of radiosonde / gdas files).'''
    if height_units == 'km':
        scale = 1E-3
    elif height_units == 'm':
        scale = 1

    min_height = station.altitude + station.start_bin_height
    top_height = station.altitude + station.end_bin_height
    heights = np.linspace(min_height * scale, top_height * scale, station.n_bins)
    # uncomment the commands below for sanity check of the desired height resolution
    # dr = heights[1]-heights[0]
    # print('h_bins=',heights.shape,' min_height=', min_height,' top_height=', top_height,' dr=',dr )

    df_sigma = pd.DataFrame(index=heights).rename_axis(f'Height[{height_units}]')
    df_beta = pd.DataFrame(index=heights).rename_axis(f'Height[{height_units}]')

    for path, timestamp in zip(gdas_txt_paths, timestamps):
        df_sonde = RadiosondeProfile(path).get_df_sonde(heights)
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply(calc_sigma_profile_df, axis=1, args=(lambda_nm, timestamp,),
                             result_type='expand').astype('float64')
        df_sigma[res.columns] = res
        res = df_sonde.apply(calc_beta_profile_df, axis=1, args=(lambda_nm, timestamp,),
                             result_type='expand').astype('float64')
        df_beta[res.columns] = res

    return df_sigma, df_beta


def get_TROPOS_dataset_file_name(start_time=None, end_time=None, file_type='profiles'):
    """
    Retrieves file pattern name of TROPOS start time, end time and  profile type.
    :param start_time: datetime.datetime object specifying specific start time of measurement or profile analysis
    :param end_time: datetime.datetime object specifying specific end time of profile analysis
    :param file_type: type of data stored in the files, E.g.: 'profiles', 'att_bsc', 'overlap', 'OC_att_bsc', 'cloudinfo', etc.
    E.g.: for analyzed profiles "*_<start_time>_<end_time>_<file_type>.nc" ( or  "*<file_type>.nc" if start_time = None, end_time = None)
          for daily attenuation backscatter profile "*<att_bsc>.nc" ( or  "*<start_time>_<file_type>.nc" if start_time is given)
    """
    if start_time and end_time and ('profiles' in file_type):
        pattern = f"*[0-9]_{start_time.strftime('%H%M')}_{end_time.strftime('%H%M')}_{file_type}.nc"
    elif start_time:
        pattern = f"*{start_time.strftime('%H_%M_%S')}_{file_type}.nc"
    else:
        pattern = f"*[0-9]_{file_type}.nc"
    return pattern


def get_TROPOS_dataset_paths(station, day_date, start_time=None, end_time=None, file_type='profiles'):
    """
    Retrieves netcdf (.nc) files from TROPOS for a given station and day_date, and type.

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param start_time: datetime.datetime object specifying specific start time of measurement or profile analysis
    :param end_time: datetime.datetime object specifying specific end time of profile analysis
    :param file_type: type of data stored in the files, E.g.: 'profiles', 'att_bsc', 'overlap', 'OC_att_bsc', 'cloudinfo', etc.
    E.g.: for analyzed profiles "*_<start_time>_<end_time>_<file_type>.nc" ( or  "*<file_type>.nc" if start_time = None, end_time = None)
          for daily attenuation backscatter profile "*<att_bsc>.nc" ( or  "*<start_time>_<file_type>.nc" if start_time is given)
    :return: paths: paths to all *<file_type>.nc for a given station and day_date
    """
    lidar_day_folder = get_TROPOS_day_folder_name(station.lidar_src_folder, day_date)
    file_name = get_TROPOS_dataset_file_name(start_time, end_time, file_type)
    paths_pattern = os.path.join(lidar_day_folder, file_name)

    paths = sorted(glob.glob(paths_pattern))

    return paths


def get_TROPOS_day_folder_name(parent_folder, day_date):
    moth_folder = get_month_folder_name(parent_folder, day_date)
    day_folder = os.path.join(moth_folder, day_date.strftime("%d"))
    return day_folder


def extract_att_bsc(bsc_paths, wavelengths):
    """
    For all .nc files under bsc_paths and for each wavelength in wavelengths
    extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used

    :param bsc_paths: path to netcdf folder
    :param wavelengths: iterable, list of wavelengths
    :return:
    """
    logger = logging.getLogger()
    for wavelength in wavelengths:
        for bsc_path in bsc_paths:
            data = Dataset(bsc_path)
            file_name = bsc_path.split('/')[-1]
            try:
                vals = data.variables[f'OC_attenuated_backscatter_{wavelength}nm']
                arr = vals * vals.Lidar_calibration_constant_used
                logger.debug(f"Extracted OC_attenuated_backscatter_{wavelength}nm from {file_name}")
            except KeyError as e:
                logger.exception(f"Key {e} does not exist in {file_name}", exc_info=False)


# %% GDAS files preprocessing functions
def get_gdas_timestamp(station, path, file_type='txt'):
    format_times = ["%Y%m%d_%H"]
    format_filename = f"{station.location.lower()}_(.*)_{station.lat:.1f}_{station.lon:.1f}.{file_type}"
    date_time = extract_date_time(path, format_filename, format_times)[0]
    return date_time


def gdas2radiosonde(src_file, dst_file, col_names=None):
    """
    Helper function that converts a gdas file from TROPOS server (saved as .gdas1), to a simple txt.
    The resulting file is without any prior info, and resembles the table format
    of a radiosonde file (see class: RadiosondeProfile).
    :param src_file: source file name
    :param dst_file: destination file name
    :param col_names: column names of the final table
    :return:
    """
    logger = logging.getLogger()
    if col_names is None:
        col_names = ['PRES', 'HGHT', 'TEMP', 'UWND', 'VWND', 'WWND', 'RELH', 'TPOT', 'WDIR', 'WSPD']
    try:
        data_src = pd.read_fwf(src_file, skiprows=[0, 1, 2, 3, 4, 5, 6, 8], delimiter="\s+",
                               skipinitialspace=True).dropna()
    except Exception:
        logger.exception(f'Failed reading {src_file}. Check the source file, '
                         'or generate it again with ARLreader module')
        write_row_to_csv('../gdas2radiosonde_failed_files.csv', [src_file, 'Read Fail', 'Broken'])
        return None

    # converting any kind of blank spaces to zeros
    try:
        for col in data_src.columns:
            if not is_numeric_dtype(data_src[col]):
                data_src[col] = pd.core.strings.str_strip(data_src[col])
                data_src[col] = data_src[col].replace('', '0').astype('float64')
        data_src.columns = col_names
        data_src.to_csv(dst_file, index=False, sep='\t', na_rep='\t')
    except Exception:
        logger.exception(f'Conversion of {src_file} to {dst_file} failed. Check the source file, '
                         'or generate it again with ARLreader module')
        write_row_to_csv('../gdas2radiosonde_failed_files.csv', [src_file, 'Conversion Fail', 'Broken'])
        dst_file = None
    return dst_file


def get_daily_gdas_paths(station, day_date, f_type='gdas1'):
    """
    Retrieves gdas container folder and file paths for a given date
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param f_type: 'gdas1' - for the original gdas files (from TROPOS), 'txt' - for converted (fixed) gdas files
    :return: gdas_folder, gdas_paths - the folder containing the gdas files and the file paths.
    NOTE: during a daily conversion, the function creates a NEW gdas_folder (for the converted txt files),
    and returns an EMPTY list of gdas_paths. The creation of NEW gdas_paths is done in convert_daily_gdas()
    """
    logger = logging.getLogger()

    if f_type == 'gdas1':
        parent_folder = station.gdas1_folder
    elif f_type == 'txt':
        parent_folder = station.gdastxt_folder
    month_folder = get_month_folder_name(parent_folder, day_date)
    if not os.path.exists(month_folder):
        try:
            os.makedirs(month_folder)
        except:
            logger.exception(f"Failed to create folder: {month_folder}")

    gdas_day_pattern = '{}_{}_*_{:.1f}_{:.1f}.{}'.format(station.location.lower(), day_date.strftime('%Y%m%d'),
                                                         station.lat, station.lon, f_type)
    path_pattern = os.path.join(month_folder, gdas_day_pattern)
    gdas_paths = sorted(glob.glob(path_pattern))
    return month_folder, gdas_paths


def get_gdas_file_name(station, time, f_type='txt'):
    file_name = '{}_{}_{}_{:.1f}_{:.1f}.{}'.format(station.location.lower(), time.strftime('%Y%m%d'),
                                                   time.strftime('%H'),
                                                   station.lat, station.lon, f_type)

    return (file_name)


def convert_daily_gdas(station, day_date):
    """
    Converting gdas files from TROPOS of type .gdas1 to .txt for a given day.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the date to be converted
    :return:
    """
    # get source container folder and file paths (.gdas1)
    src_folder, gdas1_paths = get_daily_gdas_paths(station, day_date, 'gdas1')

    # set dest container folder and file paths (.txt)
    dst_folder, _ = get_daily_gdas_paths(station, day_date, 'txt')
    path_to_convert = [sub.replace(src_folder, dst_folder).replace('gdas1', 'txt') for sub in gdas1_paths]
    converted_paths = []
    # convert each src_file (as .gdas1) to dst_file (as .txt)
    for (src_file, dst_file) in zip(gdas1_paths, path_to_convert):
        converted = gdas2radiosonde(src_file, dst_file)
        if converted:
            converted_paths.append(converted)

    return converted_paths


def convert_periodic_gdas(station, start_day, end_day):
    logger = logging.getLogger()

    day_dates = pd.date_range(start=start_day, end=end_day, freq=timedelta(days=1))
    expected_file_no = len(day_dates) * 8  # 8 timestamps per day
    gdastxt_paths = []
    for day in day_dates:
        gdastxt_paths.extend(convert_daily_gdas(station, day))
    total_converted = len(gdastxt_paths)
    logger.debug(f"Done conversion of {total_converted} gdas files for period [{start_day.strftime('%Y-%m-%d')},"
                 f"{end_day.strftime('%Y-%m-%d')}], {(expected_file_no - total_converted)} failed.")
    return gdastxt_paths


# %% Molecular preprocessing and dataset functions


def calc_sigma_profile_df(row, lambda_nm=532.0, indx_n='sigma'):
    """
    Returns pd series of extinction profile [1/m] from a radiosonde dataframe containing the
    columns:['PRES','TEMPS','RELHS']. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
    :param indx_n: index name, the column name of the result. The default is 'sigma'.
    This could be useful to get profiles for several hours each have a different index name
    (e.g., datetime.datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd series of extinction profile in units of [1/m]
    """
    return pd.Series(
        data=[rayleigh_scattering.alpha_rayleigh(wavelength=lambda_nm, pressure=row['PRES'],
                                                 temperature=row['TEMPS'], C=385.0,
                                                 rh=row['RELHS'])],
        index=[indx_n])


def cal_e_tau_df(col, altitude):
    """
    Calculate the the attenuated optical depth (tau is the optical depth).
    Calculation is per column of the backscatter (sigma) dataframe.
    :param col: column of sigma_df , the values of sigma should be in [1/m]
    :param altitude: the altitude of the the lidar station above sea level, in [m]
    :return: Series of exp(-2*tau)  , tau = integral( sigma(r) * dr)
    """

    heights = col.index.to_numpy() - altitude
    tau = mscLid.calc_tau(col, heights)
    return pd.Series(np.exp(-2 * tau))


def calc_beta_profile_df(row, lambda_nm=532.0, ind_n='beta'):
    """
    Returns pd series of backscatter profile from a radiosonde dataframe containing the
    columns:['PRES','TEMPS','RELHS']. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
    :param ind_n: index name, the column name of the result. The default is 'beta'
    but could be useful to get profiles for several hours each have a different index name
    (e.g., datetime.datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd series of backscatter profile [1/sr*m]
    """
    return pd.Series([rayleigh_scattering.beta_pi_rayleigh(wavelength=lambda_nm,
                                                           pressure=row['PRES'],
                                                           temperature=row['TEMPS'],
                                                           C=385.0, rh=row['RELHS'])],
                     index=[ind_n])


def generate_daily_molecular_chan(station, day_date, lambda_nm, time_res='30S',
                                  height_units='km', optim_size=False, verbose=False):
    """
	Generating daily molecular profiles for a given channel's wavelength
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
	:param time_res: Output time resolution required. default is 30sec (according to time resolution of pollyXT measurements)
	:param height_units:  Output units of height grid in 'km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 4 data variables:
	3 daily dataframes: beta,sigma,att_bsc with shared dimensions (Height, Time)
	and
	1 shared variable: lambda_nm with dimension (Wavelength)
	"""
    logger = logging.getLogger()
    '''Load daily gdas profiles and convert to backscatter (beta) and extinction (sigma) profiles'''

    df_sigma, df_beta = get_daily_molecular_profiles(station, day_date, lambda_nm, height_units)

    ''' Interpolate profiles through 24 hrs'''
    interp_sigma_df = (df_sigma.T.resample(time_res).interpolate(method='linear')[:-1]).T
    interp_beta_df = (df_beta.T.resample(time_res).interpolate(method='linear')[:-1]).T
    interp_sigma_df.columns.freq = None
    interp_beta_df.columns.freq = None

    '''Calculate the molecular attenuated backscatter as :  beta_mol * exp(-2*tau_mol)'''
    if height_units == 'km':
        # converting height index to meters before tau calculations
        km_index = interp_sigma_df.index
        idx_name = km_index.name
        meter_index = 1e+3 * km_index.rename(idx_name.replace('km', 'm'))
        interp_sigma_df.reset_index(inplace=True, drop=True)
        interp_sigma_df.index = meter_index
    e_tau_df = interp_sigma_df.apply(cal_e_tau_df, 0, args=(station.altitude,), result_type='expand')
    if height_units == 'km':
        # converting back height index to km before dataset creation
        interp_sigma_df.reset_index(inplace=True, drop=True)
        interp_sigma_df.index = km_index
        e_tau_df.reset_index(inplace=True, drop=True)
        e_tau_df.index = km_index

    att_bsc_mol_df = interp_beta_df.multiply(e_tau_df)

    ''' memory size - optimization '''
    if optim_size:
        if verbose:
            logger.debug('Memory optimization - converting molecular values from double to float')
            size_beta = interp_beta_df.memory_usage(deep=True).sum()
            size_sigma = interp_sigma_df.memory_usage(deep=True).sum()
            size_att_bsc = att_bsc_mol_df.memory_usage(deep=True).sum()

        interp_beta_df = (interp_beta_df.select_dtypes(include=['float64'])). \
            apply(pd.to_numeric, downcast='float')
        interp_sigma_df = (interp_sigma_df.select_dtypes(include=['float64'])). \
            apply(pd.to_numeric, downcast='float')
        att_bsc_mol_df = (att_bsc_mol_df.select_dtypes(include=['float64'])). \
            apply(pd.to_numeric, downcast='float')

        if verbose:
            size_beta_opt = interp_beta_df.memory_usage(deep=True).sum()
            size_sigma_opt = interp_sigma_df.memory_usage(deep=True).sum()
            size_att_bsc_opt = att_bsc_mol_df.memory_usage(deep=True).sum()
            logger.debug('Memory saved for wavelength {} beta: {:.2f}%, sigma: {:.2f}%, att_bsc:{:.2f}%'.
                         format(lambda_nm, 100.0 * float(size_beta - size_beta_opt) / float(size_beta),
                                100.0 * float(size_sigma - size_sigma_opt) / float(size_sigma),
                                100.0 * float(size_att_bsc - size_att_bsc_opt) / float(size_att_bsc)))

    ''' Create molecular dataset'''
    ds_chan = xr.Dataset(
        data_vars={'beta': (('Height', 'Time'), interp_beta_df),
                   'sigma': (('Height', 'Time'), interp_sigma_df),
                   'attbsc': (('Height', 'Time'), att_bsc_mol_df),
                   'lambda_nm': ('Wavelength', np.uint16([lambda_nm]))
                   },
        coords={'Height': interp_beta_df.index.to_list(),
                'Time': interp_beta_df.columns,
                'Wavelength': np.uint16([lambda_nm])
                }
    )

    # set attributes of data variables
    ds_chan.beta.attrs = {'long_name': r'$\beta$', 'units': r'$1/m \cdot sr$',
                          'info': 'Molecular backscatter coefficient'}
    ds_chan.sigma.attrs = {'long_name': r'$\sigma$', 'units': r'$1/m $',
                           'info': 'Molecular attenuation coefficient'}
    ds_chan.attbsc.attrs = {'long_name': r'$\beta \cdot \exp(-2\tau)$', 'units': r'$1/m \cdot sr$',
                            'info': 'Molecular attenuated backscatter coefficient'}
    # set attributes of coordinates
    ds_chan.Height.attrs = {'units': fr'${height_units}$', 'info': 'Measurements heights above sea level'}
    ds_chan.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}

    return ds_chan


def generate_daily_molecular(station, day_date, time_res='30S', height_units='km',
                             optim_size=False, verbose=False, USE_KM_UNITS=True):
    """
	Generating daily molecular profiles for all elastic channels (355,532,1064)
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param time_res: Output time resolution required. default is 30sec (according to time resolution of pollyXT measurements)
	:param height_units:  Output units of height grid in 'km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 5 data variables:
			 3 daily dataframes: beta,sigma,att_bsc with shared dimensions(Height, Time, Wavelength)
			 and 2 shared variables: lambda_nm with dimension (Wavelength), and date
	"""

    date_datetime = datetime.combine(date=day_date, time=time.min) \
        if isinstance(day_date, date) else day_date

    wavelengths = gs.LAMBDA_nm().get_elastic()
    ds_list = []
    # t = TicToc()
    # t.tic()
    for lambda_nm in wavelengths:
        ds_chan = generate_daily_molecular_chan(station, date_datetime, lambda_nm, time_res=time_res,
                                                height_units=height_units, optim_size=optim_size,
                                                verbose=verbose)
        ds_list.append(ds_chan)
    # t.toc()
    '''concatenating molecular profiles of all channels'''
    mol_ds = xr.concat(ds_list, dim='Wavelength')
    mol_ds['date'] = date_datetime
    mol_ds.attrs = {'info': 'Daily molecular profiles',
                    'location': station.name,
                    'source_type': 'gdas'}
    if USE_KM_UNITS:
        mol_ds = convert_profiles_units(mol_ds, units=['1/m', '1/km'], scale=1e+3)

    return mol_ds


def save_molecular_dataset(station, dataset, save_mode='sep'):
    """
	Save the input dataset to netcdf file
	:param station: station: gs.station() object of the lidar station
	:param dataset: array.Dataset() holding 5 data variables:
					3 daily dataframes: beta,sigma,att_bsc with shared dimensions(Height, Time, Wavelength)
					and 2 shared variables: lambda_nm with dimension (Wavelength), and date
	:param save_mode: save mode options:
					'sep' - for separated profiles (each is file is per profile per wavelength)
					'single' - save the dataset a single file per day
					'both' -saving both options
	:return: ncpaths - the paths of the saved dataset/s . None - for failure.
	"""
    date_datetime = get_daily_ds_date(dataset)
    month_folder = get_month_folder_name(station.molecular_dataset, date_datetime)

    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []
    if save_mode in ['both', 'sep']:
        data_vars = list(dataset.data_vars)
        profiles = data_vars[0:3]
        shared_vars = data_vars[3:]
        # print('The data variables for profiles:' ,profiles) #For debug.should be : 'beta','sigma','attbsc'
        # print('The shared variables: ',shared_vars) #For debug.should be : 'lambda_nm' (wavelength),'date'
        for profile in profiles:
            profile_vars = [profile]
            for lambda_nm in dataset.Wavelength.values:
                profile_vars.extend(shared_vars)
                ds_profile = dataset.get(profile_vars).sel(Wavelength=lambda_nm)
                file_name = get_prep_dataset_file_name(station, date_datetime, data_source='molecular',
                                                       lambda_nm=lambda_nm,
                                                       file_type=profile)
                ncpath = save_dataset(ds_profile, month_folder, file_name)
                if ncpath:
                    ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_prep_dataset_file_name(station, date_datetime, data_source='molecular',
                                               lambda_nm='all',
                                               file_type='all')
        ncpath = save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


# %% lidar preprocessing and dataset functions


def get_daily_range_corr(station, day_date, height_units='km',
                         optim_size=False, verbose=False, USE_KM_UNITS=True):
    """
	Retrieving daily range corrected lidar signal (pr^2) from attenuated_backscatter signals in three channels (355,532,1064).
	The attenuated_backscatter are from 4 files of 6-hours *att_bsc.nc for a given day_date and station
    :param USE_KM_UNITS:
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param height_units:  Output units of height grid in 'km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
			 1 daily dataset of range_corrected signal in 3 channels, with dimensions of : Height, Time, Wavelength
			 3 variables : lambda_nm, plot_min_range, plot_max_range, with dimension of : Wavelength
			 1 shared variable: date
	"""

    '''get netcdf paths of the attenuation backscatter for given day_date'''
    date_datetime = datetime.combine(date=day_date, time=time.min) \
        if isinstance(day_date, date) else day_date

    bsc_paths = get_TROPOS_dataset_paths(station, date_datetime, file_type='att_bsc')
    bsc_ds0 = load_dataset(bsc_paths[0])
    altitude = bsc_ds0.altitude.values[0]
    profiles = [dvar for dvar in list(bsc_ds0.data_vars) if 'attenuated_backscatter' in dvar]
    wavelengths = [np.uint(pname.split(sep='_')[-1].strip('nm')) for pname in profiles]

    min_range = np.empty((len(wavelengths), len(bsc_paths)))
    max_range = np.empty((len(wavelengths), len(bsc_paths)))

    ds_range_corrs = []
    for ind_path, bsc_path in enumerate(bsc_paths):
        cur_ds = load_dataset(bsc_path)
        '''get 6-hours range corrected dataset for three channels [355,532,1064]'''
        ds_chans = []
        for ind_wavelength, (pname, lambda_nm) in enumerate(zip(profiles, wavelengths)):
            cur_darry = cur_ds.get(pname).transpose(transpose_coords=True)
            ds_chan, LC = get_range_corr_ds_chan(cur_darry, altitude, lambda_nm, height_units, optim_size,
                                                 verbose)
            min_range[ind_wavelength, ind_path] = LC * cur_darry.attrs['plot_range'][0]
            max_range[ind_wavelength, ind_path] = LC * cur_darry.attrs['plot_range'][1]
            ds_chans.append(ds_chan)

        cur_ds_range_corr = xr.concat(ds_chans, dim='Wavelength')
        ds_range_corrs.append(cur_ds_range_corr)

    '''merge range corrected of lidar through 24-hours'''
    range_corr_ds = xr.merge(ds_range_corrs, compat='no_conflicts')

    # Fixing missing timestamps values:
    time_indx = pd.date_range(start=date_datetime,
                              end=(date_datetime + timedelta(hours=24) - timedelta(seconds=30)),
                              freq='30S')
    range_corr_ds = range_corr_ds.reindex({"Time": time_indx}, fill_value=0)
    range_corr_ds = range_corr_ds.assign({'plot_min_range': ('Wavelength', min_range.min(axis=1)),
                                          'plot_max_range': ('Wavelength', max_range.max(axis=1))})
    range_corr_ds['date'] = date_datetime
    range_corr_ds.attrs = {'location': station.location,
                           'info': 'Daily range corrected lidar signal',
                           'source_type': 'att_bsc'}
    if USE_KM_UNITS:
        range_corr_ds = convert_profiles_units(range_corr_ds, units=[r'$m^2$', r'$km^2$'], scale=1e-6)
    return range_corr_ds


def get_range_corr_ds_chan(darray, altitude, lambda_nm, height_units='km', optim_size=False,
                           verbose=False):
    """
	Retrieving a 6-hours range corrected lidar signal (pr^2) from attenuated_backscatter signals in three channels (355,532,1064).
	The attenuated_backscatter are from a 6-hours *att_bsc.nc loaded earlier by darray
	:param darray: is xarray.DataArray object, containing a 6-hours of attenuated_backscatter (loaded from TROPOS *att_bsc.nc)
	:param altitude: altitude of the station [m]
	:param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
	:param height_units:  Output units of height grid in 'km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 2 data variables:
	         1. 6-hours dataframe: range_corr with dimensions (Height, Time)
			 2. shared variable: lambda_nm with dimension (Wavelength)
			 LC - The lidar constant calibration used
	"""
    logger = logging.getLogger()
    LC = darray.attrs['Lidar_calibration_constant_used']
    times = pd.to_datetime(
        [datetime.utcfromtimestamp(np.round(vtime)) for vtime in darray.time.values]).values
    if height_units == 'km':
        scale = 1e-3
    elif height_units == 'm':
        scale = 1
    heights_ind = scale * (darray.height.values + altitude)
    rangecorr_df = pd.DataFrame(LC * darray.values, index=heights_ind, columns=times)

    ''' memory size - optimization '''
    if optim_size:
        if verbose:
            logger.debug('Memory optimization - converting molecular values from double to float')
            size_rangecorr = rangecorr_df.memory_usage(deep=True).sum()

        rangecorr_df = (rangecorr_df.select_dtypes(include=['float64'])). \
            apply(pd.to_numeric, downcast='float')

        if verbose:
            size_rangecorr_opt = rangecorr_df.memory_usage(deep=True).sum()
            logger.debug('Memory saved for wavelength {} range corrected: {:.2f}%'.
                         format(lambda_nm,
                                100.0 * float(size_rangecorr - size_rangecorr_opt) / float(size_rangecorr)))

    ''' Create range_corr_chan lidar dataset'''
    range_corr_ds_chan = xr.Dataset(
        data_vars={'range_corr': (('Height', 'Time'), rangecorr_df),
                   'lambda_nm': ('Wavelength', [lambda_nm])
                   },
        coords={'Height': rangecorr_df.index.to_list(),
                'Time': rangecorr_df.columns,
                'Wavelength': [lambda_nm]
                }
    )
    range_corr_ds_chan.range_corr.attrs = {'long_name': r'$LC \beta \cdot \exp(-2\tau)$',
                                           'units': r'$photons$' + r'$\cdot$' + r'$m^2$',
                                           'info': 'Range corrected lidar signal from attenuated backscatter multiplied by LC'}
    # set attributes of coordinates
    range_corr_ds_chan.Height.attrs = {'units': fr'${height_units}$',
                                       'info': 'Measurements heights above sea level'}
    range_corr_ds_chan.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}

    return range_corr_ds_chan, LC


def save_range_corr_dataset(station, dataset, save_mode='sep'):
    """
	Save the input dataset to netcdf file
	:param station: station: gs.station() object of the lidar station
	:param dataset: array.Dataset() a daily range corrected lidar signal, holding 5 data variables:
			 1 daily dataset of range_corrected signal in 3 channels, with dimensions of : Height, Time, Wavelength
			 3 variables : lambda_nm, plot_min_range, plot_max_range, with dimension of : Wavelength
			 1 shared variable: date
	:param save_mode: save mode options:
					'sep' - for separated profiles (each is file is per profile per wavelength)
					'single' - save the dataset a single file per day
					'both' -saving both options
	:return: ncpaths - the paths of the saved dataset/s . None - for failure.
	"""

    date_datetime = get_daily_ds_date(dataset)
    month_folder = get_month_folder_name(station.lidar_dataset, date_datetime)

    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []
    profile = list(dataset.data_vars)[0]
    if save_mode in ['both', 'sep']:
        for lambda_nm in dataset.Wavelength.values:
            ds_profile = dataset.sel(Wavelength=lambda_nm)
            file_name = get_prep_dataset_file_name(station, date_datetime, data_source='lidar',
                                                   lambda_nm=lambda_nm, file_type=profile)
            ncpath = save_dataset(ds_profile, month_folder, file_name)
            if ncpath:
                ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_prep_dataset_file_name(station, date_datetime, data_source='lidar',
                                               lambda_nm='all', file_type='all')
        ncpath = save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


# %% General functions to handle preprocessing (prep) datasets (figures ,(netcdf) files)
def get_daily_ds_date(dataset):
    logger = logging.getLogger()
    try:
        date_64 = dataset.date.values
    except ValueError:
        logger.exception("The dataset does not contain a data variable named 'date'")
        return None
    date_datetime = datetime.utcfromtimestamp(date_64.tolist() / 1e9)
    return date_datetime


def get_prep_dataset_file_name(station, day_date, data_source='molecular', lambda_nm='*', file_type='*'):
    """
     Retrieves file pattern name of preprocessed dataset according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm] or all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or 'all' (meaning the dataset contains several profile types)

    :return: dataset file name (netcdf) file of the data_type required per given day and wavelength, data_source and file_type
    """
    if file_type == '*' or lambda_nm == '*':
        # retrieves any file of this date
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_{file_type}_{lambda_nm}*{data_source}.nc"
    else:
        # this option is mainly to set new file names
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_{file_type}_{lambda_nm}_{data_source}.nc"
    file_name = file_name.replace('all', '').replace('__', '_').replace('__', '_')
    return (file_name)


def get_prep_dataset_paths(station, day_date, data_source='molecular', lambda_nm='*', file_type='*'):
    """
     Retrieves file paths of preprocessed datasets according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm], or 'all' (meaning the dataset contains several profile types)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or 'all' (meaning the dataset contains several profile types)

    :return: paths to all datasets netcdf files of the data_type required per given day and wavelength
    """
    if data_source == 'molecular':
        parent_folder = station.molecular_dataset
    elif data_source == 'lidar':
        parent_folder = station.lidar_dataset

    month_folder = get_month_folder_name(parent_folder, day_date)
    file_name = get_prep_dataset_file_name(station, day_date, data_source, lambda_nm, file_type)

    # print(os.listdir(month_folder))
    file_pattern = os.path.join(month_folder, file_name)

    paths = sorted(glob.glob(file_pattern))

    return paths


def visualize_ds_profile_chan(dataset, lambda_nm=532, profile_type='range_corr', USE_RANGE=None,
                              SAVE_FIG=False, dst_folder=os.path.join('../..', 'Figures'), format_fig='png',
                              dpi=1000):
    logger = logging.getLogger()
    date_datetime = get_daily_ds_date(dataset)
    sub_ds = dataset.sel(Wavelength=lambda_nm).get(profile_type)

    # Currently only a dataset with range_corrected variable, has min/max plot_range values
    USE_RANGE = None if (profile_type != 'range_corr') else USE_RANGE
    if USE_RANGE == 'MID':
        [maxv, minv] = [
            dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_max_range').values.tolist(),
            dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_min_range').values.tolist()]
    elif USE_RANGE == 'LOW':
        try:
            maxv = dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_min_range').values.tolist()
        except:
            logger.debug("The dataset doesn't 'contain plot_min_range', setting maxv=0")
            maxv = 0
        minv = np.nanmin(sub_ds.values)
    elif USE_RANGE == 'HIGH':
        try:
            minv = dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_max_range').values.tolist()
        except:
            logger.debug("The dataset doesn't 'contain plot_min_range', setting maxv=0")
            minv = np.nanmin(sub_ds.values)
        maxv = np.nanmax(sub_ds.values)
    elif USE_RANGE is None:
        [maxv, minv] = [np.nanmax(sub_ds.values), np.nanmin(sub_ds.values)]

    dims = sub_ds.dims
    if 'Time' not in dims:
        logger.error(f"The dataset should have a 'Time' dimension.")
        return None
    if 'Height' in dims:  # plot x- time, y- height
        g = sub_ds.where(sub_ds < maxv).where(sub_ds > minv).plot(x='Time', y='Height', cmap='turbo',
                                                                  figsize=(10, 6))  # ,robust=True)
    elif len(dims) == 2:  # plot x- time, y- other dimension
        g = sub_ds.where(sub_ds < maxv).where(sub_ds > minv).plot(x='Time', cmap='turbo',
                                                                  figsize=(10, 6))
    elif len(dims) == 1:  # plot x- time, y - values in profile type
        g = sub_ds.plot(x='Time', figsize=(10, 6))[0]

    # Set time on x-axis
    xfmt = mdates.DateFormatter('%H:%M')
    g.axes.xaxis.set_major_formatter(xfmt)
    g.axes.xaxis_date()
    g.axes.get_xaxis().set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(g.axes.get_xticklabels(), rotation=0, horizontalalignment='center')

    # Set title description
    date_str = date_datetime.strftime('%d/%m/%Y')
    stitle = f"{sub_ds.attrs['info']} - {lambda_nm}nm \n {dataset.attrs['location']} {date_str}"
    plt.title(stitle, y=1.05)
    plt.tight_layout()
    plt.show()

    if SAVE_FIG:

        fname = f"{date_datetime.strftime('%Y-%m-%d')}_{dataset.attrs['location']}_{profile_type}_" \
                f"{lambda_nm}_source_{dataset.attrs['source_type']}" \
                f"_plot_range_{str(USE_RANGE).lower()}.{format_fig}"

        if not os.path.exists(dst_folder):
            try:
                os.makedirs(dst_folder, exist_ok=True)
                logger.debug(f"Creating folder: {dst_folder}")
            except Exception:
                raise OSError(f"Failed to create folder: {dst_folder}")

        fpath = os.path.join(dst_folder, fname)
        g.figure.savefig(fpath, bbox_inches='tight', format=format_fig, dpi=dpi)

    return g


def get_prep_ds_timestamp(station, path, data_source='molecular'):
    format_times = ["%Y_%m_%d"]
    format_filename = f"(.*)_{station.location}_*_{data_source}.nc"
    date_time = extract_date_time(path, format_filename, format_times)
    return date_time


def gen_daily_ds(day_date):
    logger = logging.getLogger()

    # TODO: Find a way to pass: optim_size, save_mode, USE_KM_UNITS
    #  as variables when running with multiprocessing.
    optim_size = False
    save_mode = 'both'
    USE_KM_UNITS = True

    logger.debug(f"Start generation of molecular dataset for {day_date.strftime('%Y-%m-%d')}")
    station = gs.Station(stations_csv_path='../stations.csv', station_name='haifa')
    # generate molecular dataset
    mol_ds = generate_daily_molecular(station, day_date,
                                      optim_size=optim_size, USE_KM_UNITS=USE_KM_UNITS)

    # save molecular dataset
    ncpaths = save_molecular_dataset(station, mol_ds, save_mode=save_mode)
    logger.debug(f"Done saving molecular datasets for {day_date.strftime('%Y-%m-%d')}, to: {ncpaths}")


def convert_profiles_units(dataset, units=[r'$1/m$', r'$1/km$'], scale=1e+3):
    """
    Converting units of or profiles in dataset,
    :param dataset: lidar dataset or molecular dataset
    :param units: list of strings: [str_source,str_dest], E.g.:
                    For range corrected signal (pr^2) units=['m^2','km^2']
                    For beta (or sigma) converting units= ['1/m','1/km']
    :param scale: float, the scale to use for conversion. E.g.:
                    For range corrected signal, converting distance units or r^2: scale = 10^-6
                    For beta (or sigma), converting the distance units of 1/m: scale = 10^3
    :return: the datasets with units converted profiles
    """
    profiles = [var for var in dataset.data_vars if (len(dataset[var].shape) > 1)]

    for profile in profiles:
        conv_profiles = xr.apply_ufunc(lambda x: x * scale, dataset[profile], keep_attrs=True)
        conv_profiles.attrs["units"] = conv_profiles.units.replace(units[0], units[1])
        dataset[profile] = conv_profiles
    return dataset


# %% MAIN
def main(station_name='haifa', start_date=datetime(2017, 9, 1), end_date=datetime(2017, 9, 2)):
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logger = create_and_configer_logger('preprocessing_log.log', level=logging.INFO)
    CONV_GDAS = False
    GEN_MOL_DS = True
    GEN_LIDAR_DS = False
    USE_KM_UNITS = False

    """set day,location"""
    station = gs.Station(stations_csv_path='../stations.csv', station_name=station_name)
    logger.info(f"Loading {station.location} station")
    logger.debug(f"Station info: {station}")
    logger.info(
        f"Start preprocessing for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Generate molecular datasets for required period'''
    gdas_paths = []
    if CONV_GDAS:
        # Convert gdas files for a period
        gdas_paths.extend(convert_periodic_gdas(station, start_date, end_date))

    # get all days having a converted (to txt) gdas files in the required period
    if (GEN_MOL_DS or GEN_LIDAR_DS) and not gdas_paths:
        logger.info('Get all days in the required period that have a converted gdas file')
        dates = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()
        for day in tqdm(dates):
            _, curpath = get_daily_gdas_paths(station, day, f_type='txt')
            if curpath:
                gdas_paths.extend(curpath)
        timestamps = [get_gdas_timestamp(station, path) for path in gdas_paths]
        df_times = pd.DataFrame(data=timestamps, columns=['timestamp'])
        days_g = df_times.groupby([df_times.timestamp.dt.date]).groups
        valid_gdas_days = list(days_g.keys())

    if GEN_MOL_DS:
        # TODO: Check for existing molecular paths, to avoid creation for them (Since it takes long time to generate these datasest)
        '''molpaths = [ ]
        for day_date in valid_gdas_days:
            cpath = get_prep_dataset_paths ( station = station ,
                                             day_date = day_date ,
                                             lambda_nm = '*',
                                             file_type = 'attbsc')
            if cpath:
                molpaths.extend(cpath)

        timestamps = [
            datetime.strptime ( '_'.join ( (os.path.basename ( cpath )).split ( '_' ) [ 0 :3 ] ) , '%Y_%m_%d' ) for
            cpath in molpaths ]
        df_times = pd.DataFrame ( data = timestamps , columns = [ 'timestamp' ] )
        days_g = df_times.groupby ( [ df_times.timestamp.dt.date ] ).groups
        mol_days = list ( days_g.keys ( ) )
        # separate exiting days from required period (not to genarate again)
        inds_exist = [ valid_gdas_days.index(mol_day)  for mol_day in mol_days ] 
        logger.debug ( f"Existing days of 'attbsc' profiles: {mol_days}" )
'''
        # Generate and save molecular dataset for each valid day in valid_gdas_days :
        logger.info(
            f"Start generating molecular datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        len_mol_days = len(valid_gdas_days)
        num_processes = np.min((cpu_count() - 1, len_mol_days))
        chunksize = np.ceil(np.float(len_mol_days) / num_processes).astype(int)
        # TODO: add here tqdm
        with Pool(num_processes) as p:
            p.map(gen_daily_ds, valid_gdas_days, chunksize=chunksize)

        logger.info(
            f"Finished generating and saving of molecular datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Generate lidar datasets for required period'''
    if GEN_LIDAR_DS:
        lidarpaths = []
        logger.info(
            f"Start generating lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        for day_date in tqdm(valid_gdas_days):
            # Generate daily range corrected
            range_corr_ds = get_daily_range_corr(station, day_date, height_units='km',
                                                 optim_size=False, verbose=False, USE_KM_UNITS=USE_KM_UNITS)

            # Save lidar dataset
            lidar_paths = save_range_corr_dataset(station, range_corr_ds, save_mode='both')
            lidarpaths.extend(lidar_paths)
        logger.info(
            f"Done creation of lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        logger.debug(f'Lidar paths: {lidar_paths}')


if __name__ == '__main__':
    station_name = 'haifa'
    start_date = datetime(2017, 10, 1)
    end_date = datetime(2017, 10, 2)
    main(station_name, start_date, end_date)
