import glob
import logging
import os
import re
from datetime import datetime, timedelta, time, date
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import dates as mdates, pyplot as plt
from molecular import rayleigh_scattering
from pandas.core.dtypes.common import is_numeric_dtype

from learning_lidar.utils import misc_lidar, vis_utils, xr_utils, global_settings as gs
from learning_lidar.utils.utils import write_row_to_csv


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


def cal_e_tau_df(col, height_bins):
    """
    Calculate the the attenuated optical depth (tau is the optical depth).
    Calculation is per column of the backscatter (sigma) dataframe.
    :param col: column of sigma_df , the values of sigma should be in [1/m]
    :param height_bins: np.array of height bins above ground level
    (distances of measurements relative to the sensor), in [m]
    :return: e_tau: pd.Series() of exp(-2*tau), where: the optical depth is tau = integral( sigma(r) * dr)
    """

    tau = misc_lidar.calc_tau(col, height_bins)
    e_tau = pd.Series(np.exp(-2 * tau))
    return e_tau


def calc_beta_profile_df(row, lambda_nm=532.0, ind_n='beta'):
    """
    Returns pd series of backscatter profile from a radiosonde dataframe containing the
    columns:['PRES','TEMPS','RELHS']. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
    :param ind_n: index name, the column name of the result. The default is 'beta'
    but could be useful to get profiles for several hours each have a different index name
    (e.g., datetime.datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd.Series() of backscatter profile [1/sr*m]
    """
    return pd.Series([rayleigh_scattering.beta_pi_rayleigh(wavelength=lambda_nm,
                                                           pressure=row['PRES'],
                                                           temperature=row['TEMPS'],
                                                           C=385.0, rh=row['RELHS'])],
                     index=[ind_n])


def gdas2radiosonde(src_file, dst_file, col_names=None):
    """
    Helper function that converts a gdas file from TROPOS server (saved as .gdas1), to a simple txt.
    The resulting file is without any prior info, and resembles the table format
    of a radiosonde file (see class: RadiosondeProfile).
    :param src_file: source file name
    :param dst_file: destination file name
    :param col_names: column names of the final table
    :return: dst_file: string. The file name of the new radiosonde (converted from gdas).
    """
    logger = logging.getLogger()
    if col_names is None:
        col_names = ['PRES', 'HGHT', 'TEMP', 'UWND', 'VWND', 'WWND', 'RELH', 'TPOT', 'WDIR', 'WSPD']
    try:
        data_src = pd.read_fwf(src_file, skiprows=[0, 1, 2, 3, 4, 5, 6, 8], delimiter="\s+",
                               skipinitialspace=True).dropna()
    except Exception:
        logger.exception(f'\nFailed reading {src_file}. Check the source file, '
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
        logger.exception(f'\nConversion of {src_file} to {dst_file} failed. Check the source file, '
                         'or generate it again with ARLreader module')
        write_row_to_csv('../gdas2radiosonde_failed_files.csv', [src_file, 'Conversion Fail', 'Broken'])
        dst_file = None
    return dst_file


def get_month_folder_name(parent_folder, day_date):
    month_folder = os.path.join(parent_folder, day_date.strftime("%Y"), day_date.strftime("%m"))
    return month_folder


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
    The resulted profile has a grid height above (in 'km' or 'm' - according to input), above sea level.
    start at min_height, end at top_height and extrapolated to have h_bins.
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

    heights = station.calc_height_index(USE_KM_UNITS=(height_units == 'km'))

    df_sigma = pd.DataFrame(index=heights).rename_axis(f'Height[{height_units}]')
    df_beta = pd.DataFrame(index=heights).rename_axis(f'Height[{height_units}]')

    for path, timestamp in zip(gdas_txt_paths, timestamps):
        df_sonde = misc_lidar.RadiosondeProfile(path).get_df_sonde(heights)
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply(calc_sigma_profile_df, axis=1, args=(lambda_nm, timestamp,),
                             result_type='expand').astype('float64')
        df_sigma[res.columns] = res
        res = df_sonde.apply(calc_beta_profile_df, axis=1, args=(lambda_nm, timestamp,),
                             result_type='expand').astype('float64')
        df_beta[res.columns] = res

    return df_sigma, df_beta


def get_gdas_timestamp(station, path, file_type='txt'):
    format_times = ["%Y%m%d_%H"]
    format_filename = f"{station.location.lower()}_(.*)_{station.lat:.1f}_{station.lon:.1f}.{file_type}"
    date_time = extract_date_time(path, format_filename, format_times)[0]
    return date_time


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
    # TODO if gdas1 file does not exists - download from NOA
    if f_type == 'gdas1':
        parent_folder = station.gdas1_folder
    elif f_type == 'txt':
        parent_folder = station.gdastxt_folder
    month_folder = get_month_folder_name(parent_folder, day_date)
    if not os.path.exists(month_folder):
        try:
            os.makedirs(month_folder)
        except:
            logger.exception(f"\nFailed to create folder: {month_folder}")

    gdas_day_pattern = '{}_{}_*_{:.1f}_{:.1f}.{}'.format(station.location.lower(), day_date.strftime('%Y%m%d'),
                                                         station.lat, station.lon, f_type)
    path_pattern = os.path.join(month_folder, gdas_day_pattern)
    gdas_paths = sorted(glob.glob(path_pattern))
    return month_folder, gdas_paths


def convert_daily_gdas(station, day_date):
    """
    Converting gdas files from TROPOS of type .gdas1 to .txt for a given day.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the date to be converted
    :return: converted_paths: list of strings. The file names of the converted gdas per day.
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


def visualize_ds_profile_chan(dataset, lambda_nm=532, profile_type='range_corr', USE_RANGE=None,
                              SAVE_FIG=False, dst_folder=os.path.join('../../..', 'Figures'), format_fig='png',
                              dpi=1000):
    logger = logging.getLogger()
    date_datetime = xr_utils.get_daily_ds_date(dataset)
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
            logger.debug("\nThe dataset doesn't 'contain plot_min_range', setting maxv=0")
            maxv = 0
        minv = np.nanmin(sub_ds.values)
    elif USE_RANGE == 'HIGH':
        try:
            minv = dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_max_range').values.tolist()
        except:
            logger.debug("\nThe dataset doesn't 'contain plot_min_range', setting maxv=0")
            minv = np.nanmin(sub_ds.values)
        maxv = np.nanmax(sub_ds.values)
    elif USE_RANGE is None:
        [maxv, minv] = [np.nanmax(sub_ds.values), np.nanmin(sub_ds.values)]

    dims = sub_ds.dims
    if 'Time' not in dims:
        logger.error(f"\nThe dataset should have a 'Time' dimension.")
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
    g.axes.xaxis.set_major_formatter(vis_utils.TIMEFORMAT)
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
                logger.debug(f"\nCreating folder: {dst_folder}")
            except Exception:
                raise OSError(f"\nFailed to create folder: {dst_folder}")

        fpath = os.path.join(dst_folder, fname)
        g.figure.savefig(fpath, bbox_inches='tight', format=format_fig, dpi=dpi)

    return g


def get_TROPOS_dataset_file_name(start_time=None, end_time=None, file_type='profiles'):
    """
    Retrieves file pattern name of TROPOS start time, end time and  profile type.
    :param start_time: datetime.datetime object specifying specific start time of measurement or profile analysis
    :param end_time: datetime.datetime object specifying specific end time of profile analysis
    :param file_type: type of data stored in the files,
    E.g.: 'profiles', 'att_bsc', 'overlap', 'OC_att_bsc', 'cloudinfo', etc.
    E.g.: for analyzed profiles "*_<start_time>_<end_time>_<file_type>.nc"
    ( or  "*<file_type>.nc" if start_time = None, end_time = None)
          for daily attenuation backscatter profile "*<att_bsc>.nc"
          ( or  "*<start_time>_<file_type>.nc" if start_time is given)
          for daily lidar raw signal - None # TODO adi correct me please
    """
    if file_type is None:
        pattern = f"*[0-9].nc"
    elif start_time and end_time and ('profiles' in file_type):
        pattern = f"*[0-9]_{start_time.strftime('%H%M')}_{end_time.strftime('%H%M')}_{file_type}.nc"
    elif start_time:
        pattern = f"*{start_time.strftime('%H_%M_%S')}_{file_type}.nc"
    else:
        pattern = f"*[0-9]_{file_type}.nc"
    return pattern


def get_TROPOS_dataset_paths(station, day_date, start_time=None, end_time=None, file_type='profiles', level='level0'):
    """
    Retrieves netcdf (.nc) files from TROPOS for a given station and day_date, and type.

    :param level: str, should be 'level0' or 'level1a' according to tropos data.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param start_time: datetime.datetime object specifying specific start time of measurement or profile analysis
    :param end_time: datetime.datetime object specifying specific end time of profile analysis
    :param file_type: type of data stored in the files,
    E.g.: 'profiles', 'att_bsc', 'overlap', 'OC_att_bsc', 'cloudinfo', etc.
    E.g.: for analyzed profiles "*_<start_time>_<end_time>_<file_type>.nc"
    (or  "*<file_type>.nc" if start_time = None, end_time = None)
          for daily attenuation backscatter profile "*<att_bsc>.nc" (
           or  "*<start_time>_<file_type>.nc" if start_time is given)
    :return: paths: paths to all *<file_type>.nc for a given station and day_date
    """
    if level == 'level0':
        parent_folder = station.lidar_src_folder
    elif level == 'level1a':
        parent_folder = station.lidar_src_calib_folder
    lidar_day_folder = get_TROPOS_day_folder_name(parent_folder=parent_folder, day_date=day_date)
    file_name = get_TROPOS_dataset_file_name(start_time, end_time, file_type)
    paths_pattern = os.path.join(lidar_day_folder, file_name)

    paths = sorted(glob.glob(paths_pattern))
    return paths


def get_TROPOS_day_folder_name(parent_folder, day_date):
    moth_folder = get_month_folder_name(parent_folder, day_date)
    day_folder = os.path.join(moth_folder, day_date.strftime("%d"))
    return day_folder


def get_range_corr_ds_chan(darray, altitude, lambda_nm, height_units='km', optim_size=False,
                           verbose=False):
    """
    Retrieving a 6-hours range corrected lidar signal (pr^2)
    from attenuated_backscatter signals in three channels (355,532,1064).
    The attenuated_backscatter are from a 6-hours *att_bsc.nc loaded earlier by darray

    :param darray: is xarray.DataArray object, containing a 6-hours of attenuated_backscatter
                    (loaded from TROPOS *att_bsc.nc)
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
    LC = darray.attrs['Lidar_calibration_constant_used']
    times = pd.to_datetime(
        [datetime.utcfromtimestamp(np.round(vtime)) for vtime in darray.time.values]).values
    if height_units == 'km':
        scale = 1e-3
    elif height_units == 'm':
        scale = 1
    heights_ind = scale * (darray.height.values + altitude)
    rangecorr_df = pd.DataFrame(LC * darray.values, index=heights_ind, columns=times)

    range_corr_ds_chan = create_range_corr_ds_chan(rangecorr_df=rangecorr_df, lambda_nm=lambda_nm,
                                                   height_units=height_units, optim_size=optim_size, verbose=verbose)

    return range_corr_ds_chan, LC


def create_range_corr_ds_chan(rangecorr_df: pd.DataFrame, lambda_nm: int, height_units: str,
                              optim_size: bool = False, verbose: bool = False) -> xr.Dataset:
    """
    Retrieving a 6-hours range corrected lidar signal (pr^2)
    from attenuated_backscatter signals in three channels (355,532,1064).
    The attenuated_backscatter are from a 6-hours *att_bsc.nc loaded earlier by darray

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
    # memory size - optimization
    if optim_size:
        if verbose:
            logger.debug('\nMemory optimization - converting molecular values from double to float')
            size_rangecorr = rangecorr_df.memory_usage(deep=True).sum()

        rangecorr_df = (rangecorr_df.select_dtypes(include=['float64'])). \
            apply(pd.to_numeric, downcast='float')

        if verbose:
            size_rangecorr_opt = rangecorr_df.memory_usage(deep=True).sum()
            logger.debug('\nMemory saved for wavelength {} range corrected: {:.2f}%'.
                         format(lambda_nm, 100.0 * float(size_rangecorr - size_rangecorr_opt) / float(size_rangecorr)))

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
                                           'info': 'Range corrected lidar signal from attenuated backscatter '
                                                   'multiplied by LC'}
    # set attributes of coordinates
    range_corr_ds_chan.Height.attrs = {'units': fr'${height_units}$',
                                       'info': 'Measurements heights above sea level'}
    range_corr_ds_chan.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}

    return range_corr_ds_chan


def generate_daily_molecular_chan(station, day_date, lambda_nm, time_res='30S',
                                  height_units='km', optim_size=False, verbose=False):
    """
    Generating daily molecular profiles for a given channel's wavelength
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
    :param time_res: Output time resolution required. default=30sec (according to pollyXT measurements time resolution)
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

    # Load daily gdas profiles and convert to backscatter (beta) and extinction (sigma) profiles

    df_sigma, df_beta = get_daily_molecular_profiles(station, day_date, lambda_nm, height_units)

    # Interpolate profiles through 24 hrs
    interp_sigma_df = (df_sigma.T.resample(time_res).interpolate(method='linear')[:-1]).T
    interp_beta_df = (df_beta.T.resample(time_res).interpolate(method='linear')[:-1]).T
    interp_sigma_df.columns.freq = None
    interp_beta_df.columns.freq = None

    '''Calculate the molecular attenuated backscatter as :  beta_mol * exp(-2*tau_mol)'''
    height_bins = station.get_height_bins_values(USE_KM_UNITS=False)
    e_tau_df = interp_sigma_df.apply(cal_e_tau_df, 0, args=(height_bins,), result_type='expand')

    att_bsc_mol_df = interp_beta_df.multiply(e_tau_df)

    ''' memory size - optimization '''
    if optim_size:
        if verbose:
            logger.debug('\nMemory optimization - converting molecular values from double to float')
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
            logger.debug('\nMemory saved for wavelength {} beta: {:.2f}%, sigma: {:.2f}%, att_bsc:{:.2f}%'.
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


def calc_r2_ds(station, day_date):
    """
    calc r^2 (as 2D image)
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: xr.Dataset(). A a daily r^2 dataset
    """
    # TODO add USE_KM_UNITS flag and units is km if  USE_KM_UNITS else m
    height_bins = station.get_height_bins_values()
    wavelengths = gs.LAMBDA_nm().get_elastic()
    r_im = np.tile(height_bins.reshape(height_bins.size, 1), (len(wavelengths), 1, station.total_time_bins))
    rr_im = r_im ** 2
    r2_ds = xr.Dataset(data_vars={'r': (['Wavelength', 'Height', 'Time'], r_im,
                                        {'info': 'The heights bins',
                                         'name': 'r', 'long_name': r'$r$',
                                         'units': r'$km$'}),
                                  'r2': (['Wavelength', 'Height', 'Time'], rr_im,
                                         {'info': 'The heights bins squared',
                                          'name': 'r2', 'long_name': r'$r^2$',
                                          'units': r'$km^2$'})},
                       coords={'Wavelength': wavelengths,
                               'Height': station.calc_height_index(),
                               'Time': station.calc_daily_time_index(day_date).values})
    r2_ds = r2_ds.transpose('Wavelength', 'Height', 'Time')
    return r2_ds.r2


def convert_periodic_gdas(station, start_day, end_day):
    logger = logging.getLogger()

    day_dates = pd.date_range(start=start_day, end=end_day, freq=timedelta(days=1))
    expected_file_no = len(day_dates) * 8  # 8 timestamps per day
    gdastxt_paths = []
    for day in day_dates:
        gdastxt_paths.extend(convert_daily_gdas(station, day))
    total_converted = len(gdastxt_paths)
    logger.debug(f"\nDone conversion of {total_converted} gdas files for period [{start_day.strftime('%Y-%m-%d')},"
                 f"{end_day.strftime('%Y-%m-%d')}], {(expected_file_no - total_converted)} failed.")
    return gdastxt_paths


def generate_daily_molecular(station, day_date, time_res='30S', height_units='km',
                             optim_size=False, verbose=False, USE_KM_UNITS=True):
    """
    Generating daily molecular profiles for all elastic channels (355,532,1064)
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param time_res: Output time resolution required. default=30sec (according to pollyXT measurements time resolution)
    :param height_units:  Output units of height grid in 'km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
                                True: the retrieved values are of type 'float'.
    :param verbose: Boolean. False(default). True: prints information regarding size optimization.
    :param USE_KM_UNITS: Boolean flag, to set the scale of units of the output data ,True - km units, False - meter units
    :return: xarray.Dataset() holding 5 data variables:
             3 daily dataframes: beta,sigma,att_bsc with shared dimensions(Height, Time, Wavelength)
             and 2 shared variables: lambda_nm with dimension (Wavelength), and date
    """

    date_datetime = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date

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


def gen_daily_molecular_ds(day_date):
    """
    Generating and saving a daily molecular profile.
    The profile is of type xr.Dataset().
    Having 3 variables: sigma (extinction) ,beta(backscatter) and attbsc(beta*exp(-2tau).
    Each profile have dimensions of: Wavelength, Height, Time.
    :param day_date: datetime.date object of the required day
    :return:
    """
    # TODO: Find a way to pass: optim_size, save_mode, USE_KM_UNITS
    #  as variables when running with multiprocessing. -
    #  look at p.starmap(generate_daily_aerosol_density, zip(repeat(station), days_list)) for example
    logger = logging.getLogger()
    optim_size = False
    save_mode = 'single'
    USE_KM_UNITS = True

    logger.debug(f"\nStart generation of molecular dataset for {day_date.strftime('%Y-%m-%d')}")
    station = gs.Station(station_name='haifa')
    # generate molecular dataset
    mol_ds = generate_daily_molecular(station, day_date,
                                      optim_size=optim_size, USE_KM_UNITS=USE_KM_UNITS)

    # save molecular dataset
    ncpaths = xr_utils.save_prep_dataset(station, mol_ds, data_source='molecular', save_mode=save_mode, profiles=['attbsc'])
    logger.debug(f"\nDone saving molecular datasets for {day_date.strftime('%Y-%m-%d')}, to: {ncpaths}")


def get_daily_range_corr(station, day_date, height_units='km',
                         optim_size=False, verbose=False, use_km_unit=True):
    """
    Retrieving daily range corrected lidar signal (pr^2)
    from attenuated_backscatter signals in three channels (355,532,1064).

    The attenuated_backscatter are from 4 files of 6-hours *att_bsc.nc for a given day_date and station

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param height_units:  Output units of height grid in 'km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
                                True: the retrieved values are of type 'float'.
    :param verbose: Boolean. False(default). True: prints information regarding size optimization.
    :param use_km_unit: Boolean flag, to set the scale of units of the output data ,True - km units, False - meter units
    :return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
             1 daily dataset of range_corrected signal in 3 channels, with dimensions of : Height, Time, Wavelength
             3 variables : lambda_nm, plot_min_range, plot_max_range, with dimension of : Wavelength
             1 shared variable: date
    """

    """ get netcdf paths of the attenuation backscatter for given day_date"""
    date_datetime = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date

    bsc_paths = get_TROPOS_dataset_paths(station, date_datetime, file_type='att_bsc', level='level1a')
    bsc_ds0 = xr_utils.load_dataset(bsc_paths[0])
    altitude = bsc_ds0.altitude.values[0]
    profiles = [dvar for dvar in list(bsc_ds0.data_vars) if 'attenuated_backscatter' in dvar]
    wavelengths = [np.uint(pname.split(sep='_')[-1].strip('nm')) for pname in profiles]

    min_range = np.empty((len(wavelengths), len(bsc_paths)))
    max_range = np.empty((len(wavelengths), len(bsc_paths)))

    ds_range_corrs = []
    for ind_path, bsc_path in enumerate(bsc_paths):
        cur_ds = xr_utils.load_dataset(bsc_path)
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
    time_indx = station.calc_daily_time_index(date_datetime)
    range_corr_ds = range_corr_ds.reindex({"Time": time_indx}, fill_value=0)
    range_corr_ds = range_corr_ds.assign({'plot_min_range': ('Wavelength', min_range.min(axis=1)),
                                          'plot_max_range': ('Wavelength', max_range.max(axis=1))})
    range_corr_ds['date'] = date_datetime
    range_corr_ds.attrs = {'location': station.location,
                           'info': 'Daily range corrected lidar signal',
                           'source_type': 'att_bsc'}
    if use_km_unit:
        range_corr_ds = convert_profiles_units(range_corr_ds, units=[r'$m^2$', r'$km^2$'], scale=1e-6)
    return range_corr_ds


def get_raw_lidar_signal(station: gs.Station, day_date: datetime, height_slice: slice, ds_attr: dict,
                         use_km_units: bool) -> xr.Dataset:
    """
        Retrieving daily raw lidar signal (p / bg) from attenuated_backscatter signals in three channels
     (355,532,1064).

    The attenuated_backscatter are from 4 files of 6-hours *.nc for a given day_date and station

    Height slice determines if it is background - slice(0, station.pt_bin) or
     p - slice(station.pt_bin, station.pt_bin + station.n_bins)

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param height_slice: slice object deterining the heights to keep
    :param ds_attr: dict, the attributes of the dataset
    :param use_km_units: datetime.date object of the required date
    :return: xarray.Dataset() a daily raw lidar signal, holding 5 data variables:
             1 daily dataset of background or raw lidar signal in 3 channels,
             with dimensions of : Height, Time, Wavelength
             1 variables : lambda_nm, with dimension of : Wavelength
             1 shared variable: date
    """
    raw_paths = get_TROPOS_dataset_paths(station, day_date, file_type=None, level='level0')

    profile = 'raw_signal'
    num_times = int(station.total_time_bins / 4)
    channel_ids = gs.CHANNELS().get_elastic()
    wavelengths = gs.LAMBDA_nm().get_elastic()
    all_times = station.calc_daily_time_index(day_date)
    heights_ind = station.calc_height_index(USE_KM_UNITS=use_km_units)

    dss = []
    for part_of_day_indx, bsc_path in enumerate(raw_paths):
        cur_ds = xr_utils.load_dataset(bsc_path)
        # get 6-hours range corrected dataset for three channels [355,532,1064]
        cur_darry = cur_ds.get(profile).transpose(transpose_coords=True)
        times = list(all_times)[num_times * part_of_day_indx:num_times * (part_of_day_indx + 1)]

        darray = cur_darry.sel(channel=channel_ids, height=height_slice)
        ''' Create p dataset'''
        ds_partial = xr.Dataset(
            data_vars={'p': (('Wavelength', 'Height', 'Time'), darray.values)},
            coords={'Height': heights_ind[:height_slice.stop - height_slice.start],
                    'Time': times,
                    'Wavelength': wavelengths})

        dss.append(ds_partial)

    # merge range corrected of lidar through 24-hours
    ds = xr.merge(dss, compat='no_conflicts')

    # Fixing missing timestamps values:
    ds = ds.reindex({"Time": all_times}, fill_value=0)

    ds.p.attrs = ds_attr
    ds.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    ds.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}

    ds['date'] = day_date

    return ds


def get_daily_measurements(station: gs.Station, day_date: Union[datetime.date, datetime], use_km_units: bool = True) \
        -> xr.Dataset:
    """
    Retrieving daily range corrected lidar signal (pr^2), background and raw lidar signal
     from attenuated_backscatter signals in three channels (355,532,1064).

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param use_km_units: whether to use km units or not. If False  uses 'm'
    :return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
             3 daily datasets of range_corrected signal, background and raw lidar signal in 3 channels,
             with dimensions of : Height, Time, Wavelength
             1 variables : lambda_nm, with dimension of : Wavelength
             1 shared variable: date
    """
    day_date = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date
    # Raw Lidar Signal Dataset
    pn_ds_attr = {'info': 'Raw Lidar Signal',
                  'long_name': r'$p$', 'name': 'pn',
                  'units': r'$\rm$' + r'$photons$',
                  'location': station.location, }

    # TODO: the Hight index is wrong. pn_ds Height should start at 0.3 km not 2.3...
    pn_ds = get_raw_lidar_signal(station=station,
                                 day_date=day_date,
                                 height_slice=slice(station.pt_bin, station.pt_bin + station.n_bins),
                                 ds_attr=pn_ds_attr,
                                 use_km_units=use_km_units)

    # Raw Background Measurement Dataset
    bg_ds_attr = {'info': 'Raw Background Measurement',
                  'long_name': r'$<p_{bg}>$',
                  'units': r'$\rm photons$',
                  'name': 'pbg'}
    bg_ds = get_raw_lidar_signal(station=station,
                                 day_date=day_date,
                                 height_slice=slice(0, station.pt_bin),
                                 ds_attr=bg_ds_attr,
                                 use_km_units=use_km_units)

    bg_mean = bg_ds.mean(dim='Height', keep_attrs=True)
    p_bg = bg_mean.p.broadcast_like(pn_ds.p)

    # Raw Range Corrected Lidar Signal
    r2_ds = calc_r2_ds(station, day_date)
    pr2n = (pn_ds.p.copy(deep=True) * r2_ds)  # calc_range_corr_measurement #
    # TODO add assert np.sum(r2_ds.Height.values- pn_ds.p.Height.values)==0
    #  assert np.sum(r2_ds.Time.values- pn_ds.p.Time.values) == np.timedelta64(0,'ns')
    # TODO: The multiplication above returns empty array if indices are not the same!
    #  The Height index was (of pr2n_ds) for some reason in meters where for r2_ds it was in km .
    #  so when doing so this should be check before and stop if there is no consistency.

    pr2n.attrs = {'info': 'Raw Range Corrected Lidar Signal',
                  'long_name': r'$\rm p$' + r'$\cdot r^2$', 'name': 'range_corr',
                  'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$',
                  'location': station.location, }
    pr2n.Height.attrs = {'units': r'$\rm km$', 'info': 'Measurements heights above sea level'}
    pr2n.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}

    # Daily raw lidar measurement from TROPOS.
    lidar_ds = xr.Dataset().assign(p=pn_ds.p, range_corr=pr2n, p_bg=p_bg)
    lidar_ds['date'] = day_date
    lidar_ds.attrs = {'location': station.location,
                      'info': 'Daily raw lidar measurement from TROPOS.',
                      'source_file': os.path.basename(__file__)}

    return lidar_ds
