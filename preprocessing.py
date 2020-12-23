import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import os
from datetime import datetime , timedelta
import glob
from molecular import rayleigh_scattering
import numpy as np
from generate_atmosphere import RadiosondeProfile
import re
from netCDF4 import Dataset
import sqlite3
import fnmatch
import matplotlib.pyplot as plt
import global_settings as gs
import miscLidar as mscLid
from utils import create_and_configer_logger , write_row_to_csv
import logging
import torch.utils.data
import xarray as xr
import matplotlib.dates as mdates


# %% General functions


def get_month_folder_name ( parent_folder , day_date ) :
    month_folder = os.path.join ( parent_folder , day_date.strftime ( "%Y" ) , day_date.strftime ( "%m" ) )
    return (month_folder)


def get_day_folder_name ( parent_folder , day_date ) :
    moth_folder = get_month_folder_name ( parent_folder , day_date )
    day_folder = os.path.join ( moth_folder , day_date.strftime ( "%d" ) )
    return day_folder


def extract_date_time ( path , format_filename , format_times ) :
    """
    Extracting datetime from file name using a formatter string.
    :param path: path to folder containing the files to observe
    :param format_filename:  a formatter string
    :param format_times: format of datetime in the file name
    :return: time_stamps - A list of datetime.datetime objects
    usage:  create a formatter string: format_filename=  r'-(.*)_(.*)-(.*)-.*.txt'
            Call the function: time_stamp = extract_date_time(soundePath,r'40179_(.*).txt',['%Y%m%d_%H'])

    """
    filename = os.path.basename ( path )
    matchObj = re.match ( format_filename , filename , re.M | re.I )
    time_stamps = [ ]
    for fmt_time , grp in zip ( format_times , matchObj.groups ( ) ) :
        time_stamps.append ( datetime.strptime ( grp , fmt_time ) )
    return time_stamps


def save_dataset ( dataset , folder_name , nc_name ) :
    """
    Save the input dataset to netcdf file
    :param dataset: array.Dataset()
    :param folder_name: folder name
    :param nc_name: netcdf file name
    :return: ncpath - full path to netcdf file created if succeeded, else none
    """
    logger = logging.getLogger ( )
    if not os.path.exists ( folder_name ) :
        try :
            os.makedirs ( folder_name )
            logger.debug ( f"Creating folder: {folder_name}" )
        except Exception :
            logger.exception ( f"Failed to create folder: {folder_name}" )
            return None

    ncpath = os.path.join ( folder_name , nc_name )
    try :
        dataset.to_netcdf ( ncpath , mode = 'w' , format = 'NETCDF4' , engine = 'netcdf4' )
        dataset.close ( )
        logger.debug ( f"Saving dataset file: {ncpath}" )
    except Exception :
        logger.exception ( f"Failed to save dataset file: {ncpath}" )
        ncpath = None
    return ncpath


def load_dataset ( ncpath ) :
    """
    Load Dataset stored in the netcdf file path (ncpath)
	:param ncpath: a netcdf file path
	:return: xarray.Dataset, if fails return none
	"""
    logger = logging.getLogger ( )
    try :
        dataset = xr.open_dataset ( ncpath , engine = 'netcdf4' ).expand_dims ( )
        dataset.close ( )
        logger.debug ( f"Loading dataset file: {ncpath}" )
    except Exception :
        logger.exception ( f"Failed to load dataset file: {ncpath}" )
        return None
    return dataset


# %% Functions to handle TROPOS datasets (netcdf) files


def get_daily_molecular_profiles ( station , day_date , lambda_nm = 532 , height_units = 'Km' ) :
    """
    Generating daily molecular profile from gdas txt file
    :param station: gs.station() object of the lidar station
    :param gdas_txt_paths: paths to gdas txt files , containing table with the columns
    "PRES	HGHT	TEMP	UWND	VWND	WWND	RELH	TPOT	WDIR	WSPD"
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm]
    :param height_units: units of height grid in 'Km' (default) or 'm'
    this parameter gives dr= heights[1]-heights[0] =~ 7.5e-3[km] -> the height resolution of pollyXT lidar)
    :return: Returns backscatter and extinction profiles as pandas-dataframes, according to Rayleigh scattering.
    The resulted profile has a grid height above (in 'Km' or 'm' - according to input), above sea level. start at min_height, end at top_height and extrapolated to have h_bins.
    """
    logger = logging.getLogger ( )
    if height_units == 'Km' :
        scale = 1E-3
    elif height_units == 'm' :
        scale = 1

    # setting height vector above sea level (for interpolation of radiosonde / gdas files).
    min_height = station.altitude + station.start_bin_height
    top_height = station.altitude + station.end_bin_height
    heights = np.linspace ( min_height * scale , top_height * scale , station.n_bins )
    # uncomment the commands below for sanity check of the desired height resolution
    # dr = heights[1]-heights[0]
    # print('h_bins=',heights.shape,' min_height=', min_height,' top_height=', top_height,' dr=',dr )

    df_sigma = pd.DataFrame ( index = heights ).rename_axis ( 'Height[{}]'.format ( height_units ) )
    df_beta = pd.DataFrame ( index = heights ).rename_axis ( 'Height[{}]'.format ( height_units ) )

    _ , gdas_curday_paths = get_daily_gdas_paths ( station , day_date , 'txt' )
    if not gdas_curday_paths :
        logger.debug ( f"For {day_date.strftime ( '%Y/%m/%d' )}, "
                       f"there are not the required GDAS '.txt' files. Starting conversion from '.gdas1'" )
        gdas_curday_paths = convert_daily_gdas ( station , day_date )

    next_day = day_date + timedelta ( days = 1 )
    _ , gdas_nxtday_paths = get_daily_gdas_paths ( station , next_day , 'txt' )
    if not gdas_nxtday_paths :
        logger.debug ( f"For {day_date.strftime ( '%Y/%m/%d' )}, "
                       f"there are not the required GDAS '.txt' files. Starting conversion from '.gdas1'" )
        gdas_nxtday_paths = convert_daily_gdas ( station , next_day )

    gdas_txt_paths = gdas_curday_paths
    gdas_txt_paths.append ( gdas_nxtday_paths [ 0 ] )

    for path in gdas_txt_paths :
        df_sonde = RadiosondeProfile ( path ).get_df_sonde ( heights )
        time = extract_date_time ( path ,
                                   r'{}_(.*)_{:.1f}_{:.1f}.txt'.format ( station.location.lower ( ) , station.lat ,
                                                                         station.lon ) ,
                                   [ '%Y%m%d_%H' ] ) [ 0 ]
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply ( calc_sigma_profile_df , axis = 1 , args = (lambda_nm , time ,) ,
                               result_type = 'expand' ).astype ( 'float64' )
        df_sigma [ res.columns ] = res
        res = df_sonde.apply ( calc_beta_profile_df , axis = 1 , args = (lambda_nm , time ,) ,
                               result_type = 'expand' ).astype ( 'float64' )
        df_beta [ res.columns ] = res

    return df_sigma , df_beta


def get_TROPOS_dataset_file_name(start_time = None, end_time = None, file_type = 'profiles' ):
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

def get_TROPOS_dataset_paths ( station , day_date , start_time = None, end_time = None, file_type = 'profiles' ) :
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
    lidar_day_folder = get_day_folder_name ( station.lidar_src_folder , day_date )
    file_name = get_TROPOS_dataset_file_name ( start_time , end_time , file_type )
    os.listdir(lidar_day_folder)
    paths_pattern = os.path.join ( lidar_day_folder , file_name )

    paths = sorted ( glob.glob ( paths_pattern ) )

    return paths


def extract_att_bsc ( bsc_paths , wavelengths ) :
    """
    For all .nc files under bsc_paths and for each wavelength in wavelengths
    extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used

    :param bsc_paths: path to netcdf folder
    :param wavelengths: iterable, list of wavelengths
    :return:
    """
    logger = logging.getLogger ( )
    for wavelength in wavelengths :
        for bsc_path in bsc_paths :
            data = Dataset ( bsc_path )
            file_name = bsc_path.split ( '/' ) [ -1 ]
            try :
                vals = data.variables [ f'OC_attenuated_backscatter_{wavelength}nm' ]
                arr = vals * vals.Lidar_calibration_constant_used
                logger.debug ( f"Extracted OC_attenuated_backscatter_{wavelength}nm from {file_name}" )
            except KeyError as e :
                logger.exception ( f"Key {e} does not exist in {file_name}" , exc_info = False )


# %% GDAS files preprocessing functions


def gdas2radiosonde ( src_file , dst_file , col_names = None ) :
    """
    Helper function that converts a gdas file from TROPOS server (saved as .gdas1), to a simple txt.
    The resulting file is without any prior info, and resembles the table format
    of a radiosonde file (see class: RadiosondeProfile).
    :param src_file: source file name
    :param dst_file: destination file name
    :param col_names: column names of the final table
    :return:
    """
    logger = logging.getLogger ( )
    if col_names is None :
        col_names = [ 'PRES' , 'HGHT' , 'TEMP' , 'UWND' , 'VWND' , 'WWND' , 'RELH' , 'TPOT' , 'WDIR' , 'WSPD' ]
    try :
        data_src = pd.read_fwf ( src_file , skiprows = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 8 ] , delimiter = "\s+" ,
                                 skipinitialspace = True ).dropna ( )
    except Exception :
        logger.exception ( f'Failed reading {src_file}. Check the source file, '
                           'or generate it again with ARLreader module' )
        write_row_to_csv ( 'gdas2radiosonde_failed_files.csv' , [ src_file , 'Read Fail' , 'Broken' ] )
        return None

    # converting any kind of blank spaces to zeros
    try :
        for col in data_src.columns :
            if not is_numeric_dtype ( data_src [ col ] ) :
                data_src [ col ] = pd.core.strings.str_strip ( data_src [ col ] )
                data_src [ col ] = data_src [ col ].replace ( '' , '0' ).astype ( 'float64' )
        data_src.columns = col_names
        data_src.to_csv ( dst_file , index = False , sep = '\t' , na_rep = '\t' )
    except Exception :
        logger.exception ( f'Conversion of {src_file} to {dst_file} failed. Check the source file, '
                           'or generate it again with ARLreader module' )
        write_row_to_csv ( 'gdas2radiosonde_failed_files.csv' , [ src_file , 'Conversion Fail' , 'Broken' ] )
        dst_file = None
    return dst_file


def get_daily_gdas_paths ( station , day_date , f_type = 'gdas1' ) :
    """
    Retrieves gdas container folder and file paths for a given date
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param f_type: 'gdas1' - for the original gdas files (from TROPOS), 'txt' - for converted (fixed) gdas files
    :return: gdas_folder, gdas_paths - the folder containing the gdas files and the file paths.
    NOTE: during a daily conversion, the function creates a NEW gdas_folder (for the converted txt files),
    and returns an EMPTY list of gdas_paths. The creation of NEW gdas_paths is done in convert_daily_gdas()
    """
    logger = logging.getLogger ( )

    if f_type == 'gdas1' :
        parent_folder = station.gdas1_folder
    elif f_type == 'txt' :
        parent_folder = station.gdastxt_folder
    month_folder = get_month_folder_name ( parent_folder , day_date )
    if not os.path.exists ( month_folder ) :
        try :
            os.makedirs ( month_folder )
        except :
            logger.exception ( f"Failed to create folder: {month_folder}" )

    gdas_day_pattern = '{}_{}_*_{:.1f}_{:.1f}.{}'.format ( station.location.lower ( ) , day_date.strftime ( '%Y%m%d' ) ,
                                                           station.lat , station.lon , f_type )
    path_pattern = os.path.join ( month_folder , gdas_day_pattern )
    gdas_paths = sorted ( glob.glob ( path_pattern ) )
    return month_folder , gdas_paths


def get_gdas_file_name ( station , time , f_type = 'txt' ) :
    file_name = '{}_{}_{}_{:.1f}_{:.1f}.{}'.format ( station.location.lower ( ) , time.strftime ( '%Y%m%d' ) ,
                                                     time.strftime ( '%H' ) ,
                                                     station.lat , station.lon , f_type )

    return (file_name)


def convert_daily_gdas ( station , day_date ) :
    """
    Converting gdas files from TROPOS of type .gdas1 to .txt for a given day.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the date to be converted
    :return:
    """
    # get source container folder and file paths (.gdas1)
    src_folder , gdas1_paths = get_daily_gdas_paths ( station , day_date , 'gdas1' )

    # set dest container folder and file paths (.txt)
    dst_folder , _ = get_daily_gdas_paths ( station , day_date , 'txt' )
    path_to_convert = [ sub.replace ( src_folder , dst_folder ).replace ( 'gdas1' , 'txt' ) for sub in gdas1_paths ]
    converted_paths = [ ]
    # convert each src_file (as .gdas1) to dst_file (as .txt)
    for (src_file , dst_file) in zip ( gdas1_paths , path_to_convert ) :
        converted = gdas2radiosonde ( src_file , dst_file )
        if converted :
            converted_paths.append ( converted )

    return converted_paths


def convert_periodic_gdas ( station , start_day , end_day ) :
    logger = logging.getLogger ( )

    day_dates = pd.date_range ( start = start_day , end = end_day , freq = timedelta ( days = 1 ) )
    expected_file_no = len ( day_dates ) * 8  # 8 timestamps per day
    gdastxt_paths = [ ]
    for day in day_dates :
        gdastxt_paths.extend ( convert_daily_gdas ( station , day ) )
    total_converted = len ( gdastxt_paths )
    logger.debug ( f"Done conversion of {total_converted} gdas files from {start_day.strftime ( '%Y/%m/%d' )} to "
                   f"{end_day.strftime ( '%Y/%m/%d' )}, {(expected_file_no - total_converted)} failed." )
    return gdastxt_paths


# %% Molecular preprocessing and dataset functions


def calc_sigma_profile_df ( row , lambda_nm = 532.0 , indx_n = 'sigma' ) :
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
    return pd.Series (
        data = [ rayleigh_scattering.alpha_rayleigh ( wavelength = lambda_nm , pressure = row [ 'PRES' ] ,
                                                      temperature = row [ 'TEMPS' ] , C = 385.0 ,
                                                      rh = row [ 'RELHS' ] ) ] ,
        index = [ indx_n ] )


def cal_e_tau_df ( col , altitude ) :
    """
    Calculate the the attenuated optical depth (tau is the optical depth).
    Calculation is per column of the backscatter (sigma) dataframe.
    :param col: column of sigma_df , the values of sigma should be in [1/m]
    :param altitude: the altitude of the the lidar station above sea level, in [m]
    :return: Series of exp(-2*tau)  , tau = integral( sigma(r) * dr)
    """

    heights = col.index.to_numpy ( ) - altitude
    tau = mscLid.calc_tau ( col , heights )
    return pd.Series ( np.exp ( -2 * tau ) )


def calc_beta_profile_df ( row , lambda_nm = 532.0 , ind_n = 'beta' ) :
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
    return pd.Series ( [ rayleigh_scattering.beta_pi_rayleigh ( wavelength = lambda_nm ,
                                                                pressure = row [ 'PRES' ] ,
                                                                temperature = row [ 'TEMPS' ] ,
                                                                C = 385.0 , rh = row [ 'RELHS' ] ) ] ,
                       index = [ ind_n ] )


def generate_daily_molecular_chan ( station , day_date , lambda_nm , time_res = '30S' ,
                                    height_units = 'Km' , optim_size = False , verbose = False ) :
    """
	Generating daily molecular profiles for a given channel's wavelength
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
	:param time_res: Output time resolution required. default is 30sec (according to time resolution of pollyXT measurements)
	:param height_units:  Output units of height grid in 'Km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 4 data variables:
	3 daily dataframes: beta,sigma,att_bsc with shared dimensions (Height, Time)
	and
	1 shared variable: lambda_nm with dimension (Wavelength)
	"""
    logger = logging.getLogger ( )
    '''Load daily gdas profiles and convert to backscatter (beta) and extinction (sigma) profiles'''

    df_sigma , df_beta = get_daily_molecular_profiles ( station , day_date , lambda_nm , height_units )

    ''' Interpolate profiles through 24 hrs'''
    interp_sigma_df = (df_sigma.T.resample ( time_res ).interpolate ( method = 'linear' ) [ :-1 ]).T
    interp_beta_df = (df_beta.T.resample ( time_res ).interpolate ( method = 'linear' ) [ :-1 ]).T
    interp_sigma_df.columns.freq = None
    interp_beta_df.columns.freq = None

    '''Calculate the molecular attenuated backscatter as :  beta_mol * exp(-2*tau_mol)'''
    if height_units == 'Km' :
        # converting height index to meters before tau calculations
        km_index = interp_sigma_df.index.tolist ( )
        meter_index = (np.array ( km_index.copy ( ) ) * 1e+3).tolist ( )
        interp_sigma_df.reindex ( meter_index )
    e_tau_df = interp_sigma_df.apply ( cal_e_tau_df , 0 , args = (station.altitude ,) , result_type = 'expand' )
    if height_units == 'Km' :
        # converting back height index to km before dataset creation
        interp_sigma_df.reindex ( km_index )
        e_tau_df.reindex ( km_index )

    att_bsc_mol_df = interp_beta_df.multiply ( e_tau_df )

    ''' memory size - optimization '''
    if optim_size :
        if verbose :
            logger.debug ( 'Memory optimization - converting molecular values from double to float' )
            size_beta = interp_beta_df.memory_usage ( deep = True ).sum ( )
            size_sigma = interp_sigma_df.memory_usage ( deep = True ).sum ( )
            size_att_bsc = att_bsc_mol_df.memory_usage ( deep = True ).sum ( )

        interp_beta_df = (interp_beta_df.select_dtypes ( include = [ 'float64' ] )). \
            apply ( pd.to_numeric , downcast = 'float' )
        interp_sigma_df = (interp_sigma_df.select_dtypes ( include = [ 'float64' ] )). \
            apply ( pd.to_numeric , downcast = 'float' )
        att_bsc_mol_df = (att_bsc_mol_df.select_dtypes ( include = [ 'float64' ] )). \
            apply ( pd.to_numeric , downcast = 'float' )

        if verbose :
            size_beta_opt = interp_beta_df.memory_usage ( deep = True ).sum ( )
            size_sigma_opt = interp_sigma_df.memory_usage ( deep = True ).sum ( )
            size_att_bsc_opt = att_bsc_mol_df.memory_usage ( deep = True ).sum ( )
            logger.debug ( 'Memory saved for wavelength {} beta: {:.2f}%, sigma: {:.2f}%, att_bsc:{:.2f}%'.
                           format ( lambda_nm , 100.0 * float ( size_beta - size_beta_opt ) / float ( size_beta ) ,
                                    100.0 * float ( size_sigma - size_sigma_opt ) / float ( size_sigma ) ,
                                    100.0 * float ( size_att_bsc - size_att_bsc_opt ) / float ( size_att_bsc ) ) )

    ''' Create molecular dataset'''
    ds_chan = xr.Dataset (
        data_vars = {'beta' : (('Height' , 'Time') , interp_beta_df) ,
                     'sigma' : (('Height' , 'Time') , interp_sigma_df) ,
                     'attbsc' : (('Height' , 'Time') , att_bsc_mol_df) ,
                     'lambda_nm' : ('Wavelength' , np.uint16 ( [ lambda_nm ] ))
                     } ,
        coords = {'Height' : interp_beta_df.index.to_list ( ) ,
                  'Time' : interp_beta_df.columns ,
                  'Wavelength' : np.uint16 ( [ lambda_nm ] )
                  }
    )

    # set attributes of data variables
    ds_chan.beta.attrs = {'long_name' : r'$\beta$' , 'units' : r'$1/m \cdot sr$' ,
                          'info' : 'Molecular backscatter coefficient'}
    ds_chan.sigma.attrs = {'long_name' : r'$\sigma$' , 'units' : r'$1/m $' ,
                           'info' : 'Molecular attenuation coefficient'}
    ds_chan.attbsc.attrs = {'long_name' : r'$\beta \cdot \exp(-2\tau)$' , 'units' : r'$1/m \cdot sr$' ,
                            'info' : 'Molecular attenuated backscatter coefficient'}
    # set attributes of coordinates
    ds_chan.Height.attrs = {'units' : '{}'.format ( height_units ) , 'info' : 'Measurements heights above sea level'}
    ds_chan.Wavelength.attrs = {'long_name' : r'$\lambda$' , 'units' : r'$nm$'}

    return ds_chan


def generate_daily_molecular ( station , day_date , time_res = '30S' ,
                               height_units = 'Km' , optim_size = False , verbose = False ) :
    """
	Generating daily molecular profiles for all elastic channels (355,532,1064)
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param time_res: Output time resolution required. default is 30sec (according to time resolution of pollyXT measurements)
	:param height_units:  Output units of height grid in 'Km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 5 data variables:
			 3 daily dataframes: beta,sigma,att_bsc with shared dimensions(Height, Time, Wavelength)
			 and 2 shared variables: lambda_nm with dimension (Wavelength), and date
	"""
    wavelengths = gs.LAMBDA_nm ( ).get_elastic ( )
    ds_list = [ ]
    # from pytictoc import TicToc
    # t = TicToc()
    # t.tic()
    for lambda_nm in wavelengths :
        ds_chan = generate_daily_molecular_chan ( station , day_date , lambda_nm , time_res = time_res ,
                                                  height_units = height_units , optim_size = optim_size ,
                                                  verbose = verbose )
        ds_list.append ( ds_chan )
    # t.toc()
    '''concatenating molecular profiles of all channels'''
    ds_mol = xr.concat ( ds_list , dim = 'Wavelength' )
    ds_mol [ 'date' ] = day_date
    ds_mol.attrs = {'info' : 'Daily molecular profiles'}
    ds_mol.attrs = {'location' : station.name}
    return ds_mol


def save_molecular_dataset ( station , dataset , save_mode = 'sep' ) :
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
    logger = logging.getLogger ( )
    try :
        date = dataset.date.values
    except ValueError :
        logger.exception ( "The dataset does not contain a data variable named 'date'" )
        return None

    day_date = datetime.utcfromtimestamp ( date.tolist ( ) / 1e9 )
    month_folder = get_month_folder_name ( station.molecular_dataset , day_date )

    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = [ ]
    if save_mode in [ 'both' , 'sep' ] :
        data_vars = list ( dataset.data_vars )
        profiles = data_vars [ 0 :3 ]
        shared_vars = data_vars [ 3 : ]
        # print('The data variables for profiles:' ,profiles) #For debug.should be : 'beta','sigma','attbsc'
        # print('The shared variables: ',shared_vars) #For debug.should be : 'lambda_nm' (wavelength),'date'
        for profile in profiles :
            profile_vars = [ profile ]
            for lambda_nm in dataset.Wavelength.values :
                profile_vars.extend ( shared_vars )
                ds_profile = dataset.get ( profile_vars ).sel ( Wavelength = lambda_nm )
                file_name = get_prep_dataset_file_name ( station , day_date , lambda_nm , data_source = 'molecular' ,
                                                         file_type = profile )
                ncpath = save_dataset ( ds_profile , month_folder , file_name )
                if ncpath :
                    ncpaths.append ( ncpath )

    '''save the dataset to a single netcdf'''
    if save_mode in [ 'both' , 'single' ] :
        file_name = get_prep_dataset_file_name ( station , day_date , lambda_nm = 'all' , data_source = 'molecular' ,
                                                 file_type = 'all' )
        ncpath = save_dataset ( dataset , month_folder , file_name )
        if ncpath :
            ncpaths.append ( ncpath )
    return ncpaths


# %% lidar preprocessing and dataset functions


def get_daily_range_corr ( station , day_date , height_units = 'Km' , optim_size = False , verbose = False ) :
    """
	Retrieving daily range corrected lidar signal (pr^2) from attenuated_backscatter signals in three channels (355,532,1064).
	The attenuated_backscatter are from 4 files of 6-hours *att_bsc.nc for a given day_date and station
	:param station: gs.station() object of the lidar station
	:param day_date: datetime.date object of the required date
	:param height_units:  Output units of height grid in 'Km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
			 1 daily dataset of range_corrected signal in 3 channels, with dimensions of : Height, Time, Wavelength
			 3 variables : lambda_nm, plot_min_range, plot_max_range, with dimension of : Wavelength
			 1 shared variable: date
	"""

    '''get netcdf paths of the attenuation backscatter for given day_date'''
    bsc_paths = get_TROPOS_dataset_paths ( station , day_date , file_type = 'att_bsc' )
    bsc_ds0 = load_dataset ( bsc_paths [ 0 ] )
    altitude = bsc_ds0.altitude.values [ 0 ]
    profiles = [ dvar for dvar in list ( bsc_ds0.data_vars ) if 'attenuated_backscatter' in dvar ]
    wavelengths = [ np.uint ( pname.split ( sep = '_' ) [ -1 ].strip ( 'nm' ) ) for pname in profiles ]

    min_range = np.empty ( (len ( wavelengths ) , len ( bsc_paths )) )
    max_range = np.empty ( (len ( wavelengths ) , len ( bsc_paths )) )

    ds_range_corrs = [ ]
    for ind_path , bsc_path in enumerate ( bsc_paths ) :
        cur_ds = load_dataset ( bsc_path )
        '''get 6-hours range corrected dataset for three channels [355,532,1064]'''
        ds_chans = [ ]
        for ind_wavelength , (pname , lambda_nm) in enumerate ( zip ( profiles , wavelengths ) ) :
            cur_darry = cur_ds.get ( pname ).transpose ( transpose_coords = True )
            ds_chan , LC = get_range_corr_ds_chan ( cur_darry , altitude , lambda_nm , height_units , optim_size ,
                                                    verbose )
            min_range [ ind_wavelength , ind_path ] = LC * cur_darry.attrs [ 'plot_range' ] [ 0 ]
            max_range [ ind_wavelength , ind_path ] = LC * cur_darry.attrs [ 'plot_range' ] [ 1 ]
            ds_chans.append ( ds_chan )

        cur_ds_range_corr = xr.concat ( ds_chans , dim = 'Wavelength' )
        ds_range_corrs.append ( cur_ds_range_corr )

    '''merge range corrected of lidar through 24-hours'''
    ds_range_corr = xr.merge ( ds_range_corrs , compat = 'no_conflicts' )
    # Fixing missing timestamps values:
    time_indx = pd.date_range(start = day_date, end = (day_date+timedelta(hours = 24)-timedelta(seconds = 30)), freq = '30S')
    ds_range_corr = ds_range_corr.reindex ( {"Time" : time_indx} , fill_value = 0 )
    ds_range_corr = ds_range_corr.assign ( {'plot_min_range' : ('Wavelength' , min_range.min ( axis = 1 )) ,
                                            'plot_max_range' : ('Wavelength' , max_range.max ( axis = 1 ))} )
    ds_range_corr [ 'date' ] = day_date
    ds_range_corr.attrs [ 'location' ] = station.location
    ds_range_corr.attrs [ 'info' ] = 'Daily range corrected lidar signal'

    return ds_range_corr


def get_range_corr_ds_chan ( darray , altitude , lambda_nm , height_units = 'Km' , optim_size = False ,
                             verbose = False ) :
    """
	Retrieving a 6-hours range corrected lidar signal (pr^2) from attenuated_backscatter signals in three channels (355,532,1064).
	The attenuated_backscatter are from a 6-hours *att_bsc.nc loaded earlier by darray
	:param darray: is xarray.DataArray object, containing a 6-hours of attenuated_backscatter (loaded from TROPOS *att_bsc.nc)
	:param altitude: altitude of the station [m]
	:param lambda_nm: wavelength in [nm], e.g, for green lambda_nm = 532.0 [nm]
	:param height_units:  Output units of height grid in 'Km' (default) or 'm'
	:param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
	                            True: the retrieved values are of type 'float'.
	:param verbose: Boolean. False(default). True: prints information regarding size optimization.
	:return: xarray.Dataset() holding 2 data variables:
	         1. 6-hours dataframe: range_corr with dimensions (Height, Time)
			 2. shared variable: lambda_nm with dimension (Wavelength)
			 LC - The lidar constant calibration used
	"""
    logger = logging.getLogger ( )
    LC = darray.attrs [ 'Lidar_calibration_constant_used' ]
    times = pd.to_datetime (
        [ datetime.utcfromtimestamp ( np.round ( vtime ) ) for vtime in darray.time.values ] ).values
    if height_units == 'Km' :
        scale = 1e-3
    elif height_units == 'm' :
        scale = 1
    heights_ind = scale * (darray.height.values + altitude)
    rangecorr_df = pd.DataFrame ( LC * darray.values , index = heights_ind , columns = times )

    ''' memory size - optimization '''
    if optim_size :
        if verbose :
            logger.debug ( 'Memory optimization - converting molecular values from double to float' )
            size_rangecorr = rangecorr_df.memory_usage ( deep = True ).sum ( )

        rangecorr_df = (rangecorr_df.select_dtypes ( include = [ 'float64' ] )). \
            apply ( pd.to_numeric , downcast = 'float' )

        if verbose :
            size_rangecorr_opt = rangecorr_df.memory_usage ( deep = True ).sum ( )
            logger.debug ( 'Memory saved for wavelength {} range corrected: {:.2f}%'.
                           format ( lambda_nm ,
                                    100.0 * float ( size_rangecorr - size_rangecorr_opt ) / float ( size_rangecorr ) ) )

    ''' Create range_corr_chan lidar dataset'''
    range_corr_ds_chan = xr.Dataset (
        data_vars = {'range_corr' : (('Height' , 'Time') , rangecorr_df) ,
                     'lambda_nm' : ('Wavelength' , [ lambda_nm ])
                     } ,
        coords = {'Height' : rangecorr_df.index.to_list ( ) ,
                  'Time' : rangecorr_df.columns ,
                  'Wavelength' : [ lambda_nm ]
                  }
    )
    range_corr_ds_chan.range_corr.attrs = {'long_name' : r'$\beta \cdot \exp(-2\tau)$' ,
                                           'units' : r'$photons \cdot m^2$' ,
                                           'info' : 'Range corrected lidar signal from attenuated backscatter multiplied with LC'}
    # set attributes of coordinates
    range_corr_ds_chan.Height.attrs = {'units' : '{}'.format ( '{}'.format ( height_units ) ) ,
                                       'info' : 'Measurements heights above sea level'}
    range_corr_ds_chan.Wavelength.attrs = {'long_name' : r'$\lambda$' , 'units' : r'$nm$'}

    return range_corr_ds_chan , LC


def save_range_corr_dataset ( station , dataset , save_mode = 'sep' ) :
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
    logger = logging.getLogger ( )
    try :
        date = dataset.date.values
    except ValueError :
        logger.exception ( "The dataset does not contain a data variable named 'date'" )
        return None

    day_date = datetime.utcfromtimestamp ( date.tolist ( ) / 1e9 )
    month_folder = get_month_folder_name ( station.lidar_dataset , day_date )

    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = [ ]
    profile = list ( dataset.data_vars ) [ 0 ]
    if save_mode in [ 'both' , 'sep' ] :
        for lambda_nm in dataset.Wavelength.values :
            ds_profile = dataset.sel ( Wavelength = lambda_nm )
            file_name = get_prep_dataset_file_name ( station , day_date , lambda_nm , data_source = 'lidar' ,
                                                     file_type = profile )
            ncpath = save_dataset ( ds_profile , month_folder , file_name )
            if ncpath :
                ncpaths.append ( ncpath )

    '''save the dataset to a single netcdf'''
    if save_mode in [ 'both' , 'single' ] :
        file_name = '{}_{}_{}_lidar.nc'.format ( day_date.strftime ( '%Y_%m_%d' ) , station.location , profile )
        ncpath = save_dataset ( dataset , month_folder , file_name )
        if ncpath :
            ncpaths.append ( ncpath )
    return ncpaths


# %% General functions to handle preprocessing (prep) datasets (figures ,(netcdf) files)


def get_prep_dataset_file_name ( station , day_date , lambda_nm = 532 , data_source = 'molecular' ,
                                 file_type = 'attbsc' ) :
    """
     Retrieves file pattern name of preprocessed dataset according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm] or all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or 'all' (meaning the dataset contains several profile types)

    :return: dataset file name (netcdf) file of the data_type required per given day and wavelength, data_source and file_type
    """
    if file_type == 'all' and lambda_nm == 'all' :
        file_name = f"{day_date.strftime ( '%Y_%m_%d' )}_{station.location}_{data_source}.nc"
    else :
        file_name = f"{day_date.strftime ( '%Y_%m_%d' )}_{station.location}_{file_type}_{lambda_nm}_{data_source}.nc"

    return (file_name)


def get_prep_dataset_paths ( station , day_date , lambda_nm = 532 , data_source = 'molecular' , file_type = 'attbsc' ) :
    """
     Retrieves file paths of preprocessed datasets according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm]
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset

    :return: paths to all datasets netcdf files of the data_type required per given day and wavelength
    """
    if data_source == 'molecular' :
        parent_folder = station.molecular_dataset
    elif data_source == 'lidar' :
        parent_folder = station.lidar_dataset

    month_folder = get_month_folder_name ( parent_folder , day_date )
    file_name = get_prep_dataset_file_name ( station , day_date , lambda_nm , data_source , file_type )

    # print(os.listdir(month_folder))
    file_pattern = os.path.join ( month_folder , file_name )

    paths = sorted ( glob.glob ( file_pattern ) )

    return paths


def visualize_ds_profile_chan ( dataset , lambda_nm = 532 , profile_type = 'range_corr',USE_RANGE = None ,
                                SAVE_FIG = False , dst_folder = os.path.join('..', 'Figures'), format_fig = 'png', dpi = 1000):
    logger = logging.getLogger ( )
    sub_ds = dataset.sel ( Wavelength = lambda_nm ).get ( profile_type )

    # Currently only a dataset with range_corrected variable, has min/max plot_range values
    USE_RANGE = None if (profile_type != 'range_corr') else USE_RANGE
    if USE_RANGE=='MID' :
        [ maxv , minv ] = [
            dataset.sel ( Wavelength = lambda_nm , drop = True ).get ( 'plot_max_range' ).values.tolist ( ) ,
            dataset.sel ( Wavelength = lambda_nm , drop = True ).get ( 'plot_min_range' ).values.tolist ( ) ]
    elif USE_RANGE=='LOW':
        try:
            maxv = dataset.sel ( Wavelength = lambda_nm , drop = True ).get ( 'plot_min_range' ).values.tolist ( )
        except:
            logger.debug("The dataset doesn't 'contain plot_min_range', setting maxv=0")
            maxv=0
        minv = np.nanmin ( sub_ds.values )
    elif USE_RANGE=='HIGH':
        try :
            minv = dataset.sel ( Wavelength = lambda_nm , drop = True ).get ( 'plot_max_range' ).values.tolist ( )
        except:
            logger.debug ( "The dataset doesn't 'contain plot_min_range', setting maxv=0" )
            minv = np.nanmin ( sub_ds.values )
        maxv = np.nanmax(sub_ds.values)
    elif USE_RANGE is None :
        [ maxv , minv ] = [ np.nanmax ( sub_ds.values ) , np.nanmin ( sub_ds.values ) ]

    dims = sub_ds.dims
    if 'Time' not in dims:
        logger.error(f"The dataset should have a 'Time' dimension.")
        return None
    if 'Height' in dims: # plot x- time, y- height
        g = sub_ds.where ( sub_ds < maxv ).where ( sub_ds > minv ).plot ( x = 'Time' , y = 'Height' , cmap = 'turbo' ,figsize = (10 ,6))# ,robust=True)
    elif len(dims) == 2: # plot x- time, y- other dimension
        g = sub_ds.where ( sub_ds < maxv ).where ( sub_ds > minv ).plot ( x = 'Time' , cmap = 'turbo' , figsize = (10 , 6) )
    elif len(dims) == 1: # plot x- time, y - values in profile type
        g = sub_ds.plot ( x = 'Time' , figsize = (10 , 6) )[0]


    # Set time on x-axis
    xfmt = mdates.DateFormatter ( '%H:%M' )
    g.axes.xaxis.set_major_formatter ( xfmt )
    g.axes.xaxis_date ( )
    g.axes.get_xaxis ( ).set_major_locator ( mdates.HourLocator ( interval = 4 ) )
    plt.setp ( g.axes.get_xticklabels ( ) , rotation = 0 , horizontalalignment = 'center' )

    # Set title description
    date_64 = dataset.date.values
    date_datetime = datetime.utcfromtimestamp ( date_64.tolist ( ) / 1e9 )
    date_str = date_datetime.strftime ( '%d/%m/%Y' )
    plt.title (
        '{} - {}nm \n {} {}'.format ( sub_ds.attrs [ 'info' ] , lambda_nm , dataset.attrs [ 'location' ] , date_str ) ,
        y = 1.05 )
    plt.tight_layout ( )
    plt.show ( )

    if SAVE_FIG :

        fname = f"{date_datetime.strftime ( '%Y-%m-%d' )}_{dataset.attrs [ 'location' ]}_{profile_type}_" \
                f"{lambda_nm}_plot_range_{str ( USE_RANGE ).lower ( )}.{format_fig}"

        if not os.path.exists ( dst_folder ) :
            try:
                os.makedirs ( dst_folder , exist_ok = True )
                logger.debug ( f"Creating folder: {dst_folder}" )
            except Exception :
                raise OSError ( f"Failed to create folder: {dst_folder}" )

        fpath = os.path.join ( dst_folder , fname )
        g.figure.savefig ( fpath , bbox_inches = 'tight' , format = format_fig , dpi = dpi )



    return g


# %% Database functions
def query_database ( query = "SELECT * FROM lidar_calibration_constant;" ,
                     database_path = "pollyxt_tropos_calibration.db" ) :
    """
    Query is a string following sqlite syntax (https://www.sqlitetutorial.net/) to query the .db
    Examples:
    query_basic = "
    SELECT * -- This is a comment. Get all columns from table
    FROM lidar_calibration_constant -- Which table to query
    "

    query_advanced = "
    SELECT lcc.id, lcc.liconst, lcc.cali_start_time, lcc.cali_stop_time -- get only some columns
    FROM lidar_calibration_constant as lcc
    WHERE -- different filtering options on rows
        wavelength == 1064 AND
        cali_method LIKE 'Klet%' AND
        (cali_start_time BETWEEN '2017-09-01' AND '2017-09-02');
    "
    """
    # Connect to the db and query it directly into pandas df.
    with sqlite3.connect ( database_path ) as c :
        # Query to df
        # optionally parse 'id' as index column and 'cali_start_time', 'cali_stop_time' as dates
        df = pd.read_sql ( sql = query , con = c , parse_dates = [ 'cali_start_time' , 'cali_stop_time' ] )

    return df


def get_query ( wavelength , cali_method , start_day , till_date ) :
    query = f"""
    SELECT  lcc.liconst, lcc.uncertainty_liconst,
            lcc.cali_start_time, lcc.cali_stop_time,
            lcc.wavelength, lcc.cali_method, lcc.telescope
    FROM lidar_calibration_constant as lcc
    WHERE
        wavelength == {wavelength} AND
        cali_method == '{cali_method}' AND
        telescope == 'far_range' AND
        (cali_start_time BETWEEN '{start_day}' AND '{till_date}');
    """
    return query


def add_profiles_values ( df , station , day_date, file_type = 'profiles' ) :

    df['matched_nc_profile'] = df.apply( lambda row : get_TROPOS_dataset_paths ( station , day_date ,
                                                                      start_time = row.cali_start_time ,
                                                                      end_time = row.cali_stop_time ,
                                                                      file_type = file_type ) [0 ] ,
                              axis = 1 , result_type ='expand' )
    '''
    df['nc_path'] = df.apply(lambda row: get_TROPOS_dataset_file_name(start_time = row.cali_start_time,
                                                                      end_time = row.cali_stop_time,
                                                                      file_type = 'profiles'),
                                                                      axis = 1, result_type ='expand')
    '''

    # Find Actual matching nc profile  file (full path)
    # makes sure only one file is returned
    '''
    df['matched_nc_profile'] = df['nc_path'].apply(lambda row: fnmatch.filter(profile_paths, row)[0]
                                                if len(fnmatch.filter(profile_paths, row)) == 1
                                                else exec("raise(Exception('More than one File'))"))
    '''

    # Get the altitude, delta_r and r0
    def _get_info_from_profile_nc ( row ) :

        data = Dataset ( row [ 'matched_nc_profile' ] )
        wavelen = row.wavelength
        delta_r = data.variables [ f'reference_height_{wavelen}' ] [ : ].data [ 1 ] - \
                  data.variables [ f'reference_height_{wavelen}' ] [ : ].data [ 0 ]
        return data.variables [ 'altitude' ] [ : ].data.item ( ) , delta_r , \
               data.variables [ f'reference_height_{wavelen}' ] [ : ].data [ 0 ]

    df [ [ 'altitude' , 'delta_r' , 'r0' ] ] = df.apply ( _get_info_from_profile_nc , axis = 1 ,
                                                          result_type = 'expand' )
    return df

def add_X_path ( df , station , day_date , lambda_nm = 532 , data_source = 'molecular' , file_type = 'attbsc' ) :
    """
    Add path of 'molecular' or 'lidar' dataset to df
    :param df: pd.DataFrame ( ) result from database query
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm]
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset
    :return: ds with the added collum of the relevant raw
    """

    paths = get_prep_dataset_paths ( station , day_date , lambda_nm , data_source , file_type )
    if len ( paths ) != 1 :
        raise Exception ( f"Expected ONE {data_source} path per day. Got: {paths} " )
    df [ f"{data_source}_path" ] = paths [ 0 ]
    return df


def get_time_frames ( df ) :
    df [ 'start_time' ] = df [ 'cali_start_time' ].dt.round ( '30min' )
    df.loc [ df [ 'start_time' ] < df [ 'cali_start_time' ] , 'start_time' ] += timedelta ( minutes = 30 )

    df [ 'end_time' ] = df [ 'cali_stop_time' ].dt.round ( '30min' )
    df.loc [ df [ 'end_time' ] - df [ 'cali_stop_time' ] > timedelta ( seconds = 30 ) , 'end_time' ] -= timedelta (
        minutes = 30 )
    return df


def create_dataset ( station_name = 'haifa' , sample_size = '30min') :
    """
    CHOOSE: telescope: far_range , METHOD: Klett_Method
    each sample will have 60 bins (aka 30 mins length)
    path to db:  stationdb_file

    :key
    Date DONE
    start_time
    end_time
    period <30 min>
    wavelength DONE
    method (KLET) DONE

    Y:
    LC (linconst from .db) DONE
    LC std (un.linconst from.db) DONE
    r0 (from *_profile.nc DONE
    delta_r (r1-r0) DONE

    X:
    lidar_path (station.lidar_src_folder/<%Y/%M/%d> +nc_zip_file+'att_bsc.nc'
    molecular_path ( station .molecular_src_folder/<%Y/%M><day_mol.nc>
    """

    station = gs.Station ( station_name = station_name )
    db_path = station.db_file
    wavelengths = gs.LAMBDA_nm ( ).get_elastic ( )
    cali_method = 'Klett_Method'

    day_date1 = datetime ( 2017 , 9 , 1 )
    day_date2 = datetime ( 2017 , 9 , 4 )
    full_df = pd.DataFrame ( )
    for wavelength in wavelengths :
        for day_date in [ day_date1 , day_date2 ] :

            day_diff = timedelta ( days = 1 )
            start_day = day_date.strftime ( '%Y-%m-%d' )
            till_date = (day_date + day_diff).strftime ( '%Y-%m-%d' )
            # Query the db for a specific day, wavelength and calibration method
            query = get_query ( wavelength , cali_method , start_day , till_date )
            df = query_database ( query = query , database_path = db_path )

            df [ 'date' ] = day_date.strftime ( '%Y-%m-%d' )

            df = add_profiles_values ( df , station , day_date , file_type = 'profiles' )

            df = add_X_path(df , station, day_date , lambda_nm = wavelength , data_source = 'lidar' ,
                              file_type = 'range_corr' )
            df = add_X_path( df , station , day_date , lambda_nm = wavelength , data_source = 'molecular' ,
                              file_type = 'attbsc' )

            df = get_time_frames ( df )

            expanded_df = pd.DataFrame ( )
            for indx , row in df.iterrows ( ) :
                for start_time , end_time in zip (
                        pd.date_range ( row.loc [ 'start_time' ] , row.loc [ 'end_time' ] , freq = sample_size ) [ :-1 ] ,
                        pd.date_range ( row.loc [ 'start_time' ] , row.loc [ 'end_time' ] , freq = sample_size ).shift ( 1 ) [ :-1 ] ) :
                    mini_df = row.copy ( )
                    mini_df [ 'start_time_period' ] = start_time
                    mini_df [ 'end_time_period' ] = end_time - timedelta ( seconds = 30 )
                    expanded_df = expanded_df.append ( mini_df )

            # reorder the columns
            key = [ 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                    'start_time_period' , 'end_time_period' ]
            Y_features = [ 'liconst' , 'uncertainty_liconst' , 'delta_r' , 'r0' ]
            X_features = [ 'lidar_path' , 'molecular_path' ]
            expanded_df = expanded_df [ key + X_features + Y_features ]
            full_df = full_df.append ( expanded_df )

    return full_df.reset_index ( drop = True )


class customDataSet ( torch.utils.data.Dataset ) :
    """TODO"""

    def __init__ ( self , df ) :
        """
        Args:
            TODO
        """
        self.data = df.copy ( )
        self.key = [ 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                     'start_time_period' , 'end_time_period' ]
        self.Y_features = [ 'liconst' , 'uncertainty_liconst' , 'delta_r' , 'r0' ]
        self.X_features = [ 'lidar_path' , 'molecular_path' ]

    def __len__ ( self ) :
        return len ( self.data )

    def __getitem__ ( self , idx ) :
        row = self.data.loc [ idx , : ]
        X = row [ self.X_features ]
        Y = row [ self.Y_features ]
        # TODO transform X to torch.tensor!
        return X , torch.tensor ( Y )


def main ( ) :
    logging.getLogger ( 'matplotlib' ).setLevel ( logging.ERROR )  # Fix annoying matplotlib logs
    logging.getLogger ( 'PIL' ).setLevel ( logging.ERROR )  # Fix annoying PIL logs
    logger = create_and_configer_logger ( 'preprocessing_log.log' )
    DO_GDAS = False
    DO_NETCDF = False
    DO_DATASET = True
    wavs_nm = gs.LAMBDA_nm ( )
    logger.debug ( f'waves_nm: {wavs_nm}' )

    """set day,location"""
    day_date = datetime ( 2017 , 9 , 1 )
    haifa_station = gs.Station ( stations_csv_path = 'stations.csv' , station_name = 'haifa' )  # haifa_shubi')
    logger.debug ( f"haifa_station: {haifa_station}" )
    min_height = haifa_station.altitude + haifa_station.start_bin_height
    top_height = haifa_station.altitude + haifa_station.end_bin_height

    # GDAS
    if DO_GDAS :
        lambda_nm = wavs_nm.G
        gdas_txt_paths = convert_daily_gdas ( haifa_station , day_date )
        logger.debug ( f'gdas_dst_paths: {gdas_txt_paths}' )
        df_sigma , df_beta = get_daily_molecular_profiles ( haifa_station , day_date , lambda_nm , 'Km' )
        '''Visualize molecular profiles'''
        plt.figure ( )
        df_beta.plot ( )
        plt.ylabel ( r'$\beta_{mol}[1/m]$' )
        plt.title ( 'Molecular backscatter profiles of {} station during {} '.format ( haifa_station.location ,
                                                                                       day_date.strftime (
                                                                                           "%d-%m-%Y" ) ) )
        plt.show ( )

    # NETCDF
    if DO_NETCDF :

        # Get the paths
        logger.debug ( 'start_nc' )
        logger.debug ( f'path {haifa_station.lidar_src_folder}' )
        bsc_paths = get_TROPOS_dataset_paths ( haifa_station , day_date , file_type = 'att_bsc' )
        profile_paths = get_TROPOS_dataset_paths ( haifa_station , day_date , file_type = 'profiles' )

        waves_elastic = wavs_nm.get_elastic ( )  # [UV,G,IR]

        # Extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used for all
        # files in bsc_paths and for all wavelengths
        # TODO do something with the data!
        extract_att_bsc ( bsc_paths , waves_elastic )

        # Query the db for a specific day & wavelength and calibration method
        wavelength = wavs_nm.IR  # or wavelengths[0] # 1064 or
        day_diff = timedelta ( days = 1 )
        start_day = day_date.strftime ( '%Y-%m-%d' )
        till_date = (day_date + day_diff).strftime ( '%Y-%m-%d' )
        cali_method = 'Klett_Method'

        query = f"""
        SELECT lcc.liconst, lcc.cali_start_time, lcc.cali_stop_time, lcc.wavelength
        FROM lidar_calibration_constant as lcc
        WHERE
            wavelength == {wavelength} AND
            cali_method == '{cali_method}' AND
            (cali_start_time BETWEEN '{start_day}' AND '{till_date}');
        """

        db_path = "data_example/pollyxt_tropos_calibration.db"
        df = query_database ( query = query , database_path = db_path )

        # Build matching_nc_file
        # TODO ?? currently matches any two values
        df [ 'nc_path' ] = "*" + df [ 'cali_start_time' ].dt.strftime ( '%Y_%m_%d' ) + \
                           "_" + df [ 'cali_start_time' ].dt.day_name ( ).str.slice ( start = 0 , stop = 3 ) + \
                           "_TROPOS_" + "??_00_01_" + \
                           df [ 'cali_start_time' ].dt.strftime ( '%H%M' ) + "_" + \
                           df [ 'cali_stop_time' ].dt.strftime ( '%H%M' ) + "_profiles.nc"

        # Find Actual matching nc file (full path)
        def get_file_match ( x ) :
            matched_file = fnmatch.filter ( profile_paths , x )
            if len ( matched_file ) == 1 :
                return matched_file [ 0 ]
            else :
                raise Exception  # make sure only one file is returned

        df [ 'matched_nc_file' ] = df [ 'nc_path' ].apply ( get_file_match )

        # Get the altitude (r0) and delta_r
        def get_info_from_profile_nc ( row ) :
            data = Dataset ( row [ 'matched_nc_file' ] )
            wavelen = row.wavelength
            delta_r = data.variables [ f'reference_height_{wavelen}' ] [ : ].data [ 1 ] - \
                      data.variables [ f'reference_height_{wavelen}' ] [ : ].data [ 0 ]
            return data.variables [ 'altitude' ] [ : ].data.item ( ) , delta_r

        df [ [ 'altitude' , 'delta_r' ] ] = df.apply ( get_info_from_profile_nc , axis = 1 , result_type = 'expand' )
        # TODO do something with the df
        pass  # add breakpoint here to see the df

    if DO_DATASET :
        df = create_dataset ( station_name = 'haifa' )
        dataset = customDataSet ( df )
        dataloader = torch.utils.data.DataLoader ( dataset , batch_size = 4 ,
                                                   shuffle = True )

        for i_batch , sample_batched in enumerate ( dataloader ) :
            print ( i_batch , sample_batched )
            break


if __name__ == '__main__' :
    main ( )
