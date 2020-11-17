import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import os
from datetime import datetime, timedelta
import glob
from molecular import rayleigh_scattering
import numpy as np
from constsLidar import LAMBDA, min_height
from generate_atmosphere import RadiosondeProfile
import re
from netCDF4 import Dataset

def gdas2radiosonde(src_file, dst_file, col_names=None):
    """
    Helper function that converts a gdas file from TROPOS server, to a simple txt.
    The resulting file is without any prior info, and resembles the table format
    of a radiosonde file (see class: RadiosondeProfile)

    :param src_file: source file name
    :param dst_file: destination file name
    :param col_names: column names of the final table
    """
    if col_names is None:
        col_names = ['PRES', 'HGHT', 'TEMP', 'UWND',
                     'VWND', 'WWND', 'RELH', 'TPOT', 'WDIR', 'WSPD']

    data_src = pd.read_fwf(src_file, skiprows=[0, 1, 2, 3, 4, 5, 6, 8],
                           delimiter="\s+", skipinitialspace=True).dropna()
    # converting any kind of blank spaces to zeros
    for col in data_src.columns:
        if not is_numeric_dtype(data_src[col]):
            data_src[col] = pd.core.strings.str_strip(data_src[col])
            data_src[col] = data_src[col].replace('', '0').astype('float64')
    data_src.columns = col_names
    data_src.to_csv(dst_file, index=False, sep='\t', na_rep='\t')


# TODO: add warning if failed


def gdas_tropos2txt(day_date, lon, lat):
    # Converting gdas files from TROPOS to txt
    # set source gdas files
    # TODO: Add namings and existing path validation (print warnings and errors)
    src_folder = os.path.join(os.getcwd(), 'data_example', 'data_examples', 'gdas')
    if not os.path.exists(src_folder):
        os.makedirs(src_folder)
    gdas_curd_pattern = 'haifa_{}_*_{}_{}.gdas1'.format(day_date.strftime('%Y%m%d'), lat, lon)
    gdas_nxtd_pattern = 'haifa_{}_00_{}_{}.gdas1'.format((day_date + timedelta(days=1)).strftime('%Y%m%d'), lat, lon)
    gdas_src_paths = sorted(glob.glob(os.path.join(src_folder, gdas_curd_pattern)))
    gdas_src_paths.append(os.path.join(src_folder, gdas_nxtd_pattern))

    '''set dest txt files'''
    dst_folder = os.path.join(os.getcwd(), 'tmp2')
    # TODO: Add validation and indicate if folder already existed or created now (print warnings and errors)
    Path(dst_folder).mkdir(parents=True, exist_ok=True)
    gdas_dst_paths = [sub.replace(src_folder, dst_folder).replace('gdas1', 'txt') for sub in gdas_src_paths]

    for (src_file, dst_file) in zip(gdas_src_paths, gdas_dst_paths):
        gdas2radiosonde(src_file, dst_file)
        print('\n Done conversion src: ', src_file, 'dst: ', dst_file)
        # sanity check
        # data_dst =pd.read_csv(dst_file,delimiter="\s+")
        # print(data_dst)

    # TODO : set 'gdas_src_paths' and 'src_folder' according to 'start_date' and 'end_date' -
    #  for conversion of a big chunk of gdas files. See examples below.
    # TODO : set 'gdas_dst_paths' in similar way. Such that the folders of gdas and txt files will have same tree
    #  (or just save it in the same folders?)
    '''gdas_month_folder = os.path.join(gdas_parent_folder, day_date.strftime("%Y\%m"))
    #print (os.path.exists(gdas_month_folder))
    
    gdas_cur_pattern = 'haifa_{}_*_{}_{}.gdas1'.format(day_date.strftime('%Y%m%d'),lat,lon)
    gdas_next = 'haifa_{}_00_{}_{}.gdas1'.format((day_date+timedelta(days=1)).strftime('%Y%m%d'),lon,lat)
    gdas_pattern  = os.path.join(gdas_month_folder,gdas_cur_pattern)
    gdas_paths = sorted(glob.glob(gdas_pattern))
    gdas_paths.append(os.path.join(gdas_month_folder,gdas_next))
    #gdas_file = Ar.fname_from_date(day_date)
    #print('name of input file ', gdas_pattern)'''
    return gdas_dst_paths


def extract_date_time(path, format_filename, format_times):
    # Extracting datetime from file name using a formatter string.
    #
    # Usage:
    # create a formatter string: format_filename=  r'-(.*)_(.*)-(.*)-.*.txt'
    # Call the function:        f time_stamp = extract_date_time(soundePath,r'40179_(.*).txt',['%Y%m%d_%H'])
    # Output:
    #       time_stamps - A list of datetime objects
    filename = os.path.basename(path)
    # print(filename)
    matchObj = re.match(format_filename, filename, re.M | re.I)
    # print(matchObj)
    time_stamps = []
    for fmt_time, grp in zip(format_times, matchObj.groups()):
        time_stamps.append(datetime.strptime(grp, fmt_time))
    return time_stamps


def calc_sigma_profile_df(row, lambda_um=532.0, indx_n='sigma'):
    """
    Returns pd series of extinction profile [1/m] from a radiosonde dataframe containing the
    columns:['PRES','TEMPS',RELHS]. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_um: wavelength in [um], e.g, for green lambda_um = 532.0 [um]
    :param indx_n: index name, the column name of the result. The default is 'sigma'
    but could be useful to get profiles for several hours each have a different index name
    (e.g., datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd series of extinction profile [1/m]
    """
    return pd.Series([rayleigh_scattering.alpha_rayleigh(wavelength=lambda_um,
                                                         pressure=row['PRES'],
                                                         temperature=row['TEMPS'],
                                                         C=385.0, rh=row['RELHS'])], index=[indx_n])


def calc_beta_profile_df(row, lambda_um=532.0, ind_n='beta'):
    """
    Returns pd series of backscatter profile from a radiosonde dataframe containing the
    columns:['PRES','TEMPS',RELHS]. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_um: wavelength in [um], e.g, for green lambda_um = 532.0 [um]
    :param ind_n: index name, the column name of the result. The default is 'beta'
    but could be useful to get profiles for several hours each have a different index name
    (e.g., datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd series of backscatter profile [1/sr*m]
    """
    return pd.Series([rayleigh_scattering.beta_pi_rayleigh(wavelength=lambda_um,
                                                           pressure=row['PRES'],
                                                           temperature=row['TEMPS'],
                                                           C=385.0, rh=row['RELHS'])], index=[ind_n])


def load_att_bsc(lidar_parent_folder, day_date):
    # Load netcdf file of the attenuation backscatter profile(att_bsc.nc)

    lidar_day_folder = os.path.join(lidar_parent_folder, day_date.strftime("%Y/%m/%d"))
    os.listdir(lidar_day_folder)

    bsc_pattern = os.path.join(lidar_day_folder, "*_att_bsc.nc")
    profile_pattern = os.path.join(lidar_day_folder, "*[0-9]_profiles.nc")

    bsc_paths = sorted(glob.glob(bsc_pattern))
    profile_paths = sorted(glob.glob(profile_pattern))

    return bsc_paths, profile_paths


def generate_daily_molecular_profile(gdas_dst_paths):
    # GENERATE DAILY MOLECULAR PROFILE from the converted files (above)

    '''physical parameters'''
    lambda_um = LAMBDA.G * 1e+9
    dr = 7.47e-3  # Height resolution dr~= 7.5e-3[km] (similar to the lidar height resolution)
    top_height = 22.48566  # Top height for interest signals (similar to the top height of the lidar measurement)
    heights = np.arange(min_height, top_height, dr)  # setting heights for interpolation of gdas/radiosonde inputs
    print(heights.shape, min_height)
    df_sigma = pd.DataFrame()
    df_beta = pd.DataFrame()

    for dst_file in gdas_dst_paths:
        df_sonde = RadiosondeProfile(dst_file).get_df_sonde(heights)
        # df_sonde = sondeprofile.get_df_sonde(heights)
        time = extract_date_time(dst_file, r'haifa_(.*)_32.8_35.0.txt', ['%Y%m%d_%H'])[0]
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply(calc_sigma_profile_df, axis=1, args=(lambda_um, time,), result_type='expand').astype(
            'float64')
        df_sigma[res.columns] = res
        res = df_sonde.apply(calc_beta_profile_df, axis=1, args=(lambda_um, time,), result_type='expand').astype(
            'float64')
        df_beta[res.columns] = res

    return df_sigma, df_beta


def extract_att_bsc(lidar_parent_folder, day_date, wavelengths):
    """
    For all .nc files under lidar_parent_folder with day_date dat, and for each wavelength in wavelengths
    extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used

    :param lidar_parent_folder: aath to netcdf folder
    :param day_date: datetime to extract
    :param wavelengths: iterable, list of wavelengths

    :return:
    """

    bsc_paths, profile_paths = load_att_bsc(lidar_parent_folder, day_date)
    for wavelen in wavelengths:
        for bsc_path in bsc_paths:
            data = Dataset(bsc_path)
            try:
                vals = data.variables[f'OC_attenuated_backscatter_{wavelen}nm']
                arr = vals * vals.Lidar_calibration_constant_used
                print(f"Extracted OC_attenuated_backscatter_{wavelen}nm from {bsc_path.split('/')[-1]}")
            except KeyError as e:
                print(f"Key {e} does not exist in {bsc_path.split('/')[-1]}")


def main():
    """set day,location"""
    lon = 35.0
    lat = 32.8
    day_date = datetime(2017, 9, 1)

    gdas_dst_paths = gdas_tropos2txt(day_date, lon, lat)
    df_sigma, df_beta = generate_daily_molecular_profile(gdas_dst_paths)

    wavelengths = ['355', '532', '1064']
    lidar_parent_folder = 'data_example/data_examples/netcdf'
    extract_att_bsc(lidar_parent_folder, day_date, wavelengths)


if __name__ == '__main__':
    main()
