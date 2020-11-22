import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
import os
from datetime import datetime, timedelta
import glob
from molecular import rayleigh_scattering
import numpy as np
import global_settings as gs #import #WAVELEN, min_height
from generate_atmosphere import RadiosondeProfile
import re
from netCDF4 import Dataset
import sqlite3
import fnmatch
import matplotlib.pyplot as plt




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


def gdas_tropos2txt(day_date, location='haifa', lat=32.8, lon=35.0):
    # Converting gdas files from TROPOS to txt
    # set source gdas files
    # TODO: Add namings and existing path validation (print warnings and errors)
    src_folder = os.path.join(os.getcwd(), 'data_example', 'data_examples', 'gdas')
    if not os.path.exists(src_folder):
        os.makedirs(src_folder)
    gdas_curd_pattern = '{}_{}_*_{}_{}.gdas1'.format(location,day_date.strftime('%Y%m%d'), lat, lon)
    gdas_nxtd_pattern = '{}_{}_00_{}_{}.gdas1'.format(location,(day_date + timedelta(days=1)).strftime('%Y%m%d'), lat, lon)
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


def generate_daily_molecular_profile(gdas_txt_paths, lambda_um=532,
                                     location='haifa', lat=32.8, lon=35.0,
                                     min_height=0.229, top_height=22.71466, h_bins=3000):
    '''
    Generating daily molecular profile from gdas txt file
    :param gdas_txt_paths: paths to gdas txt files , containing table with the columns
    "PRES	HGHT	TEMP	UWND	VWND	WWND	RELH	TPOT	WDIR	WSPD"
    :param lambda_um: wavelength [um] e.g., for the green channel 532 [um]
    :param location: location name of the lidar station, e.g., Haifa
    :param lat: latitude of the station
    :param lon: longitude of the station
    :param min_height: Min height [km] **above sea level** (preferably this is the lidar height above sea level, e.g., Haifa min_height=0.229[km])
    :param top_height: Top height [km] **above see level** (preferably this is lidar_top_height_of + min_height)
    :param h_bins: Number of height bins                   (preferably this is the lidar height bins, sanity check for
    this parameter gives dr= heights[1]-heights[0] =~ 7.5e-3[km] -> the height resolution of pollyXT lidar)
    :return: Returns backscatter and extinction profiles as pandas-dataframes, according to Rayleigh scattering.
    The resulted profile start at min_height, end at top_height and extrapolated to have h_bins.
    '''

    heights = np.linspace(min_height, top_height, h_bins) # setting heights for interpolation of gdas/radiosonde inputs
    # uncomment the commands below for sanity check of the desired height resolution
    # dr = heights[1]-heights[0]
    # print('h_bins=',heights.shape,' min_height=', min_height,' top_height=', top_height,' dr=',dr )

    df_sigma = pd.DataFrame()
    df_beta = pd.DataFrame()

    for path in gdas_txt_paths:
        df_sonde = RadiosondeProfile(path).get_df_sonde(heights)
        time = extract_date_time(path, r'{}_(.*)_{}_{}.txt'.format(location,lat,lon), ['%Y%m%d_%H'])[0]
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply(calc_sigma_profile_df, axis=1, args=(lambda_um, time,), result_type='expand').astype(
            'float64')
        df_sigma[res.columns] = res
        res = df_sonde.apply(calc_beta_profile_df, axis=1, args=(lambda_um, time,), result_type='expand').astype(
            'float64')
        df_beta[res.columns] = res

    return df_sigma, df_beta


def extract_att_bsc(bsc_paths, wavelengths):
    """
    For all .nc files under bsc_paths and for each wavelength in wavelengths
    extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used

    :param lidar_parent_folder: aath to netcdf folder
    :param day_date: datetime to extract
    :param wavelengths: iterable, list of wavelengths

    :return:
    """

    for wavelen in wavelengths:
        for bsc_path in bsc_paths:
            data = Dataset(bsc_path)
            file_name = bsc_path.split('/')[-1]
            try:
                vals = data.variables[f'OC_attenuated_backscatter_{wavelen}nm']
                arr = vals * vals.Lidar_calibration_constant_used
                print(f"Extracted OC_attenuated_backscatter_{wavelen}nm from {file_name}")
            except KeyError as e:
                print(f"Key {e} does not exist in {file_name}")


def query_database(query="SELECT * FROM lidar_calibration_constant;", database_path="pollyxt_tropos_calibration.db"):
    """
    Query is a string following sqlite syntax (https://www.sqlitetutorial.net/) to query the .db
    Examples:
    query_basic = "
    SELECT * -- This is a comment. Get all columns from table
    FROM lidar_calibration_constant -- Which table to query
    ;
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
    with sqlite3.connect(database_path) as c:
        # Query to df
        # optionally parse 'id' as index column and 'cali_start_time', 'cali_stop_time' as dates
        df = pd.read_sql(sql=query, con=c, parse_dates=['cali_start_time', 'cali_stop_time'])

    return df


def main():
    DO_GDAS = True
    DO_NETCDF = True

    """set day,location"""
    day_date = datetime(2017, 9, 1)
    location = 'haifa'
    station = gs.stations[location]
    min_height = gs.altitude + gs.start_bin_height
    top_height = gs.altitude + gs.end_bin_height

    # GDAS
    if DO_GDAS:
        lambda_um = gs.LAMBDA_um.G
        gdas_dst_paths = gdas_tropos2txt(day_date, gs.location, gs.lat, gs.lon)
        df_sigma, df_beta = generate_daily_molecular_profile(gdas_dst_paths, lambda_um, gs.location,
                                                             gs.lat, gs.lon, min_height, top_height, gs.n_bins)
        '''Visualize molecular profiles'''
        plt.figure()
        df_beta.plot()
        plt.show()


    # NETCDF
    if DO_NETCDF:

        # Get the paths
        lidar_parent_folder = 'data_example/data_examples/netcdf'
        bsc_paths, profile_paths = load_att_bsc(lidar_parent_folder, day_date)

        wavelengths = gs.LAMBDA_um.get_elastic() #[UV,G,IR]

        # Extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used for all
        # files in bsc_paths and for all wavelengths
        # TODO do something with the data!
        extract_att_bsc(bsc_paths, wavelengths)

        # Query the db for a specific day & wavelength and calibration method
        wavelength = gs.LAMBDA_um.IR # or wavelengths[0] # 1064 or
        day_diff = timedelta(days=1)
        start_day = day_date.strftime('%Y-%m-%d')
        till_date = (day_date + day_diff).strftime('%Y-%m-%d')
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
        df = query_database(query=query, database_path=db_path)

        # Build matching_nc_file
        # TODO ?? currently matches any two values
        df['nc_path'] = "*" + df['cali_start_time'].dt.strftime('%Y_%m_%d') + \
                        "_" + df['cali_start_time'].dt.day_name().str.slice(start=0, stop=3) + \
                        "_TROPOS_" + "??_00_01_" + \
                        df['cali_start_time'].dt.strftime('%H%M') + "_" + \
                        df['cali_stop_time'].dt.strftime('%H%M') + "_profiles.nc"

        # Find Actual matching nc file (full path)
        def get_file_match(x):
            matched_file = fnmatch.filter(profile_paths, x)
            if len(matched_file) == 1:
                return matched_file[0]
            else:
                raise Exception  # make sure only one file is returned

        df['matched_nc_file'] = df['nc_path'].apply(get_file_match)

        # Get the altitude (r0) and delta_r
        def get_info_from_profile_nc(s):
            data = Dataset(s['matched_nc_file'])
            wavelen = s.wavelength
            delta_r = data.variables[f'reference_height_{wavelen}'][:].data[1] - \
                      data.variables[f'reference_height_{wavelen}'][:].data[0]
            return data.variables['altitude'][:].data.item(), delta_r

        df[['altitude', 'delta_r']] = df.apply(get_info_from_profile_nc, axis=1, result_type='expand')
        # TODO do something with the df
        pass # add breakpoint here to see the df


if __name__ == '__main__':
    main()
