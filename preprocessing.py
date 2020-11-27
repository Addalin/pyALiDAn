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


# %%
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
    if col_names is None :
        col_names = [ 'PRES' , 'HGHT' , 'TEMP' , 'UWND' , 'VWND' , 'WWND' , 'RELH' , 'TPOT' , 'WDIR' , 'WSPD' ]
    data_src = pd.read_fwf ( src_file , skiprows = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 8 ] , delimiter = "\s+" ,
                             skipinitialspace = True ).dropna ( )
    # converting any kind of blank spaces to zeros
    for col in data_src.columns :
        if not is_numeric_dtype ( data_src [ col ] ) :
            data_src [ col ] = pd.core.strings.str_strip ( data_src [ col ] )
            data_src [ col ] = data_src [ col ].replace ( '' , '0' ).astype ( 'float64' )
    data_src.columns = col_names
    data_src.to_csv ( dst_file , index = False , sep = '\t' , na_rep = '\t' )
    # TODO: add warning if fails and exit


def get_gdas_paths ( station , day_date , f_type = 'gdas1' ) :
    """
    Retrieves gdas container folder and file paths for a given date
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param f_type: 'gdas1' - for the original gdas files (from TROPOS), 'txt' - for converted (fixed) gdas files
    :return: gdas_folder, gdas_paths - the folder containing the gdas files and the file paths.
    NOTE: during a daily conversion, the function creates a NEW gdas_folder (for the converted txt files),
    and returns an EMPTY list of gdas_paths. The creation of NEW gdas_paths is done in convert_daily_gdas()
    """
    month_folder = os.path.join ( day_date.strftime ( '%Y' ) , day_date.strftime ( '%m' ) )
    if f_type == 'gdas1' :
        gdas_folder = os.path.join ( station.gdas1_folder , month_folder )
    elif f_type == 'txt' :
        gdas_folder = os.path.join ( station.gdastxt_folder , month_folder )

    # TODO: Add path validation and warning if not exits (exit on failure)
    if not os.path.exists ( gdas_folder ) :
        os.makedirs ( gdas_folder )

    gdas_day_pattern = '{}_{}_*_{}_{}.{}'.format ( station.location , day_date.strftime ( '%Y%m%d' ) ,
                                                   station.lat , station.lon , f_type )
    gdas_paths = sorted ( glob.glob ( os.path.join ( gdas_folder , gdas_day_pattern ) ) )
    return gdas_folder , gdas_paths


def convert_daily_gdas ( station , day_date ) :
    """
    Converting gdas files from TROPOS of type .gdas1 to .txt for a given day.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the date to be converted
    :return:
    """
    # TODO: Add namings and path validation (print warnings and errors) , e.g. if a path relate to a day_date doesn't exsit
    # get source container folder and file paths (.gdas1)
    src_folder , gdas1_paths = get_gdas_paths ( station , day_date , 'gdas1' )

    # set dest container folder and file paths (.txt)
    dst_folder , _ = get_gdas_paths ( station , day_date , 'txt' )
    gdastxt_paths = [ sub.replace ( src_folder , dst_folder ).replace ( 'gdas1' , 'txt' ) for sub in gdas1_paths ]

    # convert each src_file (as .gdas1) to dst_file (as .txt)
    for (src_file , dst_file) in zip ( gdas1_paths , gdastxt_paths ) :
        gdas2radiosonde ( src_file , dst_file )

    return gdastxt_paths
    #  TODO:  create function to convert of a big chunk of gdas files (start date to end date)


def extract_date_time ( path , format_filename , format_times ) :
    """
    Extracting datetime from file name using a formatter string.
    :param path: path to folder containing the files to observe
    :param format_filename:  a formatter string
    :param format_times: format of datetime in the file name
    :return: time_stamps - A list of datetime objects
    usage:  create a formatter string: format_filename=  r'-(.*)_(.*)-(.*)-.*.txt'
            Call the function: time_stamp = extract_date_time(soundePath,r'40179_(.*).txt',['%Y%m%d_%H'])

    """
    filename = os.path.basename ( path )
    # print(filename)
    matchObj = re.match ( format_filename , filename , re.M | re.I )
    # print(matchObj)
    time_stamps = [ ]
    for fmt_time , grp in zip ( format_times , matchObj.groups ( ) ) :
        time_stamps.append ( datetime.strptime ( grp , fmt_time ) )
    return time_stamps


def calc_sigma_profile_df ( row , lambda_um = 532.0 , indx_n = 'sigma' ) :
    """
    Returns pd series of extinction profile [1/m] from a radiosonde dataframe containing the
    columns:['PRES','TEMPS','RELHS']. The function applies on rows of the radiosonde df.
    :param row: row of radiosonde df
    :param lambda_um: wavelength in [um], e.g, for green lambda_um = 532.0 [um]
    :param indx_n: index name, the column name of the result. The default is 'sigma'
    but could be useful to get profiles for several hours each have a different index name
    (e.g., datetime object of measuring time of the radiosonde as datetime.datetime(2017, 9, 2, 0, 0))
    :return: pd series of extinction profile [1/m]
    """
    return pd.Series (
        data = [ rayleigh_scattering.alpha_rayleigh ( wavelength = lambda_um , pressure = row [ 'PRES' ] ,
                                                      temperature = row [ 'TEMPS' ] , C = 385.0 ,
                                                      rh = row [ 'RELHS' ] ) ] ,
        index = [ indx_n ] )


def cal_e_tau_df ( col , altitude ) :
    """
    Calculate the the attenuated optical depth (tau is the optical depth).
    Calculation is per column of the backscatter (sigma) dataframe.
    :param col: column of sigma_df
    :param altitude: the altitude of the the lidar station above sea level
    :return: Series of exp(-2*tau)  , tau = integral( sigma(r) * dr)
    """

    # TODO: validate height scale (m or km)
    heights = col.index.to_numpy ( ) - altitude
    tau = mscLid.calc_tau ( col , heights )
    return pd.Series ( np.exp ( -2 * tau ) )


def calc_beta_profile_df ( row , lambda_um = 532.0 , ind_n = 'beta' ) :
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
    return pd.Series ( [ rayleigh_scattering.beta_pi_rayleigh ( wavelength = lambda_um ,
                                                                pressure = row [ 'PRES' ] ,
                                                                temperature = row [ 'TEMPS' ] ,
                                                                C = 385.0 , rh = row [ 'RELHS' ] ) ] ,
                       index = [ ind_n ] )


def generate_daily_molecular_profile ( gdas_txt_paths , lambda_um = 532 ,
                                       location = 'haifa' , lat = 32.8 , lon = 35.0 ,
                                       min_height = 0.229 , top_height = 22.71466 , h_bins = 3000 ) :
    """
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
    """

    heights = np.linspace ( min_height , top_height ,
                            h_bins )  # setting heights for interpolation of gdas/radiosonde inputs
    # uncomment the commands below for sanity check of the desired height resolution
    # dr = heights[1]-heights[0]
    # print('h_bins=',heights.shape,' min_height=', min_height,' top_height=', top_height,' dr=',dr )

    df_sigma = pd.DataFrame ( )
    df_beta = pd.DataFrame ( )

    for path in gdas_txt_paths :
        df_sonde = RadiosondeProfile ( path ).get_df_sonde ( heights )
        time = extract_date_time ( path , r'{}_(.*)_{}_{}.txt'.format ( location , lat , lon ) , [ '%Y%m%d_%H' ] ) [ 0 ]
        '''Calculating molecular profiles from temperature and pressure'''
        res = df_sonde.apply ( calc_sigma_profile_df , axis = 1 , args = (lambda_um , time ,) ,
                               result_type = 'expand' ).astype (
            'float64' )
        df_sigma [ res.columns ] = res
        res = df_sonde.apply ( calc_beta_profile_df , axis = 1 , args = (lambda_um , time ,) ,
                               result_type = 'expand' ).astype (
            'float64' )
        df_beta [ res.columns ] = res
    df_sigma.index = heights
    df_beta.index = heights

    return df_sigma , df_beta


def load_att_bsc ( lidar_parent_folder , day_date ) :
    """
    Load netcdf file of the attenuation backscatter profile(att_bsc.nc) according to date
    :param lidar_parent_folder: this is the station main folder of the lidar
    :param day_date: datetime obj of the measuring date
    :return: paths to all *att_bsc.nc in the saved for day_date
    """
    #
    # TODO: rename this function to get_att_bsc_paths
    # TODO: split this function to get_att_bsc_paths(lidar_parent_folder, day_date) and get_profiles_paths(lidar_parent_folder, day_date)
    lidar_day_folder = os.path.join ( lidar_parent_folder , day_date.strftime ( "%Y" ) , day_date.strftime ( "%m" ) ,
                                      day_date.strftime ( "%d" ) )
    os.listdir ( lidar_day_folder )

    bsc_pattern = os.path.join ( lidar_day_folder , "*[0-9]_att_bsc.nc" )
    profile_pattern = os.path.join ( lidar_day_folder , "*[0-9]_profiles.nc" )

    bsc_paths = sorted ( glob.glob ( bsc_pattern ) )
    profile_paths = sorted ( glob.glob ( profile_pattern ) )

    return bsc_paths , profile_paths


def extract_att_bsc ( bsc_paths , wavelengths ) :
    """
    For all .nc files under bsc_paths and for each wavelength in wavelengths
    extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used

    :param bsc_paths: paath to netcdf folder
    :param wavelengths: iterable, list of wavelengths
    :return:
    """

    for wavelen in wavelengths :
        for bsc_path in bsc_paths :
            data = Dataset ( bsc_path )
            file_name = bsc_path.split ( '/' ) [ -1 ]
            try :
                vals = data.variables [ f'OC_attenuated_backscatter_{wavelen}nm' ]
                arr = vals * vals.Lidar_calibration_constant_used
                print ( f"Extracted OC_attenuated_backscatter_{wavelen}nm from {file_name}" )
            except KeyError as e :
                print ( f"Key {e} does not exist in {file_name}" )


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


def main ( ) :
    DO_GDAS = True
    DO_NETCDF = True
    wavs_um = gs.LAMBDA_um ( )
    print ( 'waves_um' , wavs_um )
    """set day,location"""
    day_date = datetime ( 2017 , 9 , 1 )
    haifa_station = gs.station ( )
    print ( haifa_station )
    # location = haifa_station.location
    min_height = haifa_station.altitude + haifa_station.start_bin_height
    top_height = haifa_station.altitude + haifa_station.end_bin_height

    # GDAS
    if DO_GDAS :
        lambda_um = wavs_um.G
        gdas_txt_paths = convert_daily_gdas ( haifa_station, day_date )
        # gdas_txt_paths = gdas_tropos2txt ( day_date , haifa_station.location , haifa_station.lat , haifa_station.lon )
        print ( 'gdas_dst_paths' , gdas_txt_paths )
        df_sigma , df_beta = generate_daily_molecular_profile ( gdas_txt_paths , lambda_um ,
                                                                haifa_station.location , haifa_station.lat ,
                                                                haifa_station.lon , 1E-3 * min_height ,
                                                                1E-3 * top_height ,
                                                                haifa_station.n_bins )
        '''Visualize molecular profiles'''
        plt.figure ( )
        df_beta.plot ( )
        plt.xlabel ( 'Height [km]' )
        plt.ylabel ( 'Time [hr]' )
        plt.title ( 'backscatter profiles of {} station during {} '.format ( haifa_station.location ,
                                                                             day_date.strftime ( "%d-%m-%Y" ) ) )
        plt.show ( )

    # NETCDF
    if DO_NETCDF :

        # Get the paths
        print ( 'start_nc' )
        lidar_parent_folder = os.path.join ( '.' , 'data examples' , 'netcdf' )
        print ( 'path' , lidar_parent_folder )
        bsc_paths , profile_paths = load_att_bsc ( lidar_parent_folder , day_date )

        waves_elastic = wavs_um.get_elastic ( )  # [UV,G,IR]

        # Extract the OC_attenuated_backscatter_{wavelen}nm and Lidar_calibration_constant_used for all
        # files in bsc_paths and for all wavelengths
        # TODO do something with the data!
        extract_att_bsc ( bsc_paths , waves_elastic )

        # Query the db for a specific day & wavelength and calibration method
        wavelength = wavs_um.IR  # or wavelengths[0] # 1064 or
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

        db_path = "data examples/netcdf/pollyxt_tropos_calibration.db"
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


if __name__ == '__main__' :
    main ( )
