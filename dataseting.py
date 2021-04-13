# %% #General Imports
import warnings

warnings.filterwarnings ( "ignore" )
import os,sys
from time import time
from datetime import datetime , timedelta  # time,
import multiprocess as mp
from IPython.display import display
import logging
from tqdm import tqdm

# %% Scientific and data imports
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d , gaussian_filter
from scipy.interpolate import griddata
from functools import partial

eps = np.finfo ( np.float ).eps

# %% Local modules imports
import global_settings as gs
import data_loader as dl
import preprocessing as prep
from utils import create_and_configer_logger


# %% Dataset creating helper functions
def create_dataset ( station_name = 'haifa' , start_date = datetime ( 2017 , 9 , 1 ) ,
                     end_date = datetime ( 2017 , 9 , 2 ) , sample_size = '29.5min' ,list_dates =[]) :
    """
    CHOOSE: telescope: far_range , METHOD: Klett_Method
    each sample will have 60 bins (aka 30 mins length)
    path to db:  stationdb_file

    :param start_date:
    :param end_date:
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
    Dataset of calibrated samples for learning calibration method.

    The samples were generated by TROPOS using Pollynet_Processing_Chain.
    For more info visit here: https://github.com/PollyNET/Pollynet_Processing_Chain
    # %% Generating Database of samples:
    # X = {range corrected signal (lidar source), attbsc signal (molecular source)}
    # Y = estimated values {LC,r0,r1} (and also LC_std is possible)
    # TODO : organize and update usage
    """
    logger = logging.getLogger ( )
    station = gs.Station ( station_name = station_name )
    db_path = station.db_file
    wavelengths = gs.LAMBDA_nm ( ).get_elastic ( )
    cali_method = 'Klett_Method'

    if len(list_dates)>0:
        try:
            dates = [datetime.combine(t.date(), t.time()) for t in list_dates]
        except TypeError as e :
            logger.exception ( f"{e}: `list_dates`, when not empty, is excepted to have datetime.datetime() objects.")
            return
    else:
        dates = pd.date_range ( start = start_date , end = end_date , freq = 'D' ).to_pydatetime ( ).tolist ( )

    full_df = pd.DataFrame ( )
    for wavelength in tqdm ( wavelengths ) :
        for day_date in dates :  # tqdm(dates) #TODO: find a way to run inner loop of tqdm, in a convenient way

            # Query the db for a specific day, wavelength and calibration method
            try :
                query = get_query ( wavelength , cali_method , day_date )
                df = query_database ( query = query , database_path = db_path )
                if df.empty :
                    raise Exception (
                        f"\n Not existing data for {station.location} station, during {day_date.strftime ( '%Y-%m-%d' )} in '{db_path}'" )
                df [ 'date' ] = day_date.strftime ( '%Y-%m-%d' )

                df = add_profiles_values ( df , station , day_date , file_type = 'profiles' )

                df = add_X_path ( df , station , day_date , lambda_nm = wavelength , data_source = 'lidar' ,
                                  file_type = 'range_corr' )
                df = add_X_path ( df , station , day_date , lambda_nm = wavelength , data_source = 'molecular' ,
                                  file_type = 'attbsc' )

                df = df.rename (
                    {'liconst' : 'LC' , 'uncertainty_liconst' : 'LC_std' , 'matched_nc_profile' : 'profile_path'} ,
                    axis = 'columns' )
                expanded_df = get_time_slots_expanded ( df , sample_size )

                # reorder the columns
                key = [ 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                        'start_time_period' , 'end_time_period' , 'profile_path' ]
                Y_features = [ 'LC' , 'LC_std' , 'r0' , 'r1' , 'dr' , 'bin_r0' , 'bin_r1' ]
                X_features = [ 'lidar_path' , 'molecular_path' ]
                expanded_df = expanded_df [ key + X_features + Y_features ]
                full_df = full_df.append ( expanded_df )
            except Exception as e :
                logger.exception ( f"{e}, skipping to next day." )
                continue

    # convert height bins to int
    full_df [ 'bin_r0' ] = full_df [ 'bin_r0' ].astype ( np.uint16 )
    full_df [ 'bin_r1' ] = full_df [ 'bin_r1' ].astype ( np.uint16 )

    return full_df.reset_index ( drop = True )


def get_query ( wavelength , cali_method , day_date ) :
    start_time = datetime.combine ( date = day_date.date ( ) , time = day_date.time ( ).min )
    end_time = datetime.combine ( date = day_date.date ( ) , time = day_date.time ( ).max )
    query = f"""
    SELECT  lcc.liconst, lcc.uncertainty_liconst,
            lcc.cali_start_time, lcc.cali_stop_time,
            lcc.wavelength, lcc.cali_method, lcc.telescope
    FROM lidar_calibration_constant as lcc
    WHERE
        wavelength == {wavelength} AND
        cali_method == '{cali_method}' AND
        telescope == 'far_range' AND
        (cali_start_time BETWEEN '{start_time}' AND '{end_time}');
    """
    return query


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
    # logger = logging.getLogger ( )

    # Connect to the db and query it directly into pandas df.
    #try:
    with sqlite3.connect ( database_path ) as c :
        # Query to df
        # optionally parse 'id' as index column and 'cali_start_time', 'cali_stop_time' as dates
        df = pd.read_sql ( sql = query , con = c , parse_dates = [ 'cali_start_time' , 'cali_stop_time' ] )
    #except sqlite3.OperationalError as e:
    #    logger.exception ( f"{e}: Unable to load dataset." )
    #    sys.exit(1)
    # TODO: raise load exception if file does not exsits. the above commented solution is not really catching this and continues to run

    return df


def add_profiles_values ( df , station , day_date , file_type = 'profiles' ) :
    logger = logging.getLogger ( )
    try :
        df [ 'matched_nc_profile' ] = df.apply ( lambda row : prep.get_TROPOS_dataset_paths ( station , day_date ,
                                                                                              start_time = row.cali_start_time ,
                                                                                              end_time = row.cali_stop_time ,
                                                                                              file_type = file_type ) [
            0 ] ,
                                                 axis = 1 , result_type = 'expand' )
    except Exception :
        logger.exception (
            f"Non resolved 'matched_nc_profile' for {station.location} station, at date {day_date.strftime ( '%Y-%m-%d' )} " )
        pass

    def _get_info_from_profile_nc ( row ) :
        """
        Get the r_0,r_1, and delta_r of the selected row. The values are following rebasing according to sea-level height.
        :param row:
        :return:
        """
        data = prep.load_dataset ( row [ 'matched_nc_profile' ] )
        wavelen = row.wavelength
        # get altitude to rebase the reference heights according to sea-level-height
        altitude = data.altitude.item ( )
        [ r0 , r1 ] = data [ f'reference_height_{wavelen}' ].values
        [ bin_r0 , bin_r1 ] = [ np.argmin ( abs ( data.height.values - r ) ) for r in [ r0 , r1 ] ]
        delta_r = r1 - r0
        return r0 + altitude , r1 + altitude , delta_r , bin_r0 , bin_r1

    df [ [ 'r0' , 'r1' , 'dr' , 'bin_r0' , 'bin_r1' ] ] = df.apply ( _get_info_from_profile_nc , axis = 1 ,
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

    paths = prep.get_prep_dataset_paths ( station = station ,
                                          day_date = day_date ,
                                          data_source = data_source ,
                                          lambda_nm = lambda_nm ,
                                          file_type = file_type )
    if not paths :
        raise Exception (
            f"\n Not exiting any '{data_source}' path for {station.location} station, at {day_date.strftime ( '%Y-%m-%d' )}" )
    elif len ( paths ) != 1 :
        raise Exception (
            f"\n Expected ONE '{data_source}' path for {station.location} station, at {day_date.strftime ( '%Y-%m-%d' )}.\nGot: {paths} " )
    df [ f"{data_source}_path" ] = paths [ 0 ]
    return df


def get_time_slots_expanded ( df , sample_size ) :
    if sample_size :
        expanded_df = pd.DataFrame ( )
        for indx , row in df.iterrows ( ) :
            time_slots = pd.date_range ( row.loc [ 'cali_start_time' ] , row.loc [ 'cali_stop_time' ] ,
                                         freq = sample_size )
            time_slots.freq = None
            if len ( time_slots ) < 2 :
                continue
            for start_time , end_time in zip ( time_slots [ :-1 ] , time_slots [ 1 : ] ) :
                mini_df = row.copy ( )
                mini_df [ 'start_time_period' ] = start_time
                mini_df [ 'end_time_period' ] = end_time
                expanded_df = expanded_df.append ( mini_df )
    else :
        expanded_df = df.copy ( )
        expanded_df [ 'start_time_period' ] = expanded_df [ 'cali_start_time' ]
        expanded_df [ 'end_time_period' ] = expanded_df [ 'cali_stop_time' ]
    return expanded_df


# %% Dataset extending helper functions
def extend_calibration_info ( df ) :
    """
    Extending the input dataset (df) by - recalculation of Lidar Constant and calculation info of reference range
    :param df: Dataset of calibrated samples for learning calibration method.
    The samples were generated by TROPOS using Pollynet_Processing_Chain.
    For more info visit here: https://github.com/PollyNET/Pollynet_Processing_Chain
    :return: Dataset of calibrated samples
    """
    logger = logging.getLogger ( )
    tqdm.pandas ( )
    logger.info ( f"Start LC calculations" )
    df [ [ 'LC_recalc' , 'LCp_recalc' ] ] = df.progress_apply ( recalc_LC , axis = 1 , result_type = 'expand' )
    logger.info ( f"Start mid-reference range calculations" )
    df [ 'bin_rm' ] = df [ [ 'bin_r0' , 'bin_r1' ] ].progress_apply ( np.mean , axis = 1 ,
                                                                      result_type = 'expand' ).astype ( int )
    df [ 'rm' ] = df [ [ 'r0' , 'r1' ] ].progress_apply ( np.mean , axis = 1 , result_type = 'expand' ).astype (
        float )
    logger.info ( f"Done database extension" )
    return df


def convert_Y_features_units ( df , Y_features = [ 'LC' , 'LC_std' , 'r0' , 'r1' , 'dr' ] ,
                               scales = {'LC' : 1E-9 , 'LC_std' : 1E-9 , 'r0' : 1E-3 , 'r1' : 1E-3 , 'dr' : 1E-3} ) :
    Y_scales = [ scales [ feature ] for feature in Y_features ]
    for feature , scale in zip ( Y_features , Y_scales ) :
        df [ feature ] *= scale
    return df


def create_calibration_ds ( df , station , source_file ) :
    """
    Create calibration dataset from the input df, by splitting it to a dataset according to wavelengths and time (as coordinates).
    :param df: extended dataset of calibrated samples
    :param station: gs.station() object of the lidar station
    :param source_file: string, the name of the extended dataset
    :return: calibration dataset for the the times in df
    """
    times = pd.to_datetime ( df [ 'start_time_period' ] )
    times = times.dt.to_pydatetime ( )
    wavelengths = df.wavelength.unique ( )
    ds_chan = [ ]
    tqdm.pandas ( )
    for wavelength in tqdm ( wavelengths ) :
        ds_cur = xr.Dataset (
            data_vars = {'LC' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'LC' ]) ,
                         'LCrecalc' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'LC_recalc' ]) ,
                         'LCprecalc' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'LCp_recalc' ]) ,
                         'r0' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'r0' ]) ,
                         'r1' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'r1' ]) ,
                         'rm' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'rm' ]) ,
                         'bin_r0' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'bin_r0' ]) ,
                         'bin_r1' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'bin_r1' ]) ,
                         'bin_rm' : (('Time') , df [ df [ 'wavelength' ] == wavelength ] [ 'bin_rm' ])} ,
            coords = {
                'Time' : times [ list ( df [ df [ 'wavelength' ] == wavelength ] [ 'LC' ].index.values ) ] ,
                'Wavelength' : np.uint16 ( [ wavelength ] )} )
        ds_cur.LC.attrs = {'units' : r'$\rm{photons\,sr\,km^3}$' , 'long_name' : r'$\rm{ LC_{TROPOS}}$' ,
                           'info' : 'LC - Lidar constant - by TROPOS'}
        ds_cur.LCrecalc.attrs = {'units' : r'$\rm{photons\,sr\,km^3}$' , 'long_name' : r'$\rm{LC_{recalc}}$' ,
                                 'info' : 'LC - Lidar constant - recalculated'}
        ds_cur.LCprecalc.attrs = {'units' : r'$\rm{photons\,sr\,km^3}$' , 'long_name' : r'$\rm{LC_{pos-recalc}}$' ,
                                  'info' : 'LC - Lidar constant - recalculated on non negative range corrected dignal'}
        ds_cur.r0.attrs = {'units' : r'$km$' , 'long_name' : r'$r_0$' , 'info' : 'Lower calibration height'}
        ds_cur.r1.attrs = {'units' : r'$km$' , 'long_name' : r'$r_1$' , 'info' : 'Upper calibration height'}
        ds_cur.rm.attrs = {'units' : r'$km$' , 'long_name' : r'$r_m$' , 'info' : 'Mid calibration height'}
        ds_cur.Wavelength.attrs = {'units' : r'$nm$' , 'long_name' : r'$\lambda$'}
        ds_chan.append ( ds_cur )

    ds_calibration = xr.concat ( ds_chan , dim = 'Wavelength' )
    # After concatenating, making sure the 'bin' values are integers.
    for bin in tqdm ( [ 'bin_r0' , 'bin_r1' , 'bin_rm' ] ) :
        ds_calibration [ bin ] = ds_calibration [ bin ].astype ( np.uint16 )
        ds_calibration [ bin ].attrs = {'long_name' : fr'${bin}$'}

    # Global info
    ds_calibration.attrs = {'info' : 'Periodic pollyXT calibration info' ,
                            'location' : station.name ,
                            'source_file' : source_file ,
                            'sample_size' : f"{(times [ 1 ] - times [ 0 ]).seconds}S" ,
                            'start_time' : times [ 0 ].strftime ( "%Y-%d-%m %H:%M:%S" ) ,
                            'end_time' : times [ -1 ].strftime ( "%Y-%d-%m %H:%M:%S" )}

    return ds_calibration


def recalc_LC ( row ) :
    """
	Extending database:by recalculate LC_recalc, LCp_recalc
	:param row:
	:return:
	"""
    sliced_ds , _ = get_sample_ds ( row )
    slice_ratio = sliced_ds [ 0 ].copy ( deep = True )
    slice_ratio.data /= sliced_ds [ 1 ].data
    LC_recalc = slice_ratio.mean ( ).values.item ( )
    LCp_recalc = slice_ratio.where ( slice_ratio >= 0.0 ).mean ( ).values.item ( )
    return LC_recalc , LCp_recalc


# %% General datasets functions
def _df_split ( tup_arg , **kwargs ) :
    split_ind , df_split , df_f_name = tup_arg
    return (split_ind , getattr ( df_split , df_f_name ) ( **kwargs ))

def df_multi_core ( df , df_f_name , subset = None , njobs = -1 , **kwargs ) :
    # %% testing multiproccesing from: https://gist.github.com/morkrispil/3944242494e08de4643fd42a76cb37ee
    if njobs == -1 :
        njobs = mp.cpu_count ( )
    pool = mp.Pool ( processes = njobs )

    try :
        df_sub = df [ subset ] if subset else df
        splits = np.array_split ( df_sub , njobs )
    except ValueError :
        splits = np.array_split ( df , njobs )

    pool_data = [ (split_ind , df_split , df_f_name) for split_ind , df_split in enumerate ( splits ) ]
    results = pool.map ( partial ( _df_split , **kwargs ) , pool_data )
    pool.close ( )
    pool.join ( )
    results = sorted ( results , key = lambda x : x [ 0 ] )
    results = pd.concat ( [ split [ 1 ] for split in results ] )
    return results

def get_sample_ds ( row ) :
    mol_path = row.molecular_path
    lidar_path = row.lidar_path
    bin_r0 = row.bin_r0
    bin_r1 = row.bin_r1
    t0 = row.start_time_period
    t1 = row.end_time_period
    full_ds = [ prep.load_dataset ( path ) for path in [ lidar_path , mol_path ] ]
    tslice = slice ( t0 , t1 )
    profiles = [ 'range_corr' , 'attbsc' ]
    sliced_ds = [ ds_i.sel ( Time = tslice ,
                             Height = slice ( float ( ds_i.Height [ bin_r0 ].values ) ,
                                              float ( ds_i.Height [ bin_r1 ].values ) ) ) [ profile ]
                  for ds_i , profile in zip ( full_ds , profiles ) ]
    return sliced_ds , full_ds

def make_interpolated_image ( nsamples , im ) :
    """Make an interpolated image from a random selection of pixels.

    Take nsamples random pixels from im and reconstruct the image using
    scipy.interpolate.griddata.

    """
    nx , ny = im.shape [ 1 ] , im.shape [ 0 ]
    X , Y = np.meshgrid ( np.arange ( 0 , nx , 1 ) , np.arange ( 0 , ny , 1 ) )
    ix = np.random.randint ( im.shape [ 1 ] , size = nsamples )
    iy = np.random.randint ( im.shape [ 0 ] , size = nsamples )
    samples = im [ iy , ix ]
    int_im = griddata ( (iy , ix) , samples , (Y , X) , method = 'nearest' , fill_value = 0 )
    return int_im

def get_aerBsc_profile_ds ( path , profile_df ) :
    cur_profile = prep.load_dataset ( path )
    height_units = 'km'
    height_scale = 1e-3  # converting [m] to [km]
    bsc_scale = 1e+3  # converting [1/m sr] to to [1/km sr]
    heights_indx = height_scale * (cur_profile.height.values + cur_profile.altitude.values)
    start_time = datetime.utcfromtimestamp ( cur_profile.start_time.values [ 0 ].tolist ( ) )
    end_time = datetime.utcfromtimestamp ( cur_profile.end_time.values [ 0 ].tolist ( ) )
    time_indx = pd.date_range ( start = start_time , end = end_time , freq = '30S' )
    profile_r0 = profile_df.groupby ( [ 'wavelength' ] ) [ 'bin_r0' ].unique ( ).astype ( np.int ).values
    profile_r1 = profile_df.groupby ( [ 'wavelength' ] ) [ 'bin_r0' ].unique ( ).astype ( np.int ).values
    ds_chans = [ ]

    var_names = [ f"aerBsc_klett_{wavelength}" for wavelength in [ 355 , 532 , 1064 ] ]
    for v_name , wavelength , r in zip ( var_names , [ 355 , 532 , 1064 ] , profile_r1 ) :
        vals = cur_profile [ v_name ].values
        vals [ r : ] = eps
        vals = gaussian_filter1d ( vals , 21 , mode = 'nearest' )
        vals = vals.T.reshape ( len ( heights_indx ) , 1 )
        vals [ vals < 0 ] = eps
        vals *= bsc_scale
        mat_vals = np.repeat ( vals , len ( time_indx ) , axis = 1 )
        aerbsc_df = pd.DataFrame ( mat_vals , index = heights_indx , columns = time_indx )
        aerBsc_ds_chan = xr.Dataset (
            data_vars = {'aerBsc' : (('Height' , 'Time') , aerbsc_df) ,
                         'lambda_nm' : ('Wavelength' , [ wavelength ])
                         } ,
            coords = {'Height' : aerbsc_df.index.to_list ( ) ,
                      'Time' : aerbsc_df.columns ,
                      'Wavelength' : [ wavelength ]
                      } )
        aerBsc_ds_chan.aerBsc.attrs = {'long_name' : r'$\beta$' ,  # _{{a}}$' ,
                                       'units' : r'$km^{{-1}} sr^{-1}$' ,
                                       'info' : r'$Aerosol backscatter$'}
        # set attributes of coordinates
        aerBsc_ds_chan.Height.attrs = {'units' : '{}'.format ( '{}'.format ( height_units ) ) ,
                                       'info' : 'Measurements heights above sea level'}
        aerBsc_ds_chan.Wavelength.attrs = {'long_name' : r'$\lambda$' , 'units' : r'$nm$'}
        ds_chans.append ( aerBsc_ds_chan )
    return xr.concat ( ds_chans , dim = 'Wavelength' )

# %% MAIN
def main ( station_name , start_date , end_date ) :
    logging.getLogger ( 'matplotlib' ).setLevel ( logging.ERROR )  # Fix annoying matplotlib logs
    logging.getLogger ( 'PIL' ).setLevel ( logging.ERROR )  # Fix annoying PIL logs
    logger = create_and_configer_logger ( 'dataseting_log.log' , level = logging.INFO )

    # set operating mode
    DO_DATASET = False
    EXTEND_DATASET = True
    DO_CALIB_DATASET = True
    USE_KM_UNITS = True

    # Load data of station
    station = gs.Station ( stations_csv_path = 'stations.csv' , station_name = station_name )
    logger.info ( f"Loading {station.location} station" )

    # Set new paths
    csv_path = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}.csv"
    csv_path_extended = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}_extended.csv"
    ds_path_extended = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}_extended.nc"

    if DO_DATASET :
        logger.info (
            f"Start generating dataset for period: [{start_date.strftime ( '%Y-%m-%d' )},{end_date.strftime ( '%Y-%m-%d' )}]" )
        # Generate dataset for learning
        sample_size = '29.5min'
        df = create_dataset ( station_name = station_name , start_date = start_date ,
                              end_date = end_date , sample_size = sample_size )

        # Convert m to km
        if USE_KM_UNITS :
            df = convert_Y_features_units ( df )

        # TODO: Split the dataset to train, and test, and add a key (column) to each entry to indicate if this is train or test

        df.to_csv ( csv_path , index = False )
        logger.info ( f"Done database creation, saving to: {os.path.join ( os.path.curdir , csv_path )}" )

    # Create extended dataset and save to csv - recalculation of Lidar Constant (currently this is a naive calculation)
    if EXTEND_DATASET :
        logger.info (
            f"Start extending dataset for period: [{start_date.strftime ( '%Y-%m-%d' )},{end_date.strftime ( '%Y-%m-%d' )}]" )
        if 'df' not in globals ( ) :
            df = pd.read_csv ( csv_path )
        df = extend_calibration_info ( df )
        display ( df )
        df.to_csv ( csv_path_extended , index = False )
        logger.info ( f"\n The extended dataset saved to :{os.path.join ( os.path.curdir , csv_path_extended )}" )

    # Create calibration dataset from the extended df (or csv) = by spliting df to A dataset according to wavelengths and time coordinates.
    if DO_CALIB_DATASET :
        logger.info (
            f"Start creating calibration dataset for period: [{start_date.strftime ( '%Y-%m-%d' )},{end_date.strftime ( '%Y-%m-%d' )}]" )

        if 'df' not in globals ( ) :
            df = pd.read_csv ( csv_path_extended )
        ds_calibration = create_calibration_ds ( df , station , source_file = csv_path_extended )
        # Save the calibration dataset
        prep.save_dataset ( ds_calibration , os.path.curdir , ds_path_extended )
        logger.info ( f"The calibration dataset saved to :{os.path.join ( os.path.curdir , ds_path_extended )}" )


if __name__ == '__main__' :
    station_name = 'haifa'
    start_date = datetime ( 2017 , 9 , 1 )
    end_date = datetime ( 2017 , 10 , 31 )
    main ( station_name , start_date , end_date )
