# %% Imports

import glob
import logging
import os
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

import learning_lidar.preprocessing.preprocessing_utils as prep_utils
import learning_lidar.utils.global_settings as gs
from learning_lidar.preprocessing.fix_gdas_errors import download_from_noa_gdas_files
from learning_lidar.utils.utils import create_and_configer_logger, get_base_arguments
from learning_lidar.utils.xr_utils import save_prep_dataset


def preprocessing_main(args):
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
    logger.info(args)
    station_name = args.station_name
    start_date = args.start_date
    end_date = args.end_date

    DOWNLOAD_GDAS = False
    CONV_GDAS = False
    GEN_MOL_DS = False
    GEN_LIDAR_DS = False
    GEN_LIDAR_DS_RAW = True
    USE_KM_UNITS = True
    UNZIP_TROPOS_LIDAR = True

    """set day,location"""
    station = gs.Station(station_name=station_name)
    logger.info(f"Loading {station.location} station")
    logger.debug(f"Station info: {station}")
    logger.info(
        f"\nStart preprocessing for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Generate molecular datasets for required period'''
    if DOWNLOAD_GDAS:
        # TODO Test that this works as expected
        download_from_noa_gdas_files(
            dates_to_retrieve=pd.date_range(start=start_date, end=end_date, freq=timedelta(days=1)),
            save_folder=station.gdas1_folder)

    gdas_paths = []
    if CONV_GDAS:
        # Convert gdas files for a period
        gdas_paths.extend(prep_utils.convert_periodic_gdas(station, start_date, end_date))

    # get all days having a converted (to txt) gdas files in the required period
    if (GEN_MOL_DS or GEN_LIDAR_DS or GEN_LIDAR_DS_RAW) and not gdas_paths:
        logger.debug('\nGet all days in the required period that have a converted gdas file')
        dates = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()
        for day in tqdm(dates):
            _, curpath = prep_utils.get_daily_gdas_paths(station, day, f_type='txt')
            if curpath:
                gdas_paths.extend(curpath)
        timestamps = [prep_utils.get_gdas_timestamp(station, path) for path in gdas_paths]
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
        chunksize = np.ceil(float(len_mol_days) / num_processes).astype(int)
        # TODO: add here tqdm
        with Pool(num_processes) as p:
            p.map(prep_utils.gen_daily_molecular_ds, valid_gdas_days, chunksize=chunksize)

        logger.info(
            f"\nFinished generating and saving of molecular datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Extract all zip files of raw signals from TROPOS'''
    if UNZIP_TROPOS_LIDAR:
        """
        Expects all zip files to be in station.lidar_src_folder, and saves them under the appropriate year/month/day
        """
        # TODO Delete zip files after extraction
        path_pattern = os.path.join(station.lidar_src_folder, '*.zip')
        for file_name in sorted(glob.glob(path_pattern)):
            # extract day of the file. 0 - year, 1 - month, 2 - day
            nc_year, nc_month, nc_day = os.path.basename(file_name).split("_")[0:3]
            save_path = os.path.join(station.lidar_src_folder, nc_year, nc_month, nc_day)

            logger.info(f"extracting {file_name} to {save_path}")
            ZipFile(file_name).extractall(save_path)

    ''' Generate lidar datasets for required period'''
    if GEN_LIDAR_DS:
        lidarpaths = []
        logger.info(
            f"\nStart generating lidar datasets for period [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        for day_date in tqdm(valid_gdas_days):
            # Generate daily range corrected
            lidar_ds = prep_utils.get_daily_range_corr(station, day_date, height_units='km', optim_size=False, verbose=False,
                                            USE_KM_UNITS=USE_KM_UNITS)

            # Save lidar dataset
            lidar_paths = save_prep_dataset(station, lidar_ds, data_source='lidar', save_mode='single',
                                            profiles=['range_corr'])
            lidarpaths.extend(lidar_paths)
        logger.info(
            f"\nDone creation of lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        logger.debug(f'\nLidar paths: {lidarpaths}')
    ''' Obtaining lidar datasets for required period'''
    if GEN_LIDAR_DS_RAW:
        lidarpaths = []
        logger.info(
            f"\nStart obtaining raw lidar datasets for period [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        for day_date in tqdm(valid_gdas_days):
            # Generate daily range corrected
            # TODO delete the raw separted lidar measurements after creating the merge dataset
            lidar_ds = prep_utils.get_daily_measurements(station, day_date, use_km_units=USE_KM_UNITS)

            # Save lidar dataset
            lidar_paths = save_prep_dataset(station, lidar_ds, data_source='lidar', save_mode='single')
            lidarpaths.extend(lidar_paths)
        logger.info(
            f"\nDone creation of lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        logger.debug(f'\nLidar paths: {lidarpaths}')


if __name__ == '__main__':
    parser = get_base_arguments()
    args = parser.parse_args()

    preprocessing_main(args)
    # TODO: Add flags as args.
