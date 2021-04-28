# %% #General Imports
import os
from datetime import datetime, timedelta  # time,
from IPython.display import display
import logging
from tqdm import tqdm
from pytictoc import TicToc

from itertools import repeat
from multiprocessing import Pool, cpu_count

# %% Scientific and data imports
import numpy as np
import pandas as pd
import xarray as xr

# %% Local modules imports
import learning_lidar.utils.global_settings as gs
import learning_lidar.preprocessing.preprocessing as prep
from learning_lidar.dataseting.dataseting_utils import get_query, query_database, add_profiles_values, add_X_path, \
    get_time_slots_expanded, convert_Y_features_units, recalc_LC, split_save_train_test_ds, get_generated_X_path, \
    get_mean_lc
from learning_lidar.utils.utils import create_and_configer_logger


# %% Dataset creating helper functions
def create_dataset(station_name='haifa', start_date=datetime(2017, 9, 1),
                   end_date=datetime(2017, 9, 2), sample_size='29.5min', list_dates=[]):
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
    logger = logging.getLogger()
    station = gs.Station(station_name=station_name)
    db_path = station.db_file
    wavelengths = gs.LAMBDA_nm().get_elastic()
    cali_method = 'Klett_Method'

    if len(list_dates) > 0:
        try:
            dates = [datetime.combine(t.date(), t.time()) for t in list_dates]
        except TypeError as e:
            logger.exception(f"{e}: `list_dates`, when not empty, is excepted to have datetime.datetime() objects.")
            return
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()

    full_df = pd.DataFrame()
    for wavelength in tqdm(wavelengths):
        for day_date in dates:  # tqdm(dates)

            # Query the db for a specific day, wavelength and calibration method
            try:
                query = get_query(wavelength, cali_method, day_date)
                df = query_database(query=query, database_path=db_path)
                if df.empty:
                    raise Exception(
                        f"\n Not existing data for {station.location} station, during {day_date.strftime('%Y-%m-%d')} in '{db_path}'")
                df['date'] = day_date.strftime('%Y-%m-%d')

                df = add_profiles_values(df, station, day_date, file_type='profiles')

                df = add_X_path(df, station, day_date, lambda_nm=wavelength, data_source='lidar',
                                file_type='range_corr')
                df = add_X_path(df, station, day_date, lambda_nm=wavelength, data_source='molecular',
                                file_type='attbsc')

                df = df.rename(
                    {'liconst': 'LC', 'uncertainty_liconst': 'LC_std', 'matched_nc_profile': 'profile_path'},
                    axis='columns')
                expanded_df = get_time_slots_expanded(df, sample_size)

                # reorder the columns
                key = ['date', 'wavelength', 'cali_method', 'telescope', 'cali_start_time', 'cali_stop_time',
                       'start_time_period', 'end_time_period', 'profile_path']
                Y_features = ['LC', 'LC_std', 'r0', 'r1', 'dr', 'bin_r0', 'bin_r1']
                X_features = ['lidar_path', 'molecular_path']
                expanded_df = expanded_df[key + X_features + Y_features]
                full_df = full_df.append(expanded_df)
            except Exception as e:
                logger.exception(f"{e}, skipping to next day.")
                continue

    # convert height bins to int
    full_df['bin_r0'] = full_df['bin_r0'].astype(np.uint16)
    full_df['bin_r1'] = full_df['bin_r1'].astype(np.uint16)

    return full_df.reset_index(drop=True)


# %% Dataset extending helper functions
def extend_calibration_info(df):
    """
    Extending the input dataset (df) by - recalculation of Lidar Constant and calculation info of reference range
    :param df: pd.DataFrame(). Dataset of calibrated samples for learning calibration method.
    The samples were generated by TROPOS using Pollynet_Processing_Chain.
    For more info visit here: https://github.com/PollyNET/Pollynet_Processing_Chain
    :return: Dataset of calibrated samples
    """
    logger = logging.getLogger()
    tqdm.pandas()
    logger.info(f"Start LC calculations")
    df[['LC_recalc', 'LCp_recalc']] = df.progress_apply(recalc_LC, axis=1, result_type='expand')
    logger.info(f"Start mid-reference range calculations")
    df['bin_rm'] = df[['bin_r0', 'bin_r1']].progress_apply(np.mean, axis=1,
                                                           result_type='expand').astype(int)
    df['rm'] = df[['r0', 'r1']].progress_apply(np.mean, axis=1, result_type='expand').astype(
        float)
    logger.info(f"Done database extension")
    return df


def create_calibration_ds(df, station, source_file):
    """
    Create calibration dataset from the input df, by splitting it to a dataset according to wavelengths and time (as coordinates).
    :param df: extended dataset of calibrated samples
    :param station: gs.station() object of the lidar station
    :param source_file: string, the name of the extended dataset
    :return: calibration dataset for the the times in df
    """
    times = pd.to_datetime(df['start_time_period'])
    times = times.dt.to_pydatetime()
    wavelengths = df.wavelength.unique()
    ds_chan = []
    tqdm.pandas()
    for wavelength in tqdm(wavelengths):
        ds_cur = xr.Dataset(
            data_vars={'LC': (('Time'), df[df['wavelength'] == wavelength]['LC']),
                       'LCrecalc': (('Time'), df[df['wavelength'] == wavelength]['LC_recalc']),
                       'LCprecalc': (('Time'), df[df['wavelength'] == wavelength]['LCp_recalc']),
                       'r0': (('Time'), df[df['wavelength'] == wavelength]['r0']),
                       'r1': (('Time'), df[df['wavelength'] == wavelength]['r1']),
                       'rm': (('Time'), df[df['wavelength'] == wavelength]['rm']),
                       'bin_r0': (('Time'), df[df['wavelength'] == wavelength]['bin_r0']),
                       'bin_r1': (('Time'), df[df['wavelength'] == wavelength]['bin_r1']),
                       'bin_rm': (('Time'), df[df['wavelength'] == wavelength]['bin_rm'])},
            coords={
                'Time': times[list(df[df['wavelength'] == wavelength]['LC'].index.values)],
                'Wavelength': np.uint16([wavelength])})
        ds_cur.LC.attrs = {'units': r'$\rm{photons\,sr\,km^3}$', 'long_name': r'$\rm{ LC_{TROPOS}}$',
                           'info': 'LC - Lidar constant - by TROPOS'}
        ds_cur.LCrecalc.attrs = {'units': r'$\rm{photons\,sr\,km^3}$', 'long_name': r'$\rm{LC_{recalc}}$',
                                 'info': 'LC - Lidar constant - recalculated'}
        ds_cur.LCprecalc.attrs = {'units': r'$\rm{photons\,sr\,km^3}$', 'long_name': r'$\rm{LC_{pos-recalc}}$',
                                  'info': 'LC - Lidar constant - recalculated on non negative range corrected dignal'}
        ds_cur.r0.attrs = {'units': r'$km$', 'long_name': r'$r_0$', 'info': 'Lower calibration height'}
        ds_cur.r1.attrs = {'units': r'$km$', 'long_name': r'$r_1$', 'info': 'Upper calibration height'}
        ds_cur.rm.attrs = {'units': r'$km$', 'long_name': r'$r_m$', 'info': 'Mid calibration height'}
        ds_cur.Wavelength.attrs = {'units': r'$nm$', 'long_name': r'$\lambda$'}
        ds_chan.append(ds_cur)

    ds_calibration = xr.concat(ds_chan, dim='Wavelength')
    # After concatenating, making sure the 'bin' values are integers.
    for bin in tqdm(['bin_r0', 'bin_r1', 'bin_rm']):
        ds_calibration[bin] = ds_calibration[bin].astype(np.uint16)
        ds_calibration[bin].attrs = {'long_name': fr'${bin}$'}

    # Global info
    ds_calibration.attrs = {'info': 'Periodic pollyXT calibration info',
                            'location': station.name,
                            'source_file': source_file,
                            'sample_size': f"{(times[1] - times[0]).seconds}S",
                            'start_time': times[0].strftime("%Y-%d-%m %H:%M:%S"),
                            'end_time': times[-1].strftime("%Y-%d-%m %H:%M:%S")}

    return ds_calibration


def create_generated_dataset(station, start_date, end_date, sample_size='30min'):
    """
    Creates a dataframe consisting of:
        date | wavelength | start_time | end_time | lidar_path | bg_path | molecular_path | LC
    :param station:
    :param start_date:
    :param end_date:
    :param sample_size:
    :return:
    """
    dates = pd.date_range(start=start_date,
                          end=end_date + timedelta(days=1),  # include all times in the last day
                          freq=sample_size)[0:-1]            # excludes last - the next day after end_date
    sample_size = int(sample_size.split('min')[0])
    wavelengths = gs.LAMBDA_nm().get_elastic()
    gen_df = pd.DataFrame()
    for wavelength in tqdm(wavelengths):
        try:
            df = pd.DataFrame()
            df['date'] = dates.date                     # new row for each time
            df['wavelength'] = wavelength               # broadcast single value to all rows
            df['start_time'] = dates                    # start_time - start time of the sample, end_time 29.5 min later
            df['end_time'] = dates + timedelta(minutes=sample_size) - timedelta(seconds=station.freq)

            # add bg path
            df['bg_path'] = df.apply(
                lambda row: get_generated_X_path(station=station, parent_folder=station.gen_bg_dataset,
                                                 day_date=row['start_time'], data_source='bg', wavelength=wavelength,
                                                 file_type='p_bg'),
                axis=1, result_type='expand')
            # add lidar path
            df['lidar_path'] = df.apply(
                lambda row: get_generated_X_path(station=station, parent_folder=station.gen_lidar_dataset,
                                                 day_date=row['start_time'], data_source='lidar', wavelength=wavelength,
                                                 file_type='range_corr'),
                axis=1, result_type='expand')

            # get the mean LC from signal_paths, one day at a time
            # TODO: adapt this part and get_mean_lc() to retrieve mean LC for 3 wavelength at once
            #  (should be faster then reading the file 3 times per wavelength)
            days_groups = df.groupby('date').groups
            days_list = days_groups.keys()
            num_processes = min((cpu_count() - 1, len(days_list)))
            inds_subsets = [days_groups[key] for key in days_list]
            df_subsets = [df.iloc[inds] for inds in inds_subsets]
            with Pool(num_processes) as p:
                results = p.starmap(get_mean_lc, zip(df_subsets, repeat(station), days_list))
            df = pd.concat([subset for subset in results]).sort_values('start_time')
            # Add molecular path
            for cur_day in days_list:
                df = add_X_path(df, station, cur_day, lambda_nm=wavelength, data_source='molecular', file_type='attbsc')

            gen_df = gen_df.append(df)
        except FileNotFoundError as e:
            print(e)
    return gen_df


def main(station_name, start_date, end_date):
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logger = create_and_configer_logger('dataseting_log.log', level=logging.INFO)

    # set operating mode
    DO_DATASET = False
    EXTEND_DATASET = False
    DO_CALIB_DATASET = False
    USE_KM_UNITS = False
    SPLIT_DATASET = True
    DO_GENERATED_DATASET = False

    # Load data of station
    station = gs.Station(station_name=station_name)
    logger.info(f"Loading {station.location} station")

    # Set new paths
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    csv_path = os.path.join(data_folder, f"dataset_{station_name}_"
                                        f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")
    csv_path_extended = os.path.join(data_folder, f"dataset_{station_name}_"
                                                  f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.csv")
    ds_path_extended = os.path.join(data_folder, f"dataset_{station_name}_"
                                                 f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.nc")

    csv_gen_path = os.path.join(data_folder, f"dataset_gen_{station_name}_"
                                             f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")

    if DO_DATASET:
        logger.info(
            f"Start generating dataset for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
        # Generate dataset for learning
        sample_size = '29.5min'
        df = create_dataset(station_name=station_name, start_date=start_date,
                            end_date=end_date, sample_size=sample_size)

        # Convert m to km
        if USE_KM_UNITS:
            df = convert_Y_features_units(df)

        df.to_csv(csv_path, index=False)
        logger.info(f"Done database creation, saving to: {csv_path}")

    if SPLIT_DATASET:
        logger.info(
            f"Splitting to train-test the dataset for period: [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")
        split_save_train_test_ds(csv_path=csv_gen_path)

    # Create extended dataset and save to csv - recalculation of Lidar Constant (currently this is a naive calculation)
    if EXTEND_DATASET:
        logger.info(
            f"Start extending dataset for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        df = extend_calibration_info(df)
        display(df)
        df.to_csv(csv_path_extended, index=False)
        logger.info(f"\n The extended dataset saved to :{csv_path_extended}")

    # Create calibration dataset from the extended df (or csv) = by splitting df to A dataset according to wavelengths
    # and time coordinates.
    if DO_CALIB_DATASET:
        logger.info(
            f"Start creating calibration dataset for period: [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        if 'df' not in globals():
            df = pd.read_csv(csv_path_extended)
        ds_calibration = create_calibration_ds(df, station, source_file=csv_path_extended)
        # Save the calibration dataset
        prep.save_dataset(ds_calibration, os.path.curdir, ds_path_extended)
        logger.info(f"The calibration dataset saved to :{ds_path_extended}")

    if DO_GENERATED_DATASET:
        generated_df = create_generated_dataset(station, start_date, end_date)
        generated_df.to_csv(csv_gen_path, index=False)
        logger.info(f"\n The generation dataset saved to :{csv_gen_path}")



if __name__ == '__main__':
    station_name = 'haifa'
    start_date = datetime(2017, 9, 1)
    end_date = datetime(2017, 10, 31)
    main(station_name, start_date, end_date)
