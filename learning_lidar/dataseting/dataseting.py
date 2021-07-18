# %% #General Imports
import os
from datetime import datetime, timedelta
from IPython.display import display
import logging
from tqdm import tqdm

from itertools import repeat
from multiprocessing import Pool, cpu_count

# %% Scientific and data imports
import numpy as np
import pandas as pd
import xarray as xr

# %% Local modules imports
import learning_lidar.preprocessing.preprocessing_utils as prep_utils
import learning_lidar.utils.global_settings as gs
import learning_lidar.dataseting.dataseting_utils as ds_utils
import learning_lidar.utils.xr_utils as xr_utils
from learning_lidar.generation.daily_signals_generations_utils import get_daily_bg
from learning_lidar.utils.utils import create_and_configer_logger, get_base_arguments
import learning_lidar.generation.generation_utils as gen_utils


def dataseting_main(args, log_level=logging.DEBUG):
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=log_level)
    logger.info(args)
    station_name = args.station_name
    start_date = args.start_date
    end_date = args.end_date

    # Load data of station
    station = gs.Station(station_name=station_name)

    # Set new paths
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    csv_path = os.path.join(data_folder, f"dataset_{station_name}_"
                                         f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")
    csv_path_extended = os.path.join(data_folder, f"dataset_{station_name}_"
                                                  f"{start_date.strftime('%Y-%m-%d')}_"
                                                  f"{end_date.strftime('%Y-%m-%d')}_extended.csv")
    ds_path_extended = os.path.join(data_folder, f"dataset_{station_name}_"
                                                 f"{start_date.strftime('%Y-%m-%d')}_"
                                                 f"{end_date.strftime('%Y-%m-%d')}_extended.nc")

    csv_gen_path = os.path.join(data_folder, f"dataset_gen_{station_name}_"
                                             f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")

    if args.DO_DATASET:
        logger.info(
            f"\nStart generating dataset for period: "
            f"[{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
        # Generate dataset for learning
        sample_size = '29.5min'
        df = create_dataset(station_name=station_name, start_date=start_date,
                            end_date=end_date, sample_size=sample_size)

        # Convert m to km
        if args.USE_KM_UNITS:
            df = ds_utils.convert_Y_features_units(df)

        df.to_csv(csv_path, index=False)
        logger.info(f"\nDone database creation, saving to: {csv_path}")
        # TODO: add creation of dataset statistics following its creation.
        #         adapt calc_data_statistics(station, start_date, end_date)

    if args.create_train_test_splits:
        logger.info(
            f"Splitting to train-test the dataset for period: [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")
        ds_utils.split_save_train_test_ds(csv_path=csv_path)
        logger.info(f"Done splitting train-test dataset for {csv_path}")

    # Create extended dataset and save to csv - recalculation of Lidar Constant (currently this is a naive calculation)
    if args.EXTEND_DATASET:
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
    if args.do_calibration_dataset:
        logger.info(
            f"Start creating calibration dataset for period: [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        if 'df' not in globals():
            df = pd.read_csv(csv_path_extended)
        ds_calibration = create_calibration_ds(df, station, source_file=csv_path_extended)
        # Save the calibration dataset
        xr_utils.save_dataset(ds_calibration, os.path.curdir, ds_path_extended)
        logger.info(f"The calibration dataset saved to :{ds_path_extended}")

    if args.do_generated_dataset:
        logger.info(
            f"\nStart creating generated dataset for period:"
            f" [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
        generated_df = create_generated_dataset(station=station, start_date=start_date, end_date=end_date)
        generated_df.to_csv(csv_gen_path, index=False)
        logger.info(f"\nDone generated database creation, saving to: {csv_gen_path}")

    if args.SPLIT_GENERATED_DATASET:
        logger.info(
            f"\nSplitting to train-test the dataset for period: [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")
        ds_utils.split_save_train_test_ds(csv_path=csv_gen_path)
        logger.info(f"\nDone splitting train-test dataset for {csv_gen_path}")

    if args.create_generated_time_split_samples:
        prepare_samples(station, start_date, end_date, top_height=15.3, generated=True)

    if args.create_time_split_samples:
        prepare_samples(station, start_date, end_date, top_height=15.3, generated=False)

    if args.CALC_GENERATED_STATS:
        logger.info(f"\nStart calculating generated dataset statistics")
        _, csv_stats_path = calc_data_statistics(station, start_date, end_date)
        logger.info(f"\nDone calculating generated dataset statistics. saved to:{csv_stats_path}")

    if args.CALC_STATS:
        logger.info(f"\nStart calculating dataset statistics")
        _, csv_stats_path = calc_data_statistics(station, start_date, end_date, mode='raw')
        logger.info(f"\nDone calculating dataset statistics. saved to:{csv_stats_path}")


# %% Dataset creating helper functions
def create_dataset(station_name='haifa', start_date=datetime(2017, 9, 1),
                   end_date=datetime(2017, 9, 2), sample_size='29.5min', list_dates=[]):
    """
    CHOOSE: telescope: far_range , METHOD: Klett_Method
    each sample will have 60 bins (aka 30 mins length)
    path to db:  stationdb_file
    TODO update usage
    :param list_dates:
    :param sample_size:
    :param station_name:
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
    lidar_path (station.lidar_src_calib_folder/<%Y/%M/%d> +nc_zip_file+'att_bsc.nc'
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
        for day_date in dates:

            # Query the db for a specific day, wavelength and calibration method
            try:
                query = ds_utils.get_query(wavelength, cali_method, day_date)
                df = ds_utils.query_database(query=query, database_path=db_path)
                if df.empty:
                    raise ds_utils.EmptyDataFrameError(
                        f"\n Not existing data for {station.location} station, during {day_date.strftime('%Y-%m-%d')} in '{db_path}'")
                df['date'] = day_date.strftime('%Y-%m-%d')

                df = ds_utils.add_profiles_values(df, station, day_date, file_type='profiles')

                df = df.rename(
                    {'liconst': 'LC', 'uncertainty_liconst': 'LC_std', 'matched_nc_profile': 'profile_path'},
                    axis='columns')

                expanded_df = ds_utils.get_time_slots_expanded(df, sample_size)

                # Add molecular path
                expanded_df['molecular_path'] = expanded_df.apply(
                    lambda row: ds_utils.get_prep_X_path(station=station, parent_folder=station.molecular_dataset,
                                                         day_date=row['start_time_period'], data_source='molecular',
                                                         wavelength=wavelength, file_type='attbsc',
                                                         time_slice=slice(row['start_time_period'],
                                                                          row['end_time_period'])),
                    axis=1, result_type='expand')

                # Add bg path
                expanded_df['bg_path'] = expanded_df.apply(
                    lambda row: ds_utils.get_prep_X_path(station=station, parent_folder=station.bg_dataset,
                                                         day_date=row['start_time_period'], data_source='bg',
                                                         wavelength=wavelength, file_type='p_bg',
                                                         time_slice=slice(row['start_time_period'],
                                                                          row['end_time_period'])),
                    axis=1, result_type='expand')

                # Add lidar path
                expanded_df['lidar_path'] = expanded_df.apply(
                    lambda row: ds_utils.get_prep_X_path(station=station, parent_folder=station.lidar_dataset,
                                                         day_date=row['start_time_period'], data_source='lidar',
                                                         wavelength=wavelength, file_type='range_corr',
                                                         time_slice=slice(row['start_time_period'],
                                                                          row['end_time_period'])),
                    axis=1, result_type='expand')

                # reorder the columns
                key = ['date', 'wavelength', 'cali_method', 'telescope', 'cali_start_time', 'cali_stop_time',
                       'start_time_period', 'end_time_period', 'profile_path']
                y_features = ['LC', 'LC_std', 'r0', 'r1', 'dr', 'bin_r0', 'bin_r1']
                x_features = ['lidar_path', 'molecular_path', 'bg_path']
                expanded_df = expanded_df[key + x_features + y_features]
                full_df = full_df.append(expanded_df)
            except ds_utils.EmptyDataFrameError as e:
                logger.exception(f"{e}, skipping to next day.")
                continue
    if full_df.empty:
        raise Exception("Empty Dataframe. No days retrieved!. Stopping dataseting.")
    # convert height bins to int
    full_df['bin_r0'] = full_df['bin_r0'].astype(np.uint16)
    full_df['bin_r1'] = full_df['bin_r1'].astype(np.uint16)

    return full_df.reset_index(drop=True)


def timestamp2datetime(time_stamp):
    return datetime.combine(time_stamp.date(), time_stamp.time())


def get_time_splits(station, start_date, end_date, sample_size):
    sample_start = pd.date_range(start=start_date,
                                 end=end_date + timedelta(days=1),
                                 freq=sample_size)[0:-1]
    sample_end = pd.date_range(start=start_date,
                               end=end_date + timedelta(days=1),
                               freq=sample_size)[1:]
    sample_end -= timedelta(seconds=station.freq)
    time_slices = [slice(sample_s, sample_e) for sample_s, sample_e in zip(sample_start, sample_end)]
    return time_slices


# %% Dataset extending helper functions
def extend_calibration_info(df):
    """
    Extending the input dataset (df) by - recalculation of Lidar Constant and calculation info of reference range
    :param df: pd.DataFrame(). Dataset of calibrated samples for learning calibration method.
    The samples were generated by TROPOS using Pollynet_Processing_Chain.
    For more info visit here: https://github.com/PollyNET/Pollynet_Processing_Chain
    :return: df: pd.Dataframe(). A Dataset of calibrated samples
    """
    logger = logging.getLogger()
    tqdm.pandas()
    logger.info(f"\nStart LC calculations")
    try:
        df[['LC_recalc', 'LCp_recalc']] = df.progress_apply(ds_utils.recalc_LC, axis=1, result_type='expand')
    except ValueError:
        logger.debug("Issue with ds_utils.recalc_LC. Possibly due to missing lidar/molecular paths. Inserting NaNs")
        df[['LC_recalc', 'LCp_recalc']] = None
    logger.info(f"Start mid-reference range calculations")
    df['bin_rm'] = df[['bin_r0', 'bin_r1']].progress_apply(np.mean, axis=1,
                                                           result_type='expand').astype(int)
    df['rm'] = df[['r0', 'r1']].progress_apply(np.mean, axis=1, result_type='expand').astype(
        float)

    logger.info(f"\nDone database extension")
    return df


def create_calibration_ds(df, station, source_file):
    """
    Create calibration dataset from the input df,
    by splitting it to a dataset according to wavelengths and time (as coordinates).

    :param df: extended dataset of calibrated samples
    :param station: gs.station() object of the lidar station
    :param source_file: string, the name of the extended dataset
    :return: ds_calibration : xr.Dataset(). A calibration dataset for the the times in df
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
    for bin_ in tqdm(['bin_r0', 'bin_r1', 'bin_rm']):
        ds_calibration[bin_] = ds_calibration[bin_].astype(np.uint16)
        ds_calibration[bin_].attrs = {'long_name': fr'${bin_}$'}

    # Global info
    ds_calibration.attrs = {'info': 'Periodic pollyXT calibration info',
                            'location': station.name,
                            'source_file': source_file,
                            'sample_size': f"{(times[1] - times[0]).seconds}S",
                            'start_time': times[0].strftime("%Y-%d-%m %H:%M:%S"),
                            'end_time': times[-1].strftime("%Y-%d-%m %H:%M:%S")}

    return ds_calibration


def create_generated_dataset(station, start_date, end_date, sample_size='30min', calc_mean_lc=True):
    """
    Creates a dataframe consisting of:
        date | wavelength | start_time | end_time | lidar_path | bg_path | molecular_path | signal_path |  LC
    :param calc_mean_lc: bool, whether to calculate the mean LC
    :param station: gs.station() object of the lidar station
    :param start_date: datetime.date object of the initial period date
    :param end_date: datetime.date object of the end period date
    :param sample_size: string. the sample size to create in the dataset. e.g, '30min' or '60min'
    :return: gen_df : pd.Dataframe(). A Database of generated samples for the learning phase.
    """
    logger = logging.getLogger()
    dates = pd.date_range(start=start_date,
                          end=end_date + timedelta(days=1),  # include all times in the last day
                          freq=sample_size)[0:-1]  # excludes last - the next day after end_date
    sample_size = int(sample_size.split('min')[0])
    wavelengths = gs.LAMBDA_nm().get_elastic()
    gen_df = pd.DataFrame()
    for wavelength in tqdm(wavelengths):
        try:
            df = pd.DataFrame()
            df['date'] = dates.date  # new row for each time
            df['wavelength'] = wavelength  # broadcast single value to all rows
            df['start_time_period'] = dates  # start_time - start time of the sample, end_time 29.5 min later
            df['end_time_period'] = dates + timedelta(minutes=sample_size) - timedelta(seconds=station.freq)

            # add bg path
            df['bg_path'] = df.apply(
                lambda row: ds_utils.get_generated_X_path(station=station, parent_folder=station.gen_bg_dataset,
                                                          day_date=row['start_time_period'], data_source='bg',
                                                          wavelength=wavelength, file_type='p_bg',
                                                          time_slice=slice(row['start_time_period'],
                                                                           row['end_time_period'])),
                axis=1, result_type='expand')

            # add lidar path
            df['lidar_path'] = df.apply(
                lambda row: ds_utils.get_generated_X_path(station=station, parent_folder=station.gen_lidar_dataset,
                                                          day_date=row['start_time_period'], data_source='lidar',
                                                          wavelength=wavelength, file_type='range_corr',
                                                          time_slice=slice(row['start_time_period'],
                                                                           row['end_time_period'])),
                axis=1, result_type='expand')

            # add signal path
            df['signal_path'] = df.apply(
                lambda row: ds_utils.get_generated_X_path(station=station, parent_folder=station.gen_signal_dataset,
                                                          day_date=row['start_time_period'], data_source='signal',
                                                          wavelength=wavelength, file_type='range_corr',
                                                          time_slice=slice(row['start_time_period'],
                                                                           row['end_time_period'])),
                axis=1, result_type='expand')

            # add signal path - poisson without bg
            df['signal_p_path'] = df.apply(
                lambda row: ds_utils.get_generated_X_path(station=station, parent_folder=station.gen_signal_dataset,
                                                          day_date=row['start_time_period'], data_source='signal',
                                                          wavelength=wavelength, file_type='range_corr_p',
                                                          time_slice=slice(row['start_time_period'],
                                                                           row['end_time_period'])),
                axis=1, result_type='expand')

            # Add molecular path
            df['molecular_path'] = df.apply(
                lambda row: ds_utils.get_prep_X_path(station=station, parent_folder=station.molecular_dataset,
                                                     day_date=row['start_time_period'], data_source='molecular',
                                                     wavelength=wavelength, file_type='attbsc',
                                                     time_slice=slice(row['start_time_period'],
                                                                      row['end_time_period'])),
                axis=1, result_type='expand')

            if calc_mean_lc:
                # get the mean LC from signal_paths, one day at a time
                # TODO: adapt this part and ds_utils.get_mean_lc() to retrieve mean LC for 3 wavelength at once
                #  (should be faster than reading the file 3 times per wavelength)
                days_groups = df.groupby('date').groups
                days_list = days_groups.keys()
                inds_subsets = [days_groups[key] for key in days_list]
                df_subsets = [df.iloc[inds] for inds in inds_subsets]
                num_processes = min((cpu_count() - 1, len(days_list)))
                with Pool(num_processes) as p:
                    results = p.starmap(ds_utils.get_mean_lc, zip(df_subsets, repeat(station), days_list))
                df = pd.concat([subset for subset in results]).sort_values('start_time_period')

            gen_df = gen_df.append(df)
        except FileNotFoundError as e:
            logger.error(e)
    return gen_df


def calc_data_statistics(station, start_date, end_date, top_height=15.3, mode='gen'):
    """
    Calculated statistics for the period of the dataset of the given station during start_date, end_date
    :param station:  gs.station() object of the lidar station
    :param start_date: datetime.date object of the initial period date
    :param end_date:  datetime.date object of the end period date
    :param top_height: np.float(). The Height[km] **above** ground (Lidar) level - up to which slice the samples.
    Note: default is 15.3 [km]. IF ONE CHANGES IT - THAN THIS WILL AFFECT THE INPUT DIMENSIONS AND STATISTICS !!!
    :return: df_stats: pd.Dataframe, containing statistics of mean and std values for generation signals during
     the desired period [start_date, end_date]. Note: one should previously save the generated dataset for this period.
    """
    # TODO
    """    
        Changes for non generated data:
            change dataset path (without gen)
            profiles_sources = [('range_corr', 'lidar'),  # Pois measurement signal - p_n
                                ('attbsc', 'molecular'),  # molecular signal - attbsc
                                ('p_bg', 'bg')] 
            iterate over df rows and for each run calc_day(sample)_statistics
            change calc_day(sample)_statistics()
            Mean std min max LC from csv, per wavelength, instead of second for loop in calc_day_statistics
            
            change save path
    """
    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    csv_gen_fname = f"dataset_{'gen_' if mode == 'gen' else ''}{station.name}" \
                    f"_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    csv_gen_path = os.path.join(data_folder, csv_gen_fname)
    df = pd.read_csv(csv_gen_path, parse_dates=['date', 'start_time_period', 'end_time_period'])
    days_groups = df.groupby('date').groups
    days_list = [timestamp2datetime(key) for key in days_groups.keys()]
    wavelengths = gs.LAMBDA_nm().get_elastic()

    profiles_sources = [('range_corr', 'lidar'),  # Pois measurement signal - p_n
                        ('attbsc', 'molecular'),  # molecular signal - attbsc
                        ('p_bg', 'bg')]  # background signal - p_bg

    if mode == 'gen':
        profiles_sources = [('p', 'signal'),  # clean signal - p
                            ('range_corr', 'signal'),  # clean signal - range_corr (pr2)
                            ('range_corr_p', 'signal')] + profiles_sources  # signal-range_corr (pr2) with poiss (no bg)

    columns = []
    for profile, source in profiles_sources:
        columns.extend([f"{profile}_{source}_mean", f"{profile}_{source}_std",
                        f"{profile}_{source}_min", f"{profile}_{source}_max"])

    columns.extend(['LC_mean', 'LC_std', 'LC_min', 'LC_max'])  # Lidar calibration value - LC

    df_stats = pd.DataFrame(0, index=pd.Index(wavelengths, name='wavelength'), columns=columns)
    num_processes = min((cpu_count() - 1, len(days_list)))
    with Pool(num_processes) as p:
        if mode == 'gen':
            results = p.starmap(calc_day_statistics, zip(repeat(station), days_list, repeat(top_height)))
        else:
            results = p.starmap(calc_row_statistics, zip(df.iterrows(), repeat(top_height)))
    for result in results:
        df_stats += result

    norm_scale = 1 / len(days_list)
    df_stats *= norm_scale  # normalize by number of days

    # Save stats
    stats_fname = f"stats_{'gen_' if mode == 'gen' else ''}{station.name}_" \
                  f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
    csv_stats_path = os.path.join(data_folder, stats_fname)
    df_stats.to_csv(csv_stats_path)
    return df_stats, csv_stats_path


def calc_day_statistics(station, day_date, top_height=15.3):
    """
    Calculates mean & std for params in datasets_with_names_time_height and datasets_with_names_time

    :param station: gs.station() object of the lidar station
    :param day_date: day_date: datetime.date object of the required date
    :param top_height: np.float(). The Height[km] **above** ground (Lidar) level - up to which slice the samples.
    Note: default is 15.3 [km]. IF ONE CHANGES IT - THAN THIS WILL AFFECT THE INPUT DIMENSIONS AND STATISTICS !!!
    :return: dataframe, each row corresponds to a wavelength
    """

    logger = logging.getLogger()
    wavelengths = gs.LAMBDA_nm().get_elastic()
    molsource = 'molecular'
    sigsource = 'signal'
    lidsource = 'lidar'

    df_stats = pd.DataFrame(index=pd.Index(wavelengths, name='wavelength'))
    logger.debug(f'\nCalculating stats for {day_date}')
    # folder names
    mol_folder = prep_utils.get_month_folder_name(station.molecular_dataset, day_date)
    signal_folder = prep_utils.get_month_folder_name(station.gen_signal_dataset, day_date)
    lidar_folder = prep_utils.get_month_folder_name(station.gen_lidar_dataset, day_date)

    # File names
    signal_nc_name = os.path.join(signal_folder,
                                  gen_utils.get_gen_dataset_file_name(station, day_date, data_source=sigsource))
    lidar_nc_name = os.path.join(lidar_folder,
                                 gen_utils.get_gen_dataset_file_name(station, day_date, data_source=lidsource))
    mol_nc_name = os.path.join(mol_folder, xr_utils.get_prep_dataset_file_name(station, day_date, data_source=molsource,
                                                                               lambda_nm='all'))

    # Load datasets
    mol_ds = xr_utils.load_dataset(mol_nc_name)
    signal_ds = xr_utils.load_dataset(signal_nc_name)
    lidar_ds = xr_utils.load_dataset(lidar_nc_name)
    p_bg = get_daily_bg(station, day_date)  # daily background: p_bg

    # update daily profiles stats
    datasets_with_names_time_height = [(signal_ds.p, f'p_{sigsource}'),
                                       (signal_ds.range_corr, f'range_corr_{sigsource}'),
                                       (signal_ds.range_corr_p, f'range_corr_p_{sigsource}'),
                                       (lidar_ds.p, f'p_{lidsource}'),
                                       (lidar_ds.range_corr, f'range_corr_{lidsource}'),
                                       (mol_ds.attbsc, f'attbsc_{molsource}')]

    for ds, ds_name in datasets_with_names_time_height:
        height_slice = slice(ds.Height.min().values.tolist(), ds.Height.min().values.tolist() + top_height)
        df_stats[f'{ds_name}_mean'] = ds.sel(Height=height_slice).mean(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_std'] = ds.sel(Height=height_slice).std(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_min'] = ds.sel(Height=height_slice).min(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_max'] = ds.sel(Height=height_slice).max(dim={'Height', 'Time'}).values

    datasets_with_names_time = [(p_bg, f'p_bg_bg'), (signal_ds.LC, f'LC')]
    for ds, ds_name in datasets_with_names_time:
        df_stats[f'{ds_name}_mean'] = ds.mean(dim={'Time'}).values
        df_stats[f'{ds_name}_std'] = ds.std(dim={'Time'}).values
        df_stats[f'{ds_name}_min'] = ds.min(dim={'Time'}).values
        df_stats[f'{ds_name}_max'] = ds.max(dim={'Time'}).values

    return df_stats


def calc_row_statistics(row, top_height=15.3):
    """
    Calculates mean & std for params in datasets_with_names_time_height and datasets_with_names_time

    :param station: gs.station() object of the lidar station
    :param day_date: day_date: datetime.date object of the required date
    :param top_height: np.float(). The Height[km] **above** ground (Lidar) level - up to which slice the samples.
    Note: default is 15.3 [km]. IF ONE CHANGES IT - THAN THIS WILL AFFECT THE INPUT DIMENSIONS AND STATISTICS !!!
    :return: dataframe, each row corresponds to a wavelength
    """
    # TODO
    """
        from row (in csv format) lidar_nc_name (lidar_path), mol_nc_name (molecular_path), bg_ds (bg_path), 
        datasets_with_names_time_height without signal, move (p_bg, f'p_bg_bg') to first call
    """
    row_index, row_data = row
    # Load datasets
    mol_ds = xr_utils.load_dataset(row_data['molecular_path'])
    lidar_ds = xr_utils.load_dataset(row_data['lidar_path'])
    p_bg = xr_utils.load_dataset(row_data['bg_path'])

    datasets_with_names_time_height = [(lidar_ds.p, f'p_lidar'),
                                       (lidar_ds.range_corr, f'range_corr_lidar'),
                                       (mol_ds.attbsc, f'attbsc_molecular'),
                                       (p_bg, f'p_bg_bg')]

    # update profiles stats
    wavelengths = gs.LAMBDA_nm().get_elastic()
    df_stats = pd.DataFrame(index=pd.Index(wavelengths, name='wavelength'))
    for ds, ds_name in datasets_with_names_time_height:
        height_slice = slice(ds.Height.min().values.tolist(), ds.Height.min().values.tolist() + top_height)
        df_stats[f'{ds_name}_mean'] = ds.sel(Height=height_slice).mean(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_std'] = ds.sel(Height=height_slice).std(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_min'] = ds.sel(Height=height_slice).min(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_max'] = ds.sel(Height=height_slice).max(dim={'Height', 'Time'}).values

    #  TODO Mean std min max LC from csv, per wavelength, instead of second for loop in calc_day_statistics
    # for ds, ds_name in datasets_with_names_time:
    #     df_stats[f'{ds_name}_mean'] = ds.mean(dim={'Time'}).values
    #     df_stats[f'{ds_name}_std'] = ds.std(dim={'Time'}).values
    #     df_stats[f'{ds_name}_min'] = ds.min(dim={'Time'}).values
    #     df_stats[f'{ds_name}_max'] = ds.max(dim={'Time'}).values

    return df_stats


def save_dataset2timesplits(station, dataset, data_source='lidar', mod_source='gen',
                            profiles=None, sample_size='30min', save_mode='sep',
                            time_slices=None):
    """
    Save the dataset split into time samples per wavelength

    :param station: station: station: gs.station() object of the lidar station
    :param dataset:  xarray.Dataset() a daily generated lidar signal
    :param mod_source:
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'aerosol' - aerosols dataset.
    :param profiles: A list containing the names of profiles desired to be saved separately.
    :param sample_size: string. The sample size. such as '30min'
    :param save_mode: save mode options:
                'sep' - for separated profiles (each is file is per profile per wavelength)
                'single' - save the dataset a single file per day
                'both' - saving both options
    :param time_slices:
    :return:
    """
    day_date = xr_utils.get_daily_ds_date(dataset)
    if time_slices is None:
        time_slices = get_time_splits(station, start_date=day_date, end_date=day_date, sample_size=sample_size)
    if mod_source == 'gen':
        ncpaths = gen_utils.save_generated_dataset(station,
                                                   dataset=dataset,
                                                   data_source=data_source, save_mode=save_mode,
                                                   profiles=profiles, time_slices=time_slices)
    else:
        ncpaths = xr_utils.save_prep_dataset(station,
                                             dataset=dataset,
                                             data_source=data_source, save_mode=save_mode,
                                             profiles=profiles, time_slices=time_slices)
    return ncpaths


def prepare_samples(station, start_date, end_date, top_height=15.3, generated=False):
    # TODO: Adapt this func to get a list of time slices, and group on days to seperate the signal (in case the
    #  times slots are not consecutive)
    #  TODO: prepare_raw_samples()

    logger = logging.getLogger()
    dates = pd.date_range(start_date, end_date, freq='D')
    sample_size = '30min'
    if generated:
        source_profile_path_mode = [('signal_p', 'range_corr_p', station.gen_signal_dataset, 'gen'),
                                    ('signal', 'range_corr', station.gen_signal_dataset, 'gen'),
                                    ('lidar', 'range_corr', station.gen_lidar_dataset, 'gen'),
                                    ('bg', 'p_bg', station.gen_lidar_dataset, 'gen'),
                                    ('molecular', 'attbsc', station.molecular_dataset, 'prep')]
    else:
        source_profile_path_mode = [('lidar', 'range_corr', station.lidar_dataset, 'prep'),
                                    ('bg', 'p_bg', station.lidar_dataset, 'prep'),
                                    ('molecular', 'attbsc', station.molecular_dataset, 'prep')]

    for day_date in dates:
        logger.info(f"Load and split datasets for {day_date.strftime('%Y-%m-%d')}")
        for data_source, profile, base_folder, mode in source_profile_path_mode:

            # Special care for loading/saving different profiles from a specific dataset.
            # E.g. The 'lidar' dataset contains 'range_corr' and 'p_bg' as well
            # The 'signal' dataset contains 'range_corr' and 'range_corr_p' as well
            # load_source - is from where to upload the datasets. data_source - is in what folder to save
            # TODO: Fix this 'source_folder' & 'profile' such that it want require hard coded solutions in the modules.
            data_source = 'signal' if data_source == 'signal_p' else data_source
            load_source = 'lidar' if data_source == 'bg' else data_source

            if mode == 'prep':
                nc_name = xr_utils.get_prep_dataset_file_name(station, day_date, data_source=load_source,
                                                              lambda_nm='all')
            else:
                nc_name = gen_utils.get_gen_dataset_file_name(station, day_date, data_source=load_source)
            month_folder = prep_utils.get_month_folder_name(base_folder, day_date)
            nc_path = os.path.join(month_folder, nc_name)
            dataset = xr_utils.load_dataset(ncpath=nc_path)
            height_slice = slice(dataset.Height.min().values.tolist(),
                                 dataset.Height.min().values.tolist() + top_height)
            save_dataset2timesplits(station, dataset.sel(Height=height_slice),
                                    data_source=data_source, profiles=[profile],
                                    sample_size=sample_size, mod_source=mode)


if __name__ == '__main__':
    parser = get_base_arguments()

    # TODO change modes to options instead of true\false?
    #  or generation true\false then only one do and split?
    parser.add_argument('--DO_DATASET', action='store_true',
                        help='Whether to create a dataset')
    parser.add_argument('--EXTEND_DATASET', action='store_true',
                        help='Whether to EXTEND_DATASET')
    parser.add_argument('--do_calibration_dataset', action='store_true',
                        help='Whether to do_calibration_dataset')
    parser.add_argument('--USE_KM_UNITS', action='store_true',
                        help='Whether to USE_KM_UNITS')
    parser.add_argument('--do_generated_dataset', action='store_true',
                        help='Whether to create do_generated_dataset')
    parser.add_argument('--create_train_test_splits', action='store_true',
                        help='Whether to create train test splits')
    parser.add_argument('--SPLIT_GENERATED_DATASET', action='store_true',
                        help='Whether to create SPLIT_GENERATED_DATASET')
    parser.add_argument('--CALC_GENERATED_STATS', action='store_true',
                        help='Whether to CALC_GENERATED_STATS')
    parser.add_argument('--CALC_STATS', action='store_true',
                        help='Whether to CALC_STATS')
    parser.add_argument('--create_generated_time_split_samples', action='store_true',
                        help='Whether to create_generated_time_split_samples')
    parser.add_argument('--create_time_split_samples', action='store_true',
                        help='Whether to create time split samples')

    args = parser.parse_args()

    dataseting_main(args, log_level=logging.DEBUG)
