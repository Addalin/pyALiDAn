import logging
import os
import sqlite3
import sys
from datetime import datetime
from functools import partial

import multiprocess as mp
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split

import learning_lidar.generation.generation_utils as gen_utils
import learning_lidar.preprocessing.preprocessing_utils as prep_utils
from learning_lidar.utils import xr_utils, global_settings as gs


def get_query(wavelength, cali_method, day_date):
    start_time = datetime.combine(date=day_date.date(), time=day_date.time().min)
    end_time = datetime.combine(date=day_date.date(), time=day_date.time().max)
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


def query_database(query="SELECT * FROM lidar_calibration_constant;",
                   database_path="pollyxt_tropos_calibration.db"):
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
    logger = logging.getLogger()

    # Connect to the db and query it directly into pandas df.
    try:
        with sqlite3.connect(database_path) as c:
            # Query to df
            # optionally parse 'id' as index column and 'cali_start_time', 'cali_stop_time' as dates
            df = pd.read_sql(sql=query, con=c, parse_dates=['cali_start_time', 'cali_stop_time'])
    except sqlite3.OperationalError as e:
        logger.exception(f"{e}: {database_path}. Stopping dataseting.")
        sys.exit(1)
    # TODO:
    #  raise load exception if file does not exits. the above commented solution is not really catching this and
    #  continues to run

    return df


def add_profiles_values(df, station, day_date, file_type='profiles'):
    logger = logging.getLogger()
    try:
        df['matched_nc_profile'] = df.apply(lambda row: prep_utils.get_TROPOS_dataset_paths(station, day_date,
                                                                                            start_time=row.cali_start_time,
                                                                                            end_time=row.cali_stop_time,
                                                                                            file_type=file_type,
                                                                                            level='level1a')[0],
                                            axis=1, result_type='expand')
    except Exception:
        logger.exception(
            f"Non resolved 'matched_nc_profile' for {station.location} station, at date {day_date.strftime('%Y-%m-%d')} ")
        pass

    def _get_info_from_profile_nc(row):
        """
        Get the r_0,r_1, and delta_r of the selected row. The values are following rebasing according to sea-level height.
        :param row:
        :return:
        """
        data = xr_utils.load_dataset(row['matched_nc_profile'])
        wavelen = row.wavelength
        # get altitude to rebase the reference heights according to sea-level-height
        altitude = data.altitude.item()
        [r0, r1] = data[f'reference_height_{wavelen}'].values
        [bin_r0, bin_r1] = [np.argmin(abs(data.height.values - r)) for r in [r0, r1]]
        delta_r = r1 - r0
        return r0 + altitude, r1 + altitude, delta_r, bin_r0, bin_r1

    df[['r0', 'r1', 'dr', 'bin_r0', 'bin_r1']] = df.apply(_get_info_from_profile_nc, axis=1,
                                                          result_type='expand')
    return df


def add_X_path(df, station, day_date, lambda_nm=532, data_source='molecular', file_type='attbsc'):
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
    logger = logging.getLogger()
    paths = xr_utils.get_prep_dataset_paths(station=station,
                                            day_date=day_date,
                                            data_source=data_source,
                                            lambda_nm=lambda_nm,
                                            file_type=file_type)
    if not paths:
        df.loc[:, f"{data_source}_path"] = ""
        logger.debug(
            f"\n Not existing any '{data_source}' path for {station.location} station, at {day_date.strftime('%Y-%m-%d')}")

    elif len(paths) != 1:
        raise Exception(
            f"\n Expected ONE '{data_source}' path for {station.location} station, at {day_date.strftime('%Y-%m-%d')}.\nGot: {paths} ")
    else:
        df.loc[:, f"{data_source}_path"] = paths[0]
    return df


def get_time_slots_expanded(df, sample_size):
    if sample_size:
        expanded_df = pd.DataFrame()
        for indx, row in df.iterrows():
            time_slots = pd.date_range(row.loc['cali_start_time'], row.loc['cali_stop_time'], freq=sample_size)
            time_slots.freq = None
            if len(time_slots) < 2:
                continue
            for start_time, end_time in zip(time_slots[:-1], time_slots[1:]):
                mini_df = row.copy()
                mini_df['start_time_period'] = start_time
                mini_df['end_time_period'] = end_time
                expanded_df = expanded_df.append(mini_df)
    else:
        expanded_df = df.copy()
        expanded_df['start_time_period'] = expanded_df['cali_start_time']
        expanded_df['end_time_period'] = expanded_df['cali_stop_time']
    return expanded_df


def convert_Y_features_units(df, Y_features=['LC', 'LC_std', 'r0', 'r1', 'dr'],
                             scales={'LC': 1E-9, 'LC_std': 1E-9, 'r0': 1E-3, 'r1': 1E-3, 'dr': 1E-3}):
    Y_scales = [scales[feature] for feature in Y_features]
    for feature, scale in zip(Y_features, Y_scales):
        df[feature] *= scale
    return df


def recalc_LC(row):
    """
	Extending database:by recalculate LC_recalc, LCp_recalc
	:param row:
	:return:
	"""
    sliced_ds, _ = get_sample_ds(row)
    slice_ratio = sliced_ds[0].copy(deep=True)
    slice_ratio.data /= sliced_ds[1].data
    LC_recalc = slice_ratio.mean().values.item()
    LCp_recalc = slice_ratio.where(slice_ratio >= 0.0).mean().values.item()
    return LC_recalc, LCp_recalc


def _df_split2proc(tup_arg, **kwargs):
    split_ind, df_split, df_f_name = tup_arg
    return (split_ind, getattr(df_split, df_f_name)(**kwargs))


def df_multi_core(df, df_f_name, subset=None, njobs=-1, **kwargs):
    # %% testing multiproccesing from: https://gist.github.com/morkrispil/3944242494e08de4643fd42a76cb37ee
    if njobs == -1:
        njobs = mp.cpu_count()
    pool = mp.Pool(processes=njobs)

    try:
        df_sub = df[subset] if subset else df
        splits = np.array_split(df_sub, njobs)
    except ValueError:
        splits = np.array_split(df, njobs)

    pool_data = [(split_ind, df_split, df_f_name) for split_ind, df_split in enumerate(splits)]
    results = pool.map(partial(_df_split2proc, **kwargs), pool_data)
    pool.close()
    pool.join()
    results = sorted(results, key=lambda x: x[0])
    results = pd.concat([split[1] for split in results])
    return results


def get_sample_ds(row):
    mol_path = row.molecular_path
    lidar_path = row.lidar_path
    bin_r0 = row.bin_r0
    bin_r1 = row.bin_r1
    t0 = row.start_time_period
    t1 = row.end_time_period
    full_ds = [xr_utils.load_dataset(path) for path in [lidar_path, mol_path]]
    tslice = slice(t0, t1)
    profiles = ['range_corr', 'attbsc']
    sliced_ds = [ds_i.sel(Time=tslice,
                          Height=slice(float(ds_i.Height[bin_r0].values),
                                       float(ds_i.Height[bin_r1].values)))[profile]
                 for ds_i, profile in zip(full_ds, profiles)]
    return sliced_ds, full_ds


def get_aerBsc_profile_ds(path, profile_df):
    cur_profile = xr_utils.load_dataset(path)
    height_units = 'km'
    height_scale = 1e-3  # converting [m] to [km]
    bsc_scale = 1e+3  # converting [1/m sr] to to [1/km sr]
    heights_indx = height_scale * (cur_profile.height.values + cur_profile.altitude.values)
    start_time = datetime.utcfromtimestamp(cur_profile.start_time.values[0].tolist())
    end_time = datetime.utcfromtimestamp(cur_profile.end_time.values[0].tolist())
    time_indx = pd.date_range(start=start_time, end=end_time, freq='30S')
    profile_r0 = profile_df.groupby(['wavelength'])['bin_r0'].unique().astype(np.int).values
    profile_r1 = profile_df.groupby(['wavelength'])['bin_r0'].unique().astype(np.int).values
    ds_chans = []

    var_names = [f"aerBsc_klett_{wavelength}" for wavelength in [355, 532, 1064]]
    for v_name, wavelength, r in zip(var_names, [355, 532, 1064], profile_r1):
        vals = cur_profile[v_name].values
        vals[r:] = gs.eps
        vals = gaussian_filter1d(vals, 21, mode='nearest')
        vals = vals.T.reshape(len(heights_indx), 1)
        vals[vals < 0] = gs.eps
        vals *= bsc_scale
        mat_vals = np.repeat(vals, len(time_indx), axis=1)
        aerbsc_df = pd.DataFrame(mat_vals, index=heights_indx, columns=time_indx)
        aerBsc_ds_chan = xr.Dataset(
            data_vars={'aerBsc': (('Height', 'Time'), aerbsc_df),
                       'lambda_nm': ('Wavelength', [wavelength])
                       },
            coords={'Height': aerbsc_df.index.to_list(),
                    'Time': aerbsc_df.columns,
                    'Wavelength': [wavelength]
                    })
        aerBsc_ds_chan.aerBsc.attrs = {'long_name': r'$\beta$',  # _{{a}}$' ,
                                       'units': r'$km^{{-1}} sr^{-1}$',
                                       'info': r'$Aerosol backscatter$'}
        # set attributes of coordinates
        aerBsc_ds_chan.Height.attrs = {'units': '{}'.format('{}'.format(height_units)),
                                       'info': 'Measurements heights above sea level'}
        aerBsc_ds_chan.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}
        ds_chans.append(aerBsc_ds_chan)
    return xr.concat(ds_chans, dim='Wavelength')


def split_save_train_test_ds(csv_path='', train_size=0.8, df=None):
    """
    Splitting a dataset into 2 : train and test sets. Then saving with similar abbreviation of the original dataset.
    :param train_size:  np.float(). in [0,1] - the ratio of train size from the dataset
    :param df: pd.DataFrame(). Optional. Dataset of samples for learning calibration method.
    :return: train_set, test_set. Of type pd.DataFrame().
    """

    logger = logging.getLogger()
    if df is None:
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError as e:
            logger.error(f'\n{e}.\nPlease provide a dataset (df) or an exiting csv path (csv_path).')

    source_path = csv_path.split('.csv')[0]
    train_path = f'{source_path}_train.csv'
    test_path = f'{source_path}_test.csv'
    df_copy = df.copy(deep=True)
    train_set, test_set = train_test_split(df_copy, train_size=train_size, random_state=2021, shuffle=True,
                                           stratify=df_copy['wavelength'])
    train_set = train_set.copy(deep=True)
    test_set = test_set.copy(deep=True)
    train_set.sort_index(inplace=True)
    test_set.sort_index(inplace=True)
    train_set['idx'] = train_set.index.values
    test_set['idx'] = test_set.index.values

    test_set.to_csv(test_path, index=False)
    train_set.to_csv(train_path, index=False)
    logger.info(f"\nThe dataset split to train and test datasets, saved to :\n{test_path},{train_path}")
    return train_set, test_set


def get_generated_X_path(station, parent_folder, day_date, data_source, wavelength, file_type=None, time_slice=None):
    month_folder = prep_utils.get_month_folder_name(parent_folder=parent_folder, day_date=day_date)
    nc_name = gen_utils.get_gen_dataset_file_name(station, day_date, data_source=data_source, wavelength=wavelength,
                                                  file_type=file_type, time_slice=time_slice)
    nc_path = os.path.join(month_folder, nc_name)
    return nc_path


def get_prep_X_path(station, parent_folder, day_date, data_source, wavelength, file_type=None, time_slice=None):
    month_folder = prep_utils.get_month_folder_name(parent_folder=parent_folder, day_date=day_date)
    nc_name = xr_utils.get_prep_dataset_file_name(station, day_date, data_source=data_source, lambda_nm=wavelength,
                                                  file_type=file_type, time_slice=time_slice)
    data_path = os.path.join(month_folder, nc_name)
    return data_path


def get_mean_lc(df, station, day_date):
    """

    :param df:
    :param station:
    :param wavelength:
    :return:
    """
    day_indices = df['date'] == day_date  # indices of current day in df

    # path to signal_dataset of current day
    nc_path = get_generated_X_path(station=station, parent_folder=station.gen_signal_dataset, day_date=day_date,
                                   data_source='signal', wavelength='*')

    # Load the LC of current day
    LC_day = xr_utils.load_dataset(nc_path).LC

    # Add mean LC values for each time slice
    df.loc[day_indices, ['LC']] = df.loc[day_indices]. \
        apply(lambda row: LC_day.sel(Time=slice(row['start_time_period'], row['end_time_period']))
              .mean(dim='Time').sel(Wavelength=row['wavelength']).values,
              axis=1, result_type='expand')
    return df


class Error(Exception):
    """Base class for other exceptions"""
    pass


class EmptyDataFrameError(Error):
    """Raised when the data frame is empty and should not have been"""
    pass


def calc_sample_statistics(row, top_height, mode='gen'):
    """
    Calculates mean & std for params in datasets_with_names_time_height and datasets_with_names_time

    :param row: row from raw database table (pandas.Dataframe())
    :param top_height: np.float(). The Height[km] **above** ground (Lidar) level - up to which slice the samples.
    Note: default is 15.3 [km]. IF ONE CHANGES IT - THAN THIS WILL AFFECT THE INPUT DIMENSIONS AND STATISTICS !!!
    :param mode:
    :return: dataframe, each row corresponds to a wavelength
    """
    _, row_data = row
    # Load datasets
    mol_ds = xr_utils.load_dataset(row_data['molecular_path'])
    lidar_ds = xr_utils.load_dataset(row_data['lidar_path'])
    p_bg = xr_utils.load_dataset(row_data['bg_path'])
    # TODO uncomment after correcting dataset creation
    # signal_range_corr_ds = xr_utils.load_dataset(row_data['signal_path'])
    # signal_range_corr_p_ds = xr_utils.load_dataset(row_data['signal_p_path'])
    # signal_p_ds = xr_utils.load_dataset(row_data['signal_p_only_path'])

    datasets_with_names_time_height = [(lidar_ds.range_corr, f'range_corr_lidar'),
                                       (mol_ds.attbsc, f'attbsc_molecular'),
                                       (p_bg, f'p_bg_bg')]

    # TODO uncomment after correcting dataset creation
    # if mode == 'gen':
    #     datasets_with_names_time_height = [(signal_p_ds.p, 'p_signal'),
    #                                        (signal_range_corr_ds.range_corr, 'range_corr_signal'),
    #                                        (signal_range_corr_p_ds.range_corr_p, 'range_corr_p_signal')] + datasets_with_names_time_height

    # update profiles stats
    wavelengths = gs.LAMBDA_nm().get_elastic()
    df_stats = pd.DataFrame(index=pd.Index(wavelengths, name='wavelength'))
    for ds, ds_name in datasets_with_names_time_height:
        height_slice = slice(ds.Height.min().values.tolist(), ds.Height.min().values.tolist() + top_height)
        df_stats[f'{ds_name}_mean'] = ds.sel(Height=height_slice).mean(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_std'] = ds.sel(Height=height_slice).std(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_min'] = ds.sel(Height=height_slice).min(dim={'Height', 'Time'}).values
        df_stats[f'{ds_name}_max'] = ds.sel(Height=height_slice).max(dim={'Height', 'Time'}).values

    return df_stats