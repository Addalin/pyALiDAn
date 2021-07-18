import glob
import logging
import os

import numpy as np
import xarray as xr
from tqdm import tqdm

from learning_lidar.preprocessing import preprocessing_utils as prep_utils


def save_dataset(dataset, folder_name='', nc_name='', nc_path=None, optim_size=True):
    """
    Save the input dataset to netcdf file

    :param nc_path: full path to netcdf file if already known
    :param dataset: array.Dataset()
    :param folder_name: folder name
    :param nc_name: netcdf file name
    :param optim_size: Boolean. False: the saved dataset will be type 'float64', True: the saved dataset will be type 'float64'(default).
    :return: nc_path - full path to netcdf file created if succeeded, else none

    """
    logger = logging.getLogger()
    if nc_path:
        folder_name, nc_name = os.path.dirname(nc_path), os.path.basename(nc_path)
    if optim_size:
        try:
            # Separate whether its a dataset (has data_vars) or dataarray
            if hasattr(dataset, 'data_vars'):
                for var in dataset.data_vars:
                    if dataset[var].dtype == np.float64 and len(dataset[var].dims) >= 2:
                        dataset[var] = dataset[var].astype(np.float32,
                                                           casting='same_kind',
                                                           copy=False,
                                                           keep_attrs=True)
            else:
                if dataset.dtype == np.float64 and len(dataset.dims) >= 2:
                    dataset = dataset.astype(np.float32,
                                             casting='same_kind',
                                             copy=False,
                                             keep_attrs=True)

            logger.debug(f"\nCasting float64 to float32 in file - {folder_name} {nc_name}")
        except Exception:
            logger.exception(f"\nFailed casting float64 to float32")
            return None
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            logger.debug(f"\nCreating folder: {folder_name}")
        except Exception:
            logger.exception(f"\nFailed to create folder: {folder_name}")
            return None

    nc_path = os.path.join(folder_name, nc_name)
    try:
        dataset.to_netcdf(nc_path, mode='w', format='NETCDF4', engine='netcdf4')
        dataset.close()
        logger.debug(f"\nSaving dataset file: {nc_path}")
    except Exception:
        logger.exception(f"\nFailed to save dataset file: {nc_path}")
        nc_path = None
    return nc_path


def load_dataset(ncpath):
    """
    Load Dataset stored in the netcdf file path (ncpath)
    :param ncpath: a netcdf file path
    :return: xarray.Dataset, if fails return none
    """
    logger = logging.getLogger()
    try:
        dataset = xr.load_dataset(ncpath, engine='netcdf4').expand_dims()
        dataset.close()
        logger.debug(f"\nLoading dataset file: {ncpath}")
    except Exception as e:
        logger.exception(f"\nFailed to load dataset file: {ncpath}")
        raise e
    return dataset


def get_prep_dataset_file_name(station, day_date, data_source='molecular',
                               lambda_nm='*', file_type='*', time_slice=None):
    """
     Retrieves file pattern name of preprocessed dataset according to
     date, station, wavelength dataset source, and profile type.

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm] or
    all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or
    'all' (meaning the dataset contains several profile types)

    :return: dataset file name (netcdf) file of the data_type required per
    given day and wavelength, data_source and file_type
    """
    dt_str = day_date.strftime('%Y_%m_%d')
    if time_slice:
        dt_str += f"_{time_slice.start.strftime('%H%M%S')}_{time_slice.stop.strftime('%H%M%S')}"
    if lambda_nm == 'all':
        file_type = ''
    if file_type == '*' or lambda_nm == '*':  # * for lambd_nm - means any wavelength and profile
        # retrieves any file of this date
        file_name = f"{dt_str}_{station.location}_{file_type}_{lambda_nm}*{data_source}.nc"
    else:
        # this option is mainly to set new file names
        file_name = f"{dt_str}_{station.location}_{file_type}_{lambda_nm}_{data_source}.nc"
    file_name = file_name.replace('all', '').replace('__', '_').replace('__', '_')
    return file_name


def get_prep_dataset_paths(station, day_date, data_source='molecular', lambda_nm='*', file_type='*'):
    """
     Retrieves file paths of preprocessed datasets according to
     date, station, wavelength dataset source, and profile type.

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm], or
    'all' (meaning the dataset contains several profile types)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or
    'all' (meaning the dataset contains several profile types)

    :return: paths to all datasets netcdf files of the data_type required per given day and wavelength
    """
    if data_source == 'molecular':
        parent_folder = station.molecular_dataset
    elif data_source == 'lidar':
        parent_folder = station.lidar_dataset

    month_folder = prep_utils.get_month_folder_name(parent_folder, day_date)
    file_name = get_prep_dataset_file_name(station, day_date, data_source, lambda_nm, file_type)

    # print(os.listdir(month_folder))
    file_pattern = os.path.join(month_folder, file_name)

    paths = sorted(glob.glob(file_pattern))

    return paths


def save_prep_dataset(station, dataset, data_source='lidar', save_mode='both',
                      profiles=None, time_slices=None):
    """
    Save the input dataset to netcdf file
    :param time_slices:
    :param profiles: The name of profile desired to be saved separately.
    If this name is not provided, then the first profile is automatically selected
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily preprocessed lidar or molecular signals.
    Having dimensions of : Wavelength, Height, Time.
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'molecular' - molecular dataset.
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' - saving both options
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    # TODO: merge save_prep_dataset() &  save_generated_dataset() --> save_daily_dataset() with a flag of 'gen' or 'prep'
    date_datetime = prep_utils.get_daily_ds_date(dataset)
    if data_source == 'lidar':
        base_folder = station.lidar_dataset
    elif data_source == 'bg':
        base_folder = station.bg_dataset
    elif data_source == 'molecular':
        base_folder = station.molecular_dataset
    month_folder = prep_utils.get_month_folder_name(base_folder, date_datetime)

    prep_utils.get_daily_ds_date(dataset)
    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []

    if save_mode in ['both', 'sep']:
        if not profiles:
            profiles = [list(dataset.data_vars)[0]]
        for profile in profiles:
            for wavelength in dataset.Wavelength.values:
                ds_profile = dataset.sel(Wavelength=wavelength)[profile]
                ds_profile['date'] = date_datetime
                if time_slices:
                    for time_slice in tqdm(time_slices,
                                           desc=f"Split and save time slices for: {profile}, {wavelength}"):
                        file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                                               lambda_nm=wavelength, file_type=profile,
                                                               time_slice=time_slice)
                        ds_slice = ds_profile.sel(Time=time_slice)
                        ncpath = save_dataset(ds_slice, month_folder, file_name)
                        if ncpath:
                            ncpaths.append(ncpath)
                else:
                    file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                                           lambda_nm=wavelength, file_type=profile)
                    ncpath = save_dataset(ds_profile, month_folder, file_name)
                    if ncpath:
                        ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                               lambda_nm='all', file_type='all')
        ncpath = save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths