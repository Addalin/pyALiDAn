import datetime
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np
import xarray as xr
from tqdm import tqdm

from learning_lidar.utils import utils, global_settings as gs


def save_dataset(dataset: xr.Dataset, folder_name: str = '', nc_name: str = '', nc_path: Optional[str] = None,
                 optim_size: bool = True) -> Optional[str]:
    """
    Save the input dataset to netcdf file

    :param nc_path: full path to netcdf file if already known
    :param dataset: array.Dataset()
    :param folder_name: folder name
    :param nc_name: netcdf file name
    :param optim_size: Boolean. False: the saved dataset will be type 'float64',
                                True: the saved dataset will be type 'float64'(default).
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


def load_dataset(ncpath: str) -> xr.Dataset:
    """
    Load Dataset stored in the netcdf file path (ncpath)
    :param ncpath: a netcdf file path
    :return: xarray.Dataset, if fails return none
    """
    logger = logging.getLogger()

    try:
        if sys.platform in ["linux", "ubuntu"]:
            ncpath = ncpath.replace('\\', '/').replace("//", "/")
        elif sys.platform.__contains__("win"):
            ncpath = ncpath.replace('/', '\\')
        dataset = xr.load_dataset(ncpath, engine='netcdf4').expand_dims()
        dataset.close()
        logger.debug(f"\nLoading dataset file: {ncpath}")
    except Exception as e:
        logger.exception(f"\nFailed to load dataset file: {ncpath}")
        raise e
    return dataset


def get_prep_dataset_file_name(station: gs.Station,
                               day_date: Union[datetime, datetime.date], data_source: str = 'molecular',
                               lambda_nm: str = '*', file_type: str = '*', time_slice=None) -> str:
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
    :param time_slice: TODO

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
    file_name = file_name.replace('all', '').replace('__', '_').replace('__', '_').replace('**', '*').replace('_*_*',
                                                                                                              '_*')
    return file_name


def get_prep_dataset_paths(station: gs.Station, day_date: datetime.datetime, data_source: str = 'molecular',
                           lambda_nm: str = '*', file_type: str = '*') -> list:
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
        parent_folder = station.lidar_dataset  # TODO: make sure if this should not be station.lidar_dataset_calib ?
    else:
        raise Exception("Unsupported data_source.")

    month_folder = utils.get_month_folder_name(parent_folder, day_date)
    file_name = get_prep_dataset_file_name(station, day_date, data_source, lambda_nm, file_type)

    file_pattern = os.path.join(month_folder, file_name)

    paths = sorted(glob.glob(file_pattern))

    return paths


def get_daily_prep_ds(station: gs.Station, day_date: datetime.date, data_source: str) -> xr.Dataset:
    """
    Returns the daily parameters of measures (lidar), signal, density or aerosol  creation as a dataset.

    :param type_: str, should be one of 'lidar' / 'molecular'
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Daily dataset of generation parameters.
    """
    daily_ds_path = get_prep_dataset_paths(station, day_date, data_source)[0]
    ds = load_dataset(daily_ds_path)
    return ds


def save_prep_dataset(station: gs.Station, dataset: xr.Dataset, data_source: str = 'lidar',
                      level: str = 'level1a', save_mode: str = 'both', profiles=None, time_slices=None):
    """
    Save the input dataset to netcdf file
    :param time_slices:
    :param profiles: The name of profile desired to be saved separately.
    If this name is not provided, then the first profile is automatically selected
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily preprocessed lidar or molecular signals.
    Having dimensions of : Wavelength, Height, Time.
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'molecular' - molecular dataset.
    :param level: incase data_source is 'lidar' then split to raw (level0) and to pollynet post process (level1a)
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' - saving both options
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    # TODO: merge save_prep_dataset() &  save_generated_dataset() -> save_daily_dataset() with a flag of 'gen' or 'prep'
    date_datetime = get_daily_ds_date(dataset)
    month_folder = get_prep_month_folder(station, date_datetime, data_source, level)
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


def get_prep_month_folder(station, date_datetime, data_source, level):
    if data_source == 'lidar':
        if level == 'level0':
            base_folder = station.lidar_dataset
        elif level == 'level1a':
            base_folder = station.lidar_dataset_calib
    elif data_source == 'bg':
        base_folder = station.bg_dataset
    elif data_source == 'molecular':
        base_folder = station.molecular_dataset
    else:
        raise Exception("Unsupported data_source.")
    month_folder = utils.get_month_folder_name(base_folder, date_datetime)
    return month_folder


def load_and_save_all_datasets_in_paths(base_path, paths, exclude_paths):
    """
    Was convert_To32
    \data_haifa\DATA FROM TROPOS\molecular_dataset",
                 r"\data_haifa\DATA FROM TROPOS\lidar_dataset"]

        exclude_paths = [r"D:\data_haifa\GENERATION\density_dataset\2017\04",
                         r"D:\data_haifa\GENERATION\density_dataset\2017\05"]

    :return: None
    """

    exclude_files = []
    for exclude_path in exclude_paths:
        exclude_files.extend(list(Path(exclude_path).glob("**/*.nc")))
    exclude_files = [str(x) for x in exclude_files]
    exclude_files = set(exclude_files)

    for path in paths:
        full_paths = Path(base_path + path)
        file_list = set([str(pp) for pp in full_paths.glob("**/*.nc")])
        file_list = file_list - exclude_files
        print(f"found {len(file_list)} nc files in path {base_path + path}")
        for nc_path in tqdm(file_list):
            ds = load_dataset(str(nc_path))
            save_dataset(ds, nc_path=str(nc_path))


def get_daily_ds_date(dataset):
    logger = logging.getLogger()
    try:
        date_64 = dataset.date.values
    except ValueError:
        logger.exception("\nThe dataset does not contain a data variable named 'date'")
        return None
    date_datetime = utils.dt64_2_datetime(date_64)
    return date_datetime
