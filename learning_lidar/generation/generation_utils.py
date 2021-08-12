import calendar
import os
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import xarray as xr

import learning_lidar.preprocessing.preprocessing_utils as prep_utils
from learning_lidar.utils import xr_utils, global_settings as gs

PLOT_RESULTS = False


def get_gen_dataset_file_name(station: gs.Station, day_date, wavelength='*',
                              data_source='lidar', file_type='range_corr',
                              time_slice=None) -> str:
    """
     Retrieves file pattern name of generated lidar dataset according to:
      station, date, wavelength.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the generated date
    :param wavelength: wavelength [nm] e.g., for the green channel 532 [nm] or all
    (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'aerosol' or 'lidar'
    :param file_type: string object: e.g., 'range_corr'/'lidar_power' for separated files per wavelength
    (355,532, or 1064) or 'lidar'/'aerosol' for all wavelengths
    :param time_slice:

    :return: dataset file name (netcdf) file of the data_type required per given day, wavelength, data_source and file_type
    """
    dt_str = day_date.strftime('%Y_%m_%d')
    if time_slice:
        dt_str += f"_{time_slice.start.strftime('%H%M%S')}_{time_slice.stop.strftime('%H%M%S')}"
    if wavelength == '*':
        file_name = f"{dt_str}_{station.location}_generated_{data_source}.nc"
    else:
        file_name = f"{dt_str}_{station.location}_generated_{file_type}_{wavelength}_{data_source}.nc"

    return file_name


def save_generated_dataset(station: gs.Station, dataset: xr.Dataset, data_source: str = 'lidar',
                           save_mode: str = 'both',
                           profiles=None, time_slices=None):
    """
    Save the input dataset to netcdf file
    If this name is not provided, then the first profile is automatically selected
    :param time_slices: list of time slices slice(start_time, end_time), times are of datetime.datime() object
    :param station: station: gs.station() object of the lidar station
    :param dataset: xarray.Dataset() a daily generated lidar signal, holding 5 data variables:
             4 daily dataset, with dimensions of : Height, Time, Wavelength.
              name of profiles: 'range_corr','range_corr_p', 'lidar_sig','lidar_sig_p'
             1 shared variable: date
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'aerosol' - aerosols dataset.
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' - saving both options
    :param profiles: The name of profile desired to be saved separately.
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    # TODO: merge save_prep_dataset() &  save_generated_dataset() --> save_daily_dataset() with a flag of 'gen' or 'prep'
    date_datetime = xr_utils.get_daily_ds_date(dataset)
    if data_source == 'lidar':
        base_folder = station.gen_lidar_dataset
    elif data_source == 'signal':
        base_folder = station.gen_signal_dataset
    elif data_source == 'aerosol':
        base_folder = station.gen_aerosol_dataset
    elif data_source == 'density':
        base_folder = station.gen_density_dataset
    elif data_source == 'bg':
        base_folder = station.gen_bg_dataset
    else:
        raise Exception("Unsupported data_source.")
    month_folder = prep_utils.get_month_folder_name(base_folder, date_datetime)

    xr_utils.get_daily_ds_date(dataset)
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
                        file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                                              wavelength=wavelength, file_type=profile,
                                                              time_slice=time_slice)
                        ds_slice = ds_profile.sel(Time=time_slice)
                        ncpath = xr_utils.save_dataset(ds_slice, month_folder, file_name)
                        if ncpath:
                            ncpaths.append(ncpath)
                else:
                    file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                                          wavelength=wavelength, file_type=profile)
                    ncpath = xr_utils.save_dataset(ds_profile, month_folder, file_name)
                    if ncpath:
                        ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                              wavelength='*')
        ncpath = xr_utils.save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


def get_month_gen_params_path(station: gs.Station, day_date: datetime.date, type_: str = 'density_params') -> str:
    """
    :param type_: type of generated parameter:
        'density_params' - for density sampler generator,
        'bg'- for generated background signal,
        'LC' - for generated Lidar Constant signal
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: str. Path to monthly dataset of generation parameters.
    """
    year = day_date.year
    month = day_date.month
    month_start_day = datetime(year, month, 1, 0, 0)
    _, monthdays = calendar.monthrange(year, month)
    month_end_day = datetime(year, month, monthdays, 0, 0)

    folder_name = prep_utils.get_month_folder_name(station.generation_folder, day_date)

    nc_name = f"generated_{type_}_{station.location}_{month_start_day.strftime('%Y-%m-%d')}_" \
              f"{month_end_day.strftime('%Y-%m-%d')}.nc"

    gen_source_path = os.path.join(folder_name, nc_name)
    return gen_source_path


def get_daily_ds_path(station: gs.Station, day_date: datetime.date, type_: str) -> str:
    """
    Get the path to the daily generated measure (lidar) or signal ds

    :param type_: str, 'lidar' for measure dataset. 'signal' for signal dataset
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: str. Path to monthly dataset of generation parameters.
    """
    if type_ == 'lidar':
        parent_folder = station.gen_lidar_dataset
    elif type_ == 'signal':
        parent_folder = station.gen_signal_dataset
    else:
        raise Exception("Unsupported type. Should by 'lidar' or 'signal'")

    month_folder = prep_utils.get_month_folder_name(parent_folder, day_date)
    file_name = get_gen_dataset_file_name(station, day_date, wavelength='*', data_source=type_)
    gen_source_path = os.path.join(month_folder, file_name)
    return gen_source_path


def get_month_gen_params_ds(station: gs.Station, day_date: datetime.date, type_: str = 'density_params') -> xr.Dataset:
    """
    Returns the monthly parameters of density creation as a dataset.
    :param type_: type of generated parameter:
        'density_params' - for density sampler generator,
        'bg'- for generated background signal,
        'LC' - for generated Lidar Constant signal
        'overlap' - for generated overlap parameters
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Monthly dataset of generation parameters.
    """

    gen_source_path = get_month_gen_params_path(station, day_date, type_)
    month_params_ds = xr_utils.load_dataset(gen_source_path)
    return month_params_ds


def get_daily_gen_param_ds(station: gs.Station, day_date: datetime.date, type_: str = 'density_params') -> xr.Dataset:
    """
    Returns the daily parameters of density creation as a dataset.
    :param type_: type of generated parameter:
        'density_params' - for density sampler generator,
        'bg'- for generated background signal,
        'LC' - for generated Lidar Constant signal
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Daily dataset of generation parameters.
    """
    month_params_ds = get_month_gen_params_ds(station, day_date, type_)
    day_params_ds = month_params_ds.sel(Time=slice(day_date, day_date + timedelta(days=1)))

    return day_params_ds


def get_daily_gen_ds(station: gs.Station, day_date: datetime.date, type_: str) -> xr.Dataset:
    """
    Returns the daily parameters of measures (lidar) or signal creation as a dataset.

    :param type_: str, should be one of 'signal' / 'lidar'
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Daily dataset of generation parameters.
    """
    daily_ds_path = get_daily_ds_path(station, day_date, type_)
    ds = xr_utils.load_dataset(daily_ds_path)
    return ds


def get_daily_overlap(station: gs.Station, day_date: datetime.date, height_index: xr.DataArray) -> xr.Dataset:
    """
    Generates overlap values per height index, from overlap params

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param height_index: xr.DataArray, stores the height index per Height

    :return: xr.Dataset with the overlap values per height index
    """

    overlap_params = get_month_gen_params_ds(station, day_date, type_='overlap')
    overlap = sigmoid(height_index, *overlap_params.to_array().values)
    overlap_ds = xr.Dataset(data_vars={'overlap': ('Height', overlap)},
                            coords={'Height': height_index.values},
                            attrs={'name': 'Overlap Function'})

    return overlap_ds


def dt2binscale(dt_time: datetime.date, res_sec: int = 30) -> np.ndarray:
    """
    TODO consider to move this function to Station ?
    Returns the bin index corresponds to dt_time
    binscale - is the time scale [0,2880], of a daily lidar bin index from 00:00:00 to 23:59:30.
    The lidar has a bin measure every 30 sec, in total 2880 bins per day.
    :param res_sec: TODO
    :param dt_time: datetime.datetime object
    :return: binscale - float in [0,2880]
    """
    res_minute = 60 / res_sec
    res_hour = 60 * res_minute
    res_musec = 1e-6 / res_sec
    tind = dt_time.hour * res_hour + \
           dt_time.minute * res_minute + \
           dt_time.second / res_sec + \
           dt_time.microsecond * res_musec
    return tind


def create_ratio(total_bins, mode='ones', start_height=0.3, ref_height=2.5, ref_height_bin=300):
    """
    Generating ratio, mainly for applying different endings on the daily profile, or overlap function.
    Currently the ratio is "1" for all bins.
    Change this function to produce more options of generated aerosol densities.
    :param total_bins: Total bins in one column of measurements (through height)
    :param start_height: Start measuring height ( the height of 1st bin) in km
    :param mode: The mode of the generated ratio:
        1. "ones" - all bins are equal
        2. "end" - apply different ratio on the endings of the aerosols density
        3. "overlap" - generate some overlap function on the lidar measurement, the varying ratio is on lower heights.
    :param ref_height: Reference height in km
    :param ref_height_bin: The bin number of reference height
    :return: Ratio 0-1 throughout height , that affect on the aerosols density or the lidar measurement.
    """
    t_interp = np.arange(start=1, stop=total_bins + 1, step=1)
    if mode == "end":
        t_start = start_height / ref_height
        t_ending = np.array(
            [0, 0.125 * t_start, 0.25 * t_start, .5 * t_start, t_start, 0.05, 0.1, .3, .4, .5, .6, .7, .8, .9,
             1.0]) * np.float(ref_height_bin)
        ratios = np.array([1, 1, 1, 1, 1, 1.0, 1, 1, 1, 1, 0.95, 0.85, 0.4, .3, 0.2])
        ratio_interp = np.interp(t_interp, t_ending, ratios)
        smooth_ratio = gaussian_filter1d(ratio_interp, sigma=40)
    elif mode == "overlap":
        # the overlap function is relevant to heights up 600-700 meter. setting to 95 means 90*7.5 =675 [m]
        t_start = 95 / total_bins
        r_start = 1
        t_overlap = np.array(
            [0.0, 0.2 * t_start, 0.5 * t_start, .75 * t_start, t_start, 0.05, 0.1, .3, .4, .5, .6, .7, .8, .9,
             1.0]) * np.float(ref_height_bin)
        ratios = np.array([.0, 0.2 * r_start, 0.5 * r_start, 0.75 * r_start, r_start, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        ratio_interp = np.interp(t_interp, t_overlap, ratios)
        smooth_ratio = gaussian_filter1d(ratio_interp, sigma=40)
    elif mode == "ones":
        y = np.arange(total_bins)
        smooth_ratio = np.ones_like(y)

    if PLOT_RESULTS:
        plt.figure(figsize=(3, 3))
        plt.plot(smooth_ratio, t_interp, label=mode)
        plt.ylabel('Height bins')
        plt.xlabel('ratio')
        plt.show()

    return smooth_ratio


def sigmoid(x, x0=0, A=0, K=1, B=1, v=0.4, C=1, Q=1):
    y = A + (K - A) / (gs.eps + (C + Q * np.exp(-B * (x - x0))) ** v)
    return y


def save_monthly_params_dataset(station, year, month, data, type_):
    last = (calendar.monthrange(year, month)[1])
    start_dt = datetime(year, month, 1)
    end_dt = datetime(year, month, last) + timedelta(days=1) - timedelta(seconds=station.freq)
    gen_source_path = get_month_gen_params_path(station, start_dt, type_=type_)
    month_slice = slice(start_dt, end_dt)
    xr_utils.save_dataset(dataset=data.sel(Time=month_slice), nc_path=gen_source_path)


def save_full_params_dataset(station, start_date, end_date, data, type_):
    folder_name = station.generation_folder
    nc_name = f"generated_{type_}_{station.name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.nc"
    xr_utils.save_dataset(data, folder_name, nc_name)

def valid_box_domain(x, y, bounds_x, bounds_y):
    return bounds_x[0] <= x <= bounds_x[1] and bounds_y[0] <= y <= bounds_y[1]