import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

import learning_lidar.preprocessing.preprocessing as prep
from datetime import datetime, timedelta
import os
import calendar

from learning_lidar.generation.generate_density_utils import PLOT_RESULTS
from learning_lidar.utils.global_settings import TIMEFORMAT
from tqdm import tqdm
import pandas as pd

def get_gen_dataset_file_name(station, day_date, wavelength='*',
                              data_source='lidar', file_type='range_corr',
                              time_slice=None):
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


def save_generated_dataset(station, dataset, data_source='lidar', save_mode='both',
                           profiles=None, time_slices=None):
    """
    Save the input dataset to netcdf file
    If this name is not provided, then the first profile is automatically selected
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
    date_datetime = prep.get_daily_ds_date(dataset)
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
    month_folder = prep.get_month_folder_name(base_folder, date_datetime)

    prep.get_daily_ds_date(dataset)
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
                        ncpath = prep.save_dataset(ds_slice, month_folder, file_name)
                        if ncpath:
                            ncpaths.append(ncpath)
                else:
                    file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                                          wavelength=wavelength, file_type=profile)
                    ncpath = prep.save_dataset(ds_profile, month_folder, file_name)
                    if ncpath:
                        ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                              wavelength='*')
        ncpath = prep.save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


def get_month_gen_params_path(station, day_date, type='density_params'):
    """
    :param type: type of generated parameter:
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

    nc_name = f"generated_{type}_{station.location}_{month_start_day.strftime('%Y-%m-%d')}_" \
              f"{month_end_day.strftime('%Y-%m-%d')}.nc"

    folder_name = prep.get_month_folder_name(station.generation_folder, day_date)

    gen_source_path = os.path.join(folder_name, nc_name)
    return gen_source_path


def get_month_gen_params_ds(station, day_date, type='density_params'):
    """
    Returns the monthly parameters of density creation as a dataset.
    :param type: type of generated parameter:
        'density_params' - for density sampler generator,
        'bg'- for generated background signal,
        'LC' - for generated Lidar Constant signal
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Monthly dataset of generation parameters.
    """

    gen_source_path = get_month_gen_params_path(station, day_date, type)
    month_params_ds = prep.load_dataset(gen_source_path)
    return month_params_ds


def get_daily_gen_param_ds(station, day_date, type='density_params'):
    """
    Returns the daily parameters of density creation as a dataset.
    :param type: type of generated parameter:
        'density_params' - for density sampler generator,
        'bg'- for generated background signal,
        'LC' - for generated Lidar Constant signal
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: day_params_ds: xarray.Dataset(). Daily dataset of generation parameters.
    """
    month_params_ds = get_month_gen_params_ds(station, day_date, type)
    day_params_ds = month_params_ds.sel(Time=slice(day_date, day_date + timedelta(days=1)))

    return day_params_ds


def dt2binscale(dt_time, res_sec=30):
    """
    TODO consider to move this function to Station ?
    Returns the bin index corresponds to dt_time
    binscale - is the time scale [0,2880], of a daily lidar bin index from 00:00:00 to 23:59:30.
    The lidar has a bin measure every 30 sec, in total 2880 bins per day.
    :param dt_time: datetime.datetime object
    :return: binscale - float in [0,2880]
    """
    res_minute = 60 / res_sec
    res_hour = 60 * res_minute
    res_musec = (1e-6) / res_sec
    tind = dt_time.hour * res_hour + \
           dt_time.minute * res_minute + \
           dt_time.second / res_sec + \
           dt_time.microsecond * res_musec
    return tind


def plot_daily_profile(profile_ds, height_slice=None, figsize=(16, 6)):
    # TODO : move to vis_utils.py
    # TODO: add scintific ticks on colorbar
    wavelengths = profile_ds.Wavelength.values
    if height_slice is None:
        height_slice = slice(profile_ds.Height[0].values, profile_ds.Height[-1].values)
    str_date = profile_ds.Time[0].dt.strftime("%Y-%m-%d").values.tolist()
    ncols = wavelengths.size
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
    if ncols > 1:
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            profile_ds.sel(Height=height_slice, Wavelength=wavelength).plot(cmap='turbo', ax=ax)
            ax.xaxis.set_major_formatter(TIMEFORMAT)
            ax.xaxis.set_tick_params(rotation=0)
    else:
        ax = axes
        profile_ds.sel(Height=height_slice).plot(cmap='turbo', ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
    plt.suptitle(f"{profile_ds.info} - {str_date}")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


def plot_hourly_profile(profile_ds, height_slice=None, figsize=(10, 6), times=None):
    # TODO : move to vis_utils.py
    # TODO: add scientific ticks on color-bar
    day_date = prep.dt64_2_datetime(profile_ds.Time[0].values)
    str_date = day_date.strftime("%Y-%m-%d")
    if times == None:
        times = [day_date + timedelta(hours=8),
                 day_date + timedelta(hours=12),
                 day_date + timedelta(hours=18)]
    if height_slice is None:
        height_slice = slice(profile_ds.Height[0].values, profile_ds.Height[-1].values)

    ncols = len(times)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
    for t, ax in zip(times,
                     axes.ravel()):
        profile_ds.sel(Time=t, Height=height_slice).plot.line(ax=ax, y='Height', hue='Wavelength')
        ax.set_title(pd.to_datetime(str(t)).strftime('%H:%M:%S'))
    plt.tight_layout()
    plt.suptitle(f"{profile_ds.info} - {str_date}")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.show()


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
        t_start = 95/total_bins
        r_start = 1
        t_overlap = np.array(
            [0.0, 0.2 * t_start, 0.5 * t_start, .75 * t_start, t_start, 0.05, 0.1, .3, .4, .5, .6, .7, .8, .9,
             1.0]) * np.float(ref_height_bin)
        ratios = np.array([.0, 0.2* r_start,0.5 * r_start, 0.75 * r_start, r_start, 1, 1, 1, 1, 1, 1, 1, 1, 1,1])
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