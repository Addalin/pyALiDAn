from matplotlib import pyplot as plt
import learning_lidar.preprocessing.preprocessing as prep
from datetime import datetime, date, timedelta
import os
import calendar
from learning_lidar.generation.daily_signals_generations_utils import wavelengths
from learning_lidar.utils.global_settings import TIMEFORMAT


def get_gen_dataset_file_name(station, day_date, wavelength='*', data_source='lidar', file_type='range_corr'):
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

    :return: dataset file name (netcdf) file of the data_type required per given day, wavelength, data_source and file_type
    """
    if wavelength == '*':
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{data_source}.nc"
    else:
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{file_type}_{wavelength}_{data_source}.nc"

    return file_name


def save_generated_dataset(station, dataset, data_source='lidar', save_mode='both', profiles=None):
    """
    Save the input dataset to netcdf file
    If this name is not provided, then the first profile is automatically selected
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily generated lidar signal, holding 5 data variables:
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
    tind = dt_time.hour * res_hour +\
           dt_time.minute * res_minute + \
           dt_time.second / res_sec + \
           dt_time.microsecond * res_musec
    return tind


def plot_daily_profile(data, height_slice=None, figsize=(16, 6)):
    # TODO : move to vis_utils.py
    # TODO: add low/high thresholds (single or per channel) see prep.visualize_ds_profile_chan()
    if height_slice is None:
        height_slice = slice(data.Height[0].values, data.Height[-1].values)
    str_date = data.Time[0].dt.strftime("%Y-%m-%d").values.tolist()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
    for wavelength, ax in zip(wavelengths, axes.ravel()):
        data.sel(Height=height_slice, Wavelength=wavelength).plot(cmap='turbo', ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
    plt.suptitle(f"{data.info} - {str_date}")
    plt.tight_layout()
    plt.show()