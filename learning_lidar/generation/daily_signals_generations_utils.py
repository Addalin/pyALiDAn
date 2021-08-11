import datetime
import logging
import os
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pytictoc import TicToc
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

import learning_lidar.generation.generation_utils as gen_utils
import learning_lidar.preprocessing.preprocessing_utils as prep_utils
import learning_lidar.utils.global_settings as gs
import learning_lidar.utils.vis_utils as vis_utils
import learning_lidar.utils.xr_utils as xr_utils
from learning_lidar.generation.generation_utils import PLOT_RESULTS
from learning_lidar.utils.misc_lidar import calc_tau, generate_poisson_signal_STEP
from learning_lidar.utils.utils import create_and_configer_logger
from learning_lidar.utils.vis_utils import TIMEFORMAT

logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
vis_utils.set_visualization_settings()
wavelengths = gs.LAMBDA_nm().get_elastic()


def calc_total_optical_density(station: gs.Station, day_date: datetime.date) -> xr.Dataset:
    """
    Generate total backscatter and extinction profiles
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: xr.Dataset() containing the daily total backscatter and extinction profiles
        such that:
            - beta = beta_aer + beta_mol
            - sigma = sigma_aer + sigma_mol
        The datasets' variable, share 3 dimensions : 'Wavelength', 'Height', 'Time'
    """
    # %% 1. Load generated aerosol profiles
    month_folder = prep_utils.get_month_folder_name(station.gen_aerosol_dataset, day_date)
    nc_aer = gen_utils.get_gen_dataset_file_name(station, day_date, data_source='aerosol')
    aer_ds = xr_utils.load_dataset(os.path.join(month_folder, nc_aer))

    if PLOT_RESULTS:
        height_slice = slice(0.0, 15)
        vis_utils.plot_daily_profile(profile_ds=aer_ds.sigma, height_slice=height_slice)
        vis_utils.plot_daily_profile(profile_ds=aer_ds.beta, height_slice=height_slice)

    # %% 2. Load molecular profiles
    month_folder = prep_utils.get_month_folder_name(station.molecular_dataset, day_date)
    nc_name = xr_utils.get_prep_dataset_file_name(station, day_date, data_source='molecular', lambda_nm='all')
    ds_mol = xr_utils.load_dataset(os.path.join(month_folder, nc_name))

    # %% 3. Calculate total densities
    total_sigma = (aer_ds.sigma + ds_mol.sigma).assign_attrs({'info': "Daily total extinction coefficient",
                                                              'long_name': r'$\sigma$', 'units': r'$1/km$',
                                                              'name': 'sigma'})
    total_beta = (aer_ds.beta + ds_mol.beta).assign_attrs({'info': "Daily total backscatter coefficient",
                                                           'long_name': r'$\beta$', 'units': r'$1/km$',
                                                           'name': 'beta'})

    total_ds = xr.Dataset().assign(sigma=total_sigma, beta=total_beta)
    total_ds.attrs = {'info': 'Daily generated atmosphere profiles',
                      'source_file': os.path.basename(__file__),
                      'location': station.location, }
    total_ds.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    total_ds.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    total_ds = total_ds.transpose('Wavelength', 'Height', 'Time')
    total_ds['date'] = day_date

    if PLOT_RESULTS:
        vis_utils.plot_daily_profile(profile_ds=total_ds.sigma)
        vis_utils.plot_daily_profile(profile_ds=total_ds.beta)

    return total_ds


def calc_attbsc_da(station: gs.Station, day_date: datetime.date, total_ds: xr.Dataset) -> xr.DataArray:
    """
    Calculating the attenuated backscatter: attbsc = beta*exp(-2*tau)
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param total_ds: xr.Dataset(). The total backscatter and extinction daily profiles.
    :return: attbsc_da: xr.DataArray(). The daily attenuated backscatter profile.
    Having 3 dimensions : 'Wavelength', 'Height', 'Time'
    """
    logger = logging.getLogger()
    height_bins = station.get_height_bins_values()
    exp_tau_c = []
    logger.debug(f"\nCalculating Attenuated Backscatter for {day_date.strftime('%Y-%m-%d')}")
    for wavelength in wavelengths:
        exp_tau_t = []
        for t in tqdm(total_ds.Time, desc=f"Wavelength - {wavelength} [nm]"):
            sigma_t = total_ds.sigma.sel(Time=t)
            e_tau = xr.apply_ufunc(lambda x: np.exp(-2 * calc_tau(x, height_bins)),
                                   sigma_t.sel(Wavelength=wavelength), keep_attrs=True)
            e_tau.name = r'$\exp(-2 \tau)$'
            exp_tau_t.append(e_tau)

        exp_tau_c.append(xr.concat(exp_tau_t, dim='Time'))

    exp_tau_d = xr.concat(exp_tau_c, dim='Wavelength')
    exp_tau_d = exp_tau_d.transpose('Wavelength', 'Height', 'Time')

    attbsc_da = (exp_tau_d * total_ds.beta)
    attbsc_da.attrs = {'info': "Daily total attenuated backscatter coefficient",
                       'long_name': r'$\beta \cdot \exp(-2\tau)$',
                       'units': r'$1/km$', 'name': 'attbsc',
                       'location': station.location, }
    attbsc_da.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    attbsc_da.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    attbsc_da['date'] = day_date

    if PLOT_RESULTS:
        vis_utils.plot_daily_profile(profile_ds=attbsc_da, figsize=(16, 8))

    return attbsc_da


def get_daily_LC(station: gs.Station, day_date: datetime.date) -> xr.DataArray:
    """
    Load daily generated Lidar power factor
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: xr.DataArray() of daily generated Lidar power factor
    """
    # %% 2.
    ds_gen_p = gen_utils.get_daily_gen_param_ds(station, day_date, type_='LC')
    day_slice = slice(day_date, day_date + timedelta(days=1) - timedelta(seconds=30))
    lc_da = ds_gen_p.p.sel(Time=day_slice)
    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        lc_da.plot(hue='Wavelength', linewidth=0.8)
        ax.set_title(f"{lc_da.info} - for {day_date.strftime('%d/%m/%Y')}")
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()
    return lc_da


def get_daily_bg(station: gs.Station, day_date: datetime.date) -> xr.DataArray:
    """
    Load daily generated background signal
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: xr.DataArray() of daily generated background signal. Having coordinates of: 'Wavelength','Time' (1D signal!)
    """
    bg_1D_ds = gen_utils.get_daily_gen_param_ds(station, day_date, type_='bg')
    day_slice = slice(day_date, day_date + timedelta(days=1) - timedelta(seconds=30))
    bg_1D_ds = bg_1D_ds.sel(Time=day_slice)
    p_bg = bg_1D_ds.bg

    if PLOT_RESULTS:
        fig, ax = plt.subplots(ncols=1, nrows=1)
        p_bg.sel(Time=day_slice).plot(hue='Wavelength', ax=ax, linewidth=0.8)
        ax.set_xlim([day_slice.start, day_slice.stop])
        ax.set_title(f"{p_bg.info} - {day_date.strftime('%d/%m/%Y')}")
        plt.xticks(rotation=0)
        ax.set_ybound([-.01, 2])
        plt.tight_layout()
        plt.show()

    return p_bg


def calc_range_corr_signal_da(station: gs.Station, day_date: datetime.date, attbsc_da: xr.DataArray,
                              lc_da: xr.DataArray) -> xr.DataArray:
    """
    Calculating daily range corrected signal: pr2 = LC * attbsc
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param attbsc_da: xr.DataArray(). The daily attenuated backscatter profile.
    :param lc_da: xr.DataArray() of daily generated Lidar power factor
    :return: pr2_ds: xr.DataArray(). The daily range corrected signal,
     with share 3 dimensions : 'Wavelength', 'Height', 'Time'
    """
    logger = logging.getLogger()
    logger.info(f"\nCalculating Range corrected signal for {day_date.strftime('%Y-%m-%d')}")
    pr2_c = []
    for wavelength in wavelengths:
        pr2_t = []
        for t in tqdm(attbsc_da.Time, desc=f"Wavelength - {wavelength} [nm]"):
            attbsc_t = attbsc_da.sel(Time=t)
            LC_t = lc_da.sel(Time=t)
            pr2 = xr.apply_ufunc(lambda x, y: (x * y),
                                 attbsc_t.sel(Wavelength=wavelength),
                                 LC_t.sel(Wavelength=wavelength), keep_attrs=True)
            pr2.name = r'$pr2$'
            pr2_t.append(pr2)
        pr2_c.append(xr.concat(pr2_t, dim='Time'))

    pr2_da = xr.concat(pr2_c, dim='Wavelength')
    pr2_da = pr2_da.transpose('Wavelength', 'Height', 'Time')
    pr2_da.attrs = {'info': 'Generated Range Corrected Lidar Signal',
                    'long_name': r'$LC \beta \cdot \exp(-2\tau)$', 'name': 'range_corr',
                    'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$',
                    'location': station.location, }
    attbsc_da.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    attbsc_da.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    attbsc_da['date'] = day_date
    if PLOT_RESULTS:
        vis_utils.plot_daily_profile(profile_ds=pr2_da, figsize=(16, 8))

    return pr2_da


def calc_lidar_signal_da(station: gs.Station, day_date: datetime.date, r2_da: xr.DataArray,
                         pr2_da: xr.DataArray) -> xr.DataArray:
    """
    Calculates daily lidar signal: p = pr2 / r^2
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param r2_da: xr.DataArray(). The heights bins squared.
    :param pr2_da: xr.DataArray(). The daily range corrected signal
    :return: p_da: xr.DataArray(). The daily lidar signal, having 3 dimensions : 'Wavelength', 'Height', 'Time'
    """
    logger = logging.getLogger()
    p_da = (pr2_da / r2_da)
    p_da.attrs = {'info': 'Generated Lidar Signal',
                  'long_name': r'$p$', 'name': 'p',
                  'units': r'$\rm$' + r'$photons$',
                  'location': station.location, }
    p_da.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    p_da.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    p_da['date'] = day_date
    # sanity check:
    # TODO : Does this test require try/catch outside the function?
    mask_valid = (~pd.isna(p_da.values))
    valid_size = mask_valid.astype(int).sum()
    if p_da.size != valid_size:
        msg = f"The daily lidar signal contains NaN values - {day_date}"
        logger.error(msg)
        raise ValueError(msg)

    if PLOT_RESULTS:
        vis_utils.plot_daily_profile(profile_ds=p_da, height_slice=slice(0, 5), figsize=(16, 8))
    return p_da


def calc_lidar_signal(station: gs.Station, day_date: datetime.date, total_ds: xr.Dataset) -> xr.Dataset:
    """
    Generate daily lidar signal, using the optical densities and the LC (Lidar Constant)
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param total_ds: xr.Dataset(). The total backscatter and extinction daily profiles
    :return: signal_ds: xr.Dataset(). Containing the daily lidar signal (clean)
    """
    attbsc_da = calc_attbsc_da(station, day_date, total_ds)  # attbsc = beta*exp(-2*tau)
    lc_da = get_daily_LC(station, day_date)  # LC
    pr2_da = calc_range_corr_signal_da(station, day_date, attbsc_da, lc_da)  # pr2 = LC * attbsc
    r2_da = prep_utils.calc_r2_da(station, day_date)  # r^2
    p_da = calc_lidar_signal_da(station, day_date, r2_da, pr2_da)  # p = pr2 / r^2
    pn_da = calc_poiss_measurement(station, day_date, p_da)  # lidar measurement: pn ~Poiss(p), w.o background
    pr2n_da = calc_range_corr_measurement(station, day_date, pn_da,
                                          r2_da)  # range corrected measurement: pr2n = pn * r^2
    pr2n_da.attrs['info'] += ' - w.o. background'

    signal_ds = xr.Dataset().assign(attbsc=attbsc_da, LC=lc_da,
                                    range_corr=pr2_da, range_corr_p=pr2n_da,
                                    p=p_da, r2=r2_da)
    signal_ds['date'] = day_date
    signal_ds.attrs = {'location': station.location,
                       'info': 'Daily generated lidar signals.',
                       'source_file': os.path.basename(__file__)}
    return signal_ds


def calc_mean_measurement(station: gs.Station, day_date: datetime.date, p_da: xr.DataArray,
                          bg_da: xr.DataArray) -> xr.DataArray:
    """
    Calculate mean signal measurement: mu_p = p_bg + p
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param p_da: xr.DataArray(). Containing the daily lidar signal (clean)
    :param bg_da: xr.DataArray(). The daily generated background signal
    :return: xr.DataArray(). The daily mean measurement of the lidar signal
    """
    p_mean = p_da + bg_da
    p_mean.attrs = {'info': 'Daily averaged lidar signal:',  # +r'$\mu_{p} =\mathbb{E}(p)+<p_{bg}>$',
                    'long_name': r'$\mu_{p}$', 'name': 'pmean',
                    'units': r'$\rm$' + r'$photons$',
                    'location': station.location, }
    p_mean.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    p_mean.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    p_mean['date'] = day_date
    if PLOT_RESULTS:
        vis_utils.plot_daily_profile(p_mean.where(p_mean < 20), height_slice=slice(0, 10))
    return p_mean


def calc_poiss_measurement(station: gs.Station, day_date: datetime.date, p_mean: xr.DataArray) -> xr.DataArray:
    """
    Calculate lidar signal measurement: pn ~ Poiss(mu_p)
        $P_{measure}\simPoiss(\mu_{p} ) $
        Note:for $\mu_{p} > 50$: $Poiss(\mu_{p}) = \mu_{p} + \sqrt{\mu_{p}}\cdot  \mathcal {N}(0, 1)$
        This is to save time and power of computations
        The  poisson  distribution  calculated only for values lower then 50 - to assure we dont get negative values
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param p_mean: xr.DataArray(). The daily mean measurement of the lidar signal
    :return: pn_da: xr.DataArray(). The daily lidar signal measurement.
    """
    logger = logging.getLogger()
    logger.info(f"\nCalculating Poisson signal for {day_date.strftime('%Y-%m-%d')}")
    tic0 = TicToc()
    tic0.tic()
    pn_h = xr.apply_ufunc(
        lambda mu: (mu + (np.sqrt(mu)) * np.random.normal(loc=0, scale=1.0, size=mu.shape)).astype(int),
        p_mean.where(p_mean >= 50).fillna(0), keep_attrs=True)
    pn_l = xr.apply_ufunc(lambda mu: generate_poisson_signal_STEP(mu),
                          p_mean.where(p_mean < 50).fillna(0), keep_attrs=True, dask='parallelized')
    tic0.toc()
    pn_da = pn_h + pn_l
    pn_da.attrs = {'info': 'Generated Poisson Lidar Signal',
                   'long_name': r'$p$', 'name': 'pn',
                   'units': r'$\rm$' + r'$photons$',
                   'location': station.location, }
    pn_da.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    pn_da.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    pn_da['date'] = day_date

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            pn_da.where(pn_da >= 3).sel(Wavelength=wavelength,
                                        Height=slice(0, 10)) \
                .plot(cmap='turbo', ax=ax)
            ax.xaxis.set_major_formatter(TIMEFORMAT)
            ax.xaxis.set_tick_params(rotation=0)
        plt.suptitle(f"{pn_da.info}")
        plt.tight_layout()
        plt.show()

    return pn_da


def calc_range_corr_measurement(station: gs.Station, day_date: datetime.date, pn_da: xr.DataArray,
                                r2_da: xr.DataArray) -> xr.DataArray:
    """
    Calculate generated range corrected measure: pr2n = pn * r^2
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param pn_da: xr.DataArray(). The daily lidar signal measurement.
    :param r2_da: xr.DataArray(). The heights bins squared.
    :return: pr2n_da: xr.DataArray(). The daily Generated Poisson Range Corrected Lidar Signal
    """
    pr2n_da = (pn_da.copy(deep=True) * r2_da)
    pr2n_da.attrs = {'info': 'Generated Poisson Range Corrected Lidar Signal',
                     'long_name': r'$p$' + r'$\cdot r^2$', 'name': 'range_corr',
                     'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$',
                     'location': station.location, }
    pr2n_da.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    pr2n_da.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    pr2n_da['date'] = day_date

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            pr2n_da.where(pr2n_da >= 3).sel(Wavelength=wavelength, Height=slice(0, 10)) \
                .plot(cmap='turbo', ax=ax)
            ax.xaxis.set_major_formatter(TIMEFORMAT)
            ax.xaxis.set_tick_params(rotation=0)
        plt.suptitle(f"{pr2n_da.info}")
        plt.tight_layout()
        plt.show()
        plt.show()

    return pr2n_da


def calc_daily_measurement(station: gs.Station, day_date: datetime.date, signal_ds: xr.Dataset,
                           update_overlap_only: bool = False) -> xr.Dataset:
    """
    Generate Lidar measurement, by combining background signal and the lidar signal,
    and then creating Poisson signal, which is the measurement of the mean lidar signal.

    If update_overlap_only is given, an existing ds is loaded for the lidar and signal, instead of computed from scratch

    :param update_overlap_only: bool, whether to load a precomputed lidar and signal dataset or not
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param signal_ds: xr.Dataset(), containing the daily lidar signal (clean)
    :return: measure_ds: xr.Dataset(), containing the daily lidar measurement (with background and applied photon noise)
    """
    # get the ingredients
    if update_overlap_only:
        signal_ds = gen_utils.get_daily_gen_ds(station, day_date, type_='signal')

    # apply overlap on the lidar signal
    overlap_ds = gen_utils.get_daily_overlap(station, day_date, height_indx=signal_ds.Height)
    p_da = xr.apply_ufunc(lambda x, r: (x * r),
                          signal_ds.p, overlap_ds.overlap,
                          keep_attrs=True).assign_attrs({'info': signal_ds.p.info + ' with applied overlap'})

    # add background signal
    p_bg = get_daily_bg(station, day_date)  # daily background: p_bg
    # Expand p_bg to coordinates : 'Wavelength','Height', 'Time
    bg_da = p_bg.broadcast_like(signal_ds.range_corr)

    # Calculate the total signal
    p_mean = calc_mean_measurement(station, day_date, p_da, bg_da)
    pn_da = calc_poiss_measurement(station, day_date, p_mean)  # lidar measurement: pn ~Poiss(mu_p)
    r2_da = prep_utils.calc_r2_da(station, day_date)

    pr2n_da = calc_range_corr_measurement(station, day_date, pn_da,
                                          r2_da)  # range corrected measurement: pr2n = pn * r^2
    measure_ds = xr.Dataset().assign(p=pn_da, range_corr=pr2n_da, p_mean=p_mean,
                                     p_bg=bg_da, overlap=overlap_ds.overlap)
    measure_ds['date'] = day_date
    measure_ds.attrs = {'location': station.location,
                        'info': 'Daily generated lidar signals measurement.',
                        'source_file': os.path.basename(__file__)}
    # TODO: Add plots of bg_da, p_mean, pr2n_da,pn_da
    # gen_utils.plot_daily_profile(measure_ds.p_bg)
    # gen_utils.plot_daily_profile(measure_ds.range_corr.where(measure_ds.range_corr>=3))
    return measure_ds


# %% Analysis part

def explore_orig_day(main_folder, station_name, start_date, end_date, day_date, timedelta, wavelengths, time_indx):
    # TODO - organize this part and move to gen_utils.py or Analysis notebook
    day_str = day_date.strftime('%Y-%m-%d')
    ds_path_extended = os.path.join(main_folder, 'data',
                                    f"dataset_{station_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.nc")
    csv_path_extended = os.path.join(main_folder, 'data',
                                     f"dataset_{station_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.csv")
    df = pd.read_csv(csv_path_extended)
    ds_extended = xr_utils.load_dataset(ds_path_extended)

    # %%
    day_slice = slice(day_date, day_date.date() + timedelta(days=1))
    arr_day = []
    for wavelength in wavelengths:
        ds_i = ds_extended.sel(Wavelength=wavelength, Time=day_slice)
        ds_i = ds_i.resample(Time='30S').interpolate('nearest')
        ds_i = ds_i.reindex({"Time": time_indx}, method="nearest", fill_value=0)
        arr_day.append(ds_i)
    ds_day = xr.concat(arr_day, dim='Wavelength')
    # %%

    # %% Visualize parameters
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ds_day.LC.plot(x='Time', hue='Wavelength', ax=ax)
    ax.set_title(fr"{ds_day.LC.long_name} for {day_str}")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for r in ['r0', 'r1', 'rm']:
        ds_day[r].sel(Wavelength=355).plot(label=ds_day[r].long_name, ax=ax)
    ax.set_title(fr'Reference Range - {day_str}')
    ax.set_ylabel(r'$\rm Height[km]$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    ds_smooth = ds_day.apply(func=gaussian_filter1d, sigma=80, keep_attrs=True)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ds_day.LC.plot(x='Time', hue='Wavelength', ax=ax, linewidth=0.8, linestyle='--')
    ds_smooth.LC.plot(x='Time', hue='Wavelength', ax=ax, linewidth=0.8, linestyle='-.')
    ds_extended.sel(Time=slice(day_date, day_date + timedelta(hours=24))).LC.plot(hue='Wavelength', linewidth=0.8)
    ds_extended.sel(Time=slice(day_date, day_date + timedelta(hours=24))). \
        plot.scatter(y='LC', x='Time', hue='Wavelength', s=30, hue_style='discrete', edgecolor='w')

    ax.set_title(fr"{ds_day.LC.long_name} for {day_str}")
    plt.tight_layout()
    plt.show()
