import os
from datetime import datetime, timedelta, time

import astral
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import tz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import dates as mdates
from learning_lidar.utils import utils, vis_utils, global_settings as gs
from learning_lidar.utils.misc_lidar import calc_gauss_curve


def gauss_fit(x, y, mean_0=None, sigma_0=None):
    if mean_0:
        mean = mean_0
    else:
        mean = np.mean(x)
    if sigma_0:
        sigma = sigma_0
    else:
        sigma = np.std(y) ** 2
    initial_guess = [np.mean(y), max(y), mean, sigma]
    popt, pcov = curve_fit(calc_gauss_curve, x, y, p0=initial_guess, absolute_sigma=True)
    return popt


def func_log(x, a, b, c, d):
    """Return values from a general log function."""
    return a * np.log(b * x + c) + d


def func_cos(x, a, b, c, d):
    """Return values from a general cos function."""
    return a * np.cos(np.deg2rad(b * x) + c) + d

def func_cos2(x, a, b, c):
    """Return values from a general cos function."""
    return a + b * np.cos(np.deg2rad(x)) + c*(np.cos(np.deg2rad(x))**2)

def calc_gauss_width(min_val, max_val, rel_val, FWRM):
    max_rel_ratio = (rel_val - min_val) / (max_val - min_val)
    W = FWRM / (2 * np.sqrt(2 * np.log(1 / max_rel_ratio)))
    return W


def get_params(date, lat, lon):
    loc = astral.Location(('Greenwich', 'England', lat, lon, 'Europe/London'))
    loc.solar_depression = 'nautical'
    sun = loc.sun(date)
    day_light = loc.daylight(date)
    day_len = str(day_light[1] - day_light[0])  # converts from timedelta seconds to HH:mm:ss
    sun_alt = loc.solar_elevation(date)

    return sun['noon'], day_len, sun_alt


def utc2tzloc(utc_dt, location):
    """
    Convert time zone from UTC to the locations' time zone
    :param utc_dt: datetime.datetime(), UTC time to convert
    :param location: astral.LocationInfo(), location info
    :return: datetime.datetime() , time in locations' time zone
    """
    to_zone = tz.gettz(location.region)
    from_zone = tz.gettz('UTC')
    utc_dt = utc_dt.replace(tzinfo=from_zone)
    loc_dt = utc_dt.astimezone(to_zone)
    return loc_dt


def tzloc2utc(loc_dt, location):
    """
    Convert time zone from input location to UTC
    :param loc_dt: datetime.datetime(), time in location time zone to convert
    :param location: astral.LocationInfo(), location info
    :return: datetime.datetime() , time in UTC time zone
    """
    to_zone = tz.gettz('UTC')
    from_zone = tz.gettz(location.region)
    loc_dt = loc_dt.replace(tzinfo=from_zone)
    utc_dt = loc_dt.astimezone(to_zone)
    return utc_dt


def dt2binscale(dt_time, res_sec=30):
    """
    Returns the bin index corresponds to dt_time
    binscale - is the time scale [0,2880], of a daily lidar bin index from 00:00:00 to 23:59:30.
    The lidar has a bin measure every 30 sec, in total 2880 bins per day.
    :param dt_time: datetime.datetime object
    :return: binscale - float in [0,2880]
    """

    res_minute = 60 / res_sec
    res_hour = 60 * res_minute
    res_musec = 1e-6 / res_sec
    tind = dt_time.hour * res_hour + dt_time.minute * res_minute + dt_time.second / res_sec + dt_time.microsecond * res_musec
    return tind


def binscale2dt(tind, day_date=datetime.today(), res_sec=30):
    dt_time = day_date + timedelta(seconds=res_sec) * tind
    return dt_time


def dt_delta2time(dt_delta):
    hours, remainder = divmod(dt_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    m_seconds = dt_delta.microseconds
    dt_time = time(hours, minutes, seconds, m_seconds)
    return dt_time


def plot_sun_elevation_at_noon_times(ds_year):
    # Maximum sun elevation during 2017:
    max_year_elevation = np.max(ds_year.sunelevation.values)
    day_max_num = np.argmax(ds_year.sunelevation.values)
    day_max = ds_year.Time.values[day_max_num]
    day_max_str = utils.dt64_2_datetime(day_max).strftime('%Y-%m-%d')

    # Minimum sun elevation during 2017:
    min_year_elevation = np.min(ds_year.sunelevation.values)
    day_min_num = np.argmin(ds_year.sunelevation.values)
    day_min = ds_year.Time.values[day_min_num]
    day_min_str = utils.dt64_2_datetime(day_min).strftime('%Y-%m-%d')

    # %%Plot sun elevation at noon times
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ds_year.sunelevation.plot(ax=ax)
    ax.annotate(fr'{day_max_str}', fontweight='bold', fontsize=12,
                xy=(day_max, max_year_elevation - 10), color='darkgreen', va="center",
                xytext=(ds_year.Time.values[day_max_num - 35], 70),
                bbox=dict(boxstyle='round,pad=0.2', ec="none", fc=[1, 1, 1]), )

    ax.annotate(fr'{day_min_str}',
                fontweight='bold', fontsize=12, color='darkgreen',
                xy=(day_min, 30 + min_year_elevation), va="center",
                bbox=dict(boxstyle='round,pad=0.2', ec="none", fc=[1, 1, 1]),
                xytext=(ds_year.Time.values[day_min_num - 35], 70), )

    ax.annotate(fr'{min_year_elevation:.2f}$^\circ$',
                fontweight='bold', fontsize=12, color='darkmagenta',
                xy=(day_min, min_year_elevation), va="center",
                bbox=dict(boxstyle='round,pad=0.2', fc=[1, 1, 1], ec="none"),
                xytext=(ds_year.Time.values[0], min_year_elevation))

    ax.annotate(fr'{max_year_elevation:.2f}$^\circ$',
                fontweight='bold', fontsize=12, color='darkmagenta',
                xy=(day_max, max_year_elevation), va="center",
                bbox=dict(boxstyle='round,pad=0.2', fc=[1, 1, 1], ec="none"),
                xytext=(ds_year.Time.values[0], max_year_elevation))

    ax.axhline(y=min_year_elevation, xmin=0.0, xmax=1.0, alpha=.6,
               color='darkmagenta', linestyle='--', linewidth=1)
    ax.axhline(y=max_year_elevation, xmin=0.0, xmax=1.0, alpha=.6,
               color='darkmagenta', linestyle='--', linewidth=1)
    ax.axvline(x=ds_year.Time.values[day_min_num], ymin=0.0, ymax=1.0, alpha=.6,
               color='darkgreen', linestyle='--', linewidth=1)
    ax.axvline(x=ds_year.Time.values[day_max_num], ymin=0.0, ymax=1.0, alpha=.6,
               color='darkgreen', linestyle='--', linewidth=1)
    ax.xaxis.set_major_formatter(vis_utils.MONTHFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.title('Sun elevation at noon times during 2017')
    plt.tight_layout()
    plt.show()


def fit_curve_and_plot_sun_elevation_during_day(loc, day_sun, day_0, ds_day, bins_per_day):
    day_light_sun = ds_day.sunelevation.copy(deep=True)
    # #### Gaussian curve fit of sun elevation during daylight hours:
    #  - Fitting to : $y(t) = A + H \cdot \exp \bigg( -\frac {(t - t_0) ^ 2}{2 \cdot W ^2}\bigg) $
    #  - Initial guess : $t_0=t_{\rm noon-TST}$
    #  - Initial guess : $ W = \frac{\Delta t_{\rm daylight}}{2\sqrt({2\ln({\frac{1}{\alpha_{\rm ratio}}})}) }$,
    #  $\alpha_{\rm ratio} =\frac{\alpha_{\rm twilight} - \alpha_{\rm min}}{\alpha_{\rm noon}-\alpha_{\rm min}}$
    # > Assuming:
    # > - A parametric curve $t$ with 2880 bins (similar to lidar's bins per day).
    # > - $\alpha_{\rm twilight}=-6^\circ$
    # > - $\Delta t_{\rm daylight} = t_{\rm dusk} -  t_{\rm dawn}$

    t = np.arange(0, bins_per_day)

    #  Initial guess of mean - at noon time / or when the sun elevation is at maximum angle.
    dawn_dusk_angle = -6
    day_light_sun[day_light_sun < dawn_dusk_angle] = dawn_dusk_angle
    cur_day = datetime.combine(day_sun['noon'].date(), datetime.min.time())
    MST_noon = cur_day + timedelta(hours=12)
    MST_noon = tzloc2utc(MST_noon, loc)
    TST_noon = day_sun['noon']
    mean0 = dt2binscale(TST_noon, res_sec=30)

    #  Initial guess of std (width of curve) - using width gaussian properties:
    dawn_to_dusk = (day_sun['dusk'] - day_sun['dawn'])
    min_angle = ds_day.sunelevation.values.min()
    max_angle = ds_day.sunelevation.values.max()
    sigma0 = calc_gauss_width(min_angle, max_angle, dawn_dusk_angle, dt2binscale(dawn_to_dusk + day_0))
    #  Gaussian fit to sun elevation
    y = ds_day.sunelevation.values.copy()
    A3, H3, x0_3, sigma3 = gauss_fit(t, y, mean_0=mean0, sigma_0=sigma0)
    fit3 = calc_gauss_curve(t, *gauss_fit(t, y, mean_0=mean0, sigma_0=sigma0))

    print(f"Sigma fit:{sigma3 :.2f}, Sigma init estimated:{sigma0 :.2f} ")
    print(f"Mean fit:{x0_3 :.2f}, Mean init estimated:{mean0 :.2f} ")

    #  Plot curve and fit
    sns.set_palette(sns.color_palette("tab10"))
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,5))
    ds_day.sunelevation.plot(ax=ax, label=r'$\alpha_{\rm sun}$', color='darkblue', linewidth=2.0)
    ax.fill_between(ds_day.Time, day_light_sun, ds_day.sunelevation, alpha=0.2)
    ax.fill_between(ds_day.Time, day_light_sun, dawn_dusk_angle, alpha=0.2)
    ax.fill_between(ds_day.Time, ds_day.sunelevation, day_light_sun, alpha=0.1)
    ax.plot(ds_day.Time, fit3, '-.', label=r'$\alpha_{\rm sun}$ - Gaussian fit', color='fuchsia')
    ax.axvline(x=TST_noon, ymin=0.0, ymax=1.0, alpha=.6,
               color='darkblue', linestyle='--', linewidth=1)
    ax.axvline(x=MST_noon, ymin=0.0, ymax=1.0, alpha=.6,
               color='darkgreen', linestyle='--', linewidth=1)
    TST_noon_str = fr"Noon TST: {TST_noon.strftime('%H:%M:%S')}"
    MST_noon_str = fr"Noon MST: {MST_noon.strftime('%H:%M:%S')}"
    if MST_noon < TST_noon:
        left_title = MST_noon_str
        left_c = 'darkgreen'
        left_x = TST_noon
        right_title = TST_noon_str
        right_c = 'darkblue'
        right_x = MST_noon
    else:
        left_title = TST_noon_str
        left_c = 'darkblue'
        left_x = TST_noon
        right_title = MST_noon_str
        right_c = 'darkgreen'
        right_x = TST_noon
    ax.annotate(right_title,
                fontweight='bold', fontsize=12, color=right_c,
                xy=(right_x, y.min()),
                ha="left", va="center",
                bbox=dict(boxstyle='round,pad=0.2', ec="none", fc=[1, 1, 1]),
                xytext=(right_x + timedelta(minutes=50), y.min()))
    ax.annotate(left_title,
                fontweight='bold', fontsize=12, color=left_c,
                xy=(left_x, y.min()),
                ha="right", va="center",
                bbox=dict(boxstyle='round,pad=0.2', ec="none", fc=[1, 1, 1]),
                xytext=(left_x - timedelta(minutes=50), y.min()))
    ax.xaxis.set_major_formatter(vis_utils.TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join('figures', f"theta_sun_{cur_day.strftime('%Y-%m-%d')}")
    print(f"Saving fig to {fig_path}")
    plt.savefig(fig_path + '.jpeg')
    plt.savefig(fig_path + '.svg')
    plt.title(f"Sun elevation during {cur_day.strftime('%Y-%m-%d')}")
    plt.show()


def plot_daily_bg_signal(bgmean, high_curves, low_curves, mean_curves, bins_per_day):
    wavelengths = gs.LAMBDA_nm().get_elastic()

    bg_max = bgmean.copy(deep=True).assign_attrs({'info': 'Max Background Signal'})
    bg_max.data = np.array(high_curves).reshape((3, bins_per_day))
    bg_min = bgmean.copy(deep=True).assign_attrs({'info': 'Min Background Signal'})
    bg_min.data = np.array(low_curves).reshape((3, bins_per_day))

    # Plot background signal
    sns.set_palette(sns.color_palette(vis_utils.COLORS))
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for i, (curve_h, curve_l, mean_val, c, chan, wavelength) in enumerate(
            zip(high_curves, low_curves, mean_curves, vis_utils.COLORS, ['UV', 'G', 'IR'], wavelengths)):
        ax.fill_between(bgmean.Time, curve_h, curve_l, alpha=.3, color=c)
    bg_min.plot(hue='Wavelength', ax=ax, linewidth=0.8, linestyle='--')
    bg_max.plot(hue='Wavelength', ax=ax, linewidth=0.8, linestyle='--')
    SHOW_MEAN = False
    if SHOW_MEAN:
        bgmean.plot(hue='Wavelength', ax=ax, linewidth=0.8)
    ax.xaxis.set_major_formatter(vis_utils.TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    ax.set_xlim([bgmean.Time.values[0], bgmean.Time.values[-1]])
    ax.set_ybound([-.01, 2])
    plt.tight_layout()
    plt.show()


def plot_bg_part_of_year(ds_bg_year, dslice):
    sns.set_style('white')
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16,6))
    ds_bg_year.sel(Time=dslice).bg.plot(hue='Wavelength', ax=ax, linewidth=0.05)
    ax.set_xlim([dslice.start, dslice.stop])
    ax.set_ybound([-.01, 2])
    ax.set_ylabel(r"${\rm P_{BG}[photons]}$")
    ax.set_xticks(pd.date_range(dslice.start, dslice.stop, freq="2MS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_tick_params(rotation=0)
    # leg = ax.legend()
    # for line in leg.get_lines():
    #     line.set_linewidth(3.0)

    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    plt.tight_layout()

    fig_path = os.path.join('figures', f"BG_{dslice.start.strftime('%Y')}")
    print(f"Saving fig to {fig_path}")
    plt.savefig(fig_path + '.jpeg')
    plt.savefig(fig_path + '.svg')
    ax.set_title(f"Background signal: {dslice.start.strftime('%d/%m/%Y')}--{dslice.stop.strftime('%d/%m/%Y')}")
    plt.show()


def plot_irradiance_vs_sun_elevation_at_noon_times(ds_year):
    fig, ax = plt.subplots()
    ds_year.irradiance.plot(ax=ax)
    ax1 = ax.twinx()
    ds_year.sunelevation.plot(ax=ax1, c='m')
    ax.set_title('Yearly - irradiance vs sun elevation at noon times')
    plt.tight_layout()
    plt.show()

import xarray as xr
def plot_bg_one_day(ds_bg_year, c_day, mean=None):
    dslice = slice(c_day, c_day + timedelta(days=1) - timedelta(seconds=30))
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ds = ds_bg_year.sel(Time=dslice).bg

    ds.plot(hue='Wavelength', ax=ax, linewidth=0.3)
    if mean is not None:
        aligned_mean = xr.zeros_like(ds)
        aligned_mean.values = mean.values
        aligned_mean.plot(hue='Wavelength', ax=ax, linewidth=2)
    ax.set_xlim([dslice.start, dslice.stop])
    ax.xaxis.set_major_formatter(vis_utils.TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    ax.set_ybound([-.01, 2])
    ax.set_ylabel(r"${\rm P_{BG}[photons]}$")
    plt.tight_layout()
    fig_path = os.path.join('figures', f"BG_{c_day.strftime('%Y-%m-%d')}")
    print(f"Saving fig to {fig_path}")
    plt.savefig(fig_path + '.jpeg')
    plt.savefig(fig_path + '.svg')
    ax.set_title(f"{ds_bg_year.bg.info} - {c_day.strftime('%d/%m/%Y')}")
    plt.show()


