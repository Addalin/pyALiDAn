from datetime import datetime, timedelta, time

import astral
import numpy as np
from dateutil import tz
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from learning_lidar.utils import utils, vis_utils
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
    """Return values from a general log function."""
    return a * np.cos(np.deg2rad(b * x) + c) + d


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


def plot_sun_elevation_at_noon(ds_year):
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