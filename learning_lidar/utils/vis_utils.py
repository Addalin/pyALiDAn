import random
from datetime import timedelta

import pandas as pd
import seaborn as sns
from matplotlib import dates as mdates, pyplot as plt

import learning_lidar.utils.utils as utils

FIGURE_DPI = 300
SAVEFIG_DPI = 300
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 16
BIG_FONT_SIZE = 18
TITLE_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 20
TIMEFORMAT = mdates.DateFormatter('%H:%M')
MONTHFORMAT = mdates.DateFormatter('%Y-%m')
DAYFORMAT = mdates.DateFormatter('%Y-%m-%d')
COLORS = ["darkblue", "darkgreen", "darkred"]


def set_visualization_settings():
    # TODO make sure this actually propagates to other functions
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['savefig.dpi'] = SAVEFIG_DPI

    # Create an array with the colors to use

    # Set a custom color palette
    sns.set_palette(sns.color_palette(COLORS))

    plt.rc('font', size=SMALL_FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=TITLE_FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SUPTITLE_FONT_SIZE)  # fontsize of the figure title


def plot_daily_profile(profile_ds, height_slice=None, figsize=(16, 6), save_fig=False):
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
    suptitle = f"{profile_ds.info} - {str_date}"
    plt.suptitle(suptitle)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    if save_fig:
        clean_title = ''.join(char for char in suptitle if char.isalnum())
        plt.savefig(f"{clean_title}.jpeg")
        print(f"saved fig {clean_title}")
    plt.show()


def plot_hourly_profile(profile_ds, height_slice=None, figsize=(10, 6), times=None):
    # TODO: add scientific ticks on color-bar
    day_date = utils.dt64_2_datetime(profile_ds.Time[0].values)
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

