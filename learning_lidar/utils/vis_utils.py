import logging
import os
from datetime import datetime
from datetime import timedelta
from decimal import Decimal

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import dates as mdates, pyplot as plt

import learning_lidar.utils.utils as utils
from learning_lidar.utils import xr_utils
from learning_lidar.utils.global_settings import PKG_ROOT_DIR

FIGURE_DPI = 300
SAVEFIG_DPI = 300
SMALL_FONT_SIZE = 16 + 2
MEDIUM_FONT_SIZE = 18 + 2
BIG_FONT_SIZE = 20 + 2
TITLE_FONT_SIZE = 20 + 2
SUPTITLE_FONT_SIZE = 22 + 2
TIMEFORMAT = mdates.DateFormatter(r'%H')
MONTHFORMAT = mdates.DateFormatter(r'%Y-%m')
DAYFORMAT = mdates.DateFormatter('%Y-%m-%d')
COLORS = ["darkblue", "darkgreen", "darkred"]


def stitle2figname(stitle: str, format_fig='png'):
    # Replace chars that are not acceptable to file names
    title_str = stitle.replace("\n", "").replace("  ", " ").replace("__", "_"). \
        replace("\\", "_").replace("/", "_").replace(":", '-'). \
        replace('.', '_').replace('$', '').replace("{", "").replace("}", "").replace("|", "")
    fig_name = f"{title_str}.{format_fig}"
    return fig_name


def set_visualization_settings():
    # TODO make sure this actually propagates to other functions
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['savefig.dpi'] = SAVEFIG_DPI
    plt.rcParams['figure.constrained_layout.use'] = True

    # Create an array with the colors to use

    # Set a custom color palette
    sns.set_palette(sns.color_palette(COLORS))

    plt.rc('font', size=SMALL_FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=TITLE_FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('axes', labelweight='bold')  # weight of the x and y labels
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SUPTITLE_FONT_SIZE)  # fontsize of the figure title

    plt.rc('text', usetex=True)
    plt.rc('font', weight='bold')
    plt.rcParams['text.latex.preamble'] = r'\boldmath'


def plot_daily_profile(profile_ds, height_slice=None, figsize=(16, 6), SAVE_FIG=False):
    # TODO: add scientific ticks on color-bar
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
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    if SAVE_FIG:
        fig_path = os.path.join('figures', suptitle)
        print(f"Saving fig to {fig_path}")
        plt.savefig(fig_path + '.jpeg')
        plt.savefig(fig_path + '.svg')
    plt.suptitle(suptitle)
    plt.show()


def plot_hourly_profile(profile_ds, height_slice=None, figsize=(10, 6), times=None,
                        SAVE_FIG=False,
                        folder_name: os.path = None, format_fig: str = 'png'):
    # TODO: add scientific ticks on color-bar
    day_date = utils.dt64_2_datetime(profile_ds.Time[0].values)
    str_date = day_date.strftime("%Y-%m-%d")
    if times is None:
        times = [day_date + timedelta(hours=8),
                 day_date + timedelta(hours=12),
                 day_date + timedelta(hours=18)]
    if height_slice is None:
        height_slice = slice(profile_ds.Height[0].values, profile_ds.Height[-1].values)

    ncols = len(times)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
    if ncols > 1:
        for t, ax in zip(times, axes.ravel()):
            profile_ds.sel(Time=t, Height=height_slice).plot.line(ax=ax, y='Height', hue='Wavelength')
            ax.set_title(pd.to_datetime(str(t)).strftime('%H:%M:%S'))
    else:
        profile_ds.sel(Time=times[0], Height=height_slice).plot.line(ax=axes, y='Height', hue='Wavelength')
        axes.set_title(pd.to_datetime(str(times[0])).strftime('%H:%M:%S'))
    plt.tight_layout()
    plt.suptitle(f"{profile_ds.info} - {str_date}")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    plt.show()
    if SAVE_FIG:
        fig_path = save_fig(fig, folder_name=folder_name, format_fig=format_fig)
    return fig_path


def visualize_ds_profile_chan(dataset, lambda_nm=532, profile_type='range_corr',
                              USE_RANGE=None, minv=None, maxv=None, cmap='turbo',
                              figsize=(10, 6), SAVE_FIG=False,
                              dst_folder=os.path.join(PKG_ROOT_DIR, 'figures'),
                              format_fig='png', dpi=1000):
    logger = logging.getLogger()
    date_datetime = xr_utils.get_daily_ds_date(dataset)
    sub_ds = dataset.sel(Wavelength=lambda_nm).get(profile_type)

    # Currently, only a dataset with range_corrected variable, has min/max plot_range values
    if minv is None or maxv is None:
        USE_RANGE = None if (profile_type != 'range_corr') else USE_RANGE
        if USE_RANGE == 'MID':
            try:
                [maxv, minv] = [
                    dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_max_range').values.tolist(),
                    dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_min_range').values.tolist()]
            except:
                logger.debug("\nThe dataset doesn't 'contain plot_min_range', setting min=0 and maxv=max/2")
                minv = np.nanmin(sub_ds.values)
                maxv = 0.5 * (np.nanmax(sub_ds.values) - minv)
        elif USE_RANGE == 'LOW':
            try:
                maxv = dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_min_range').values.tolist()
            except:
                logger.debug("\nThe dataset doesn't 'contain plot_min_range', setting maxv=0")
                maxv = 0
            minv = np.nanmin(sub_ds.values)
        elif USE_RANGE == 'HIGH':
            try:
                maxv = dataset.sel(Wavelength=lambda_nm, drop=True).get('plot_max_range').values.tolist()
            except:
                logger.debug("\nThe dataset doesn't 'contain plot_min_range', setting maxv=0")
                maxv = np.nanmin(sub_ds.values)
            maxv = np.nanmax(sub_ds.values)
        elif USE_RANGE is None:
            [maxv, minv] = [np.nanmax(sub_ds.values), np.nanmin(sub_ds.values)]

    dims = sub_ds.dims
    if 'Time' not in dims:
        logger.error(f"\nThe dataset should have a 'Time' dimension.")
        return None

    EXTEND_MIN = False if minv == np.nanmin(sub_ds.values) else True
    EXTEND_MAX = False if maxv == np.nanmin(sub_ds.values) else True
    if EXTEND_MIN and EXTEND_MAX:
        extend = 'both'
    elif not EXTEND_MIN and not EXTEND_MAX:
        extend = 'neither'
    elif EXTEND_MIN and not EXTEND_MAX:
        extend = 'min'
    else:
        extend = 'max'

    if 'Height' in dims:  # plot x- time, y- height
        g = sub_ds.where(sub_ds < maxv).where(sub_ds > minv).plot(x='Time', y='Height',
                                                                  cmap=cmap, extend=extend,
                                                                  figsize=figsize)  # ,robust=True)
    elif len(dims) == 2:  # plot x- time, y- other dimension
        g = sub_ds.where(sub_ds < maxv).where(sub_ds > minv).plot(x='Time',
                                                                  cmap=cmap, extend=extend,
                                                                  figsize=figsize)
    elif len(dims) == 1:  # plot x- time, y - values in profile type
        g = sub_ds.plot(x='Time', figsize=figsize)[0]

    # TODO: add 'extend' to colorbar shape in case plotting in middle/lower/higher ranges

    # Set time on x-axis
    g.axes.xaxis.set_major_formatter(TIMEFORMAT)
    g.axes.xaxis_date()
    g.axes.get_xaxis().set_major_locator(mdates.HourLocator(interval=4))

    plt.setp(g.axes.get_xticklabels(), rotation=0, horizontalalignment='center')

    # Set title description
    date_str = date_datetime.strftime('%d/%m/%Y')
    stitle = f"{sub_ds.attrs['info']} - {lambda_nm}nm \n {dataset.attrs['location']} {date_str}"
    plt.title(stitle, y=1.05)
    plt.tight_layout()
    plt.show()

    if SAVE_FIG:
        fpath = save_fig(fig=g.figure, fig_name=stitle, folder_name=dst_folder, format_fig=format_fig, dpi=dpi)
    else:
        fpath = None

    return g, fpath


def daily_ds_histogram(dataset, profile_type: str = 'range_corr', alpha: float = .5,
                       SAVE_FIG: bool = False, n_temporal_splits: int = 1,
                       nbins: int = 100, figsize=(5, 4), height_range: slice = None,
                       dst_folder: os.path = os.path.join(PKG_ROOT_DIR, 'figures'),
                       format_fig='png', dpi=1000,
                       y_scale: str = 'linear',
                       x_scale: str = 'symlog',
                       show_title: bool = True, vis_stats_table: bool = True):
    # TODO: replace function to work with seaborn histplot :https://seaborn.pydata.org/generated/seaborn.histplot.html
    # Note: if the log plot of x shows only one or two bins, it's better using x_scale='linear'
    logger = logging.getLogger()

    # Extract and set the relevant times and heights to analyse
    date_datetime = xr_utils.get_daily_ds_date(dataset)
    # Check validity of height_range
    if not (type(height_range) is slice
            or (type(height_range) is slice and height_range.start <= height_range.stop)):
        height_range = slice(dataset.Height.min().item(), dataset.Height.max().item())
    dataset = dataset.sel(Height=height_range)
    time_splits = np.array_split(dataset.Time, n_temporal_splits)
    # Adapt fig size according to the number of n_splits
    (w_fig, h_fig) = figsize

    logger.info(f"Start calculating daily histogram of {date_datetime} "
                f"for height range [{height_range}], "
                f"and {n_temporal_splits} time splits")

    # Set automatically the number of column in the plotting facet grid (ncols=2 if there are negative values)
    if (dataset.get(profile_type) < 0).any().values:
        ncols = 2
    else:
        ncols = 1
    ax_pos = ncols - 1
    ax_neg = ax_pos - 1
    new_figsize = (w_fig * n_temporal_splits, h_fig * n_temporal_splits)
    fig, axes = plt.subplots(nrows=n_temporal_splits, ncols=ncols, figsize=new_figsize,
                             sharey='row', sharex='col', squeeze=False)
    ax = axes
    th = 0
    wavelengths = dataset.Wavelength.values.tolist()

    list_ds_stats = []
    for ind_split, time_split in enumerate(time_splits):
        df_stats = pd.DataFrame(columns=['wavelength [nm]', 'Zeros %', 'Positives %', 'Negatives %'])
        for ind, (wavelength) in enumerate(wavelengths):
            ds_profile = dataset.sel(Time=time_split).get(profile_type)
            sub_ds = ds_profile.sel(Wavelength=wavelength)
            orig_size = sub_ds.where(sub_ds != np.nan).values.size

            # Calc positive values histogram
            pos_vals = sub_ds.where(sub_ds > th).where(sub_ds != np.nan).values
            pos_vals = pos_vals[~np.isnan(pos_vals)]
            pos_size = pos_vals.size

            # Calc negative values histogram
            neg_ds = sub_ds.where(sub_ds < -th).where(sub_ds != np.nan)
            neg_vals = neg_ds.values[~np.isnan(neg_ds.values)]
            neg_size = neg_vals.size

            # Update stats dataframe
            df_stats.loc[ind] = [wavelength, f"{100.0 * (orig_size - neg_size - pos_size) / orig_size:.2f}",
                                 f"{100.0 * pos_size / orig_size:.2f}",
                                 f"{100.0 * neg_size / orig_size :.2f}"]

            # Set plt.histogram dictionary
            kwargs = dict(histtype='stepfilled', alpha=alpha,  # density=True,
                          ec=COLORS[ind], linewidth=.8,
                          label=f"$\lambda={wavelength}$")

            # Plot positive histogram
            nbins_p = nbins  # if pos_size>100 else 10
            hist, bins = np.histogram(pos_vals, bins=nbins_p)
            if x_scale in ['symlog', 'log']:
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                bins = logbins
                ax[ind_split, ax_pos].set_xscale(x_scale)
            ax[ind_split, ax_pos].hist(x=pos_vals, bins=bins, **kwargs)

            # plot negative histogram
            if neg_size > 0:
                nbins_n = nbins if neg_size > 100 else 1
                histneg, binsneg = np.histogram(neg_vals, bins=nbins_n)
                if x_scale in ['symlog', 'log']:
                    neg_logbins = np.logspace(np.log10(-binsneg[-1]), np.log10(-binsneg[0]), len(binsneg))
                    binsneg = neg_logbins
                ax[ind_split, ax_neg].hist(x=-neg_vals, bins=binsneg, **kwargs)

        # Fix xticks with FixedLocator but also using MaxNLocator to avoid cramped x-labels
        if x_scale != 'linear':
            ticks_loc = ax[ind_split, ax_pos].get_xticks()
            ax[ind_split, ax_pos].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

            new_posx_ticks = [rf"$10^{{{np.log10(Decimal(n))}}}$" if n > 0 else round(n) for n in ticks_loc]
            ax[ind_split, ax_pos].set_xticklabels(new_posx_ticks)
            if neg_size > 0:
                ax[ind_split, ax_neg].set_xscale('log')
                if (ind_split + 1) % 2:
                    ax[ind_split, ax_neg].invert_xaxis()
                ax[ind_split, ax_neg].set_xscale('symlog')
                ticks_loc = ax[ind_split, ax_neg].get_xticks()
                ax[ind_split, ax_neg].xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                new_negx_vals = [np.log10(Decimal(n)) if n > 0 else round(n) for n in ticks_loc]
                # print(new_negx_vals)
                new_negx_ticks = [rf"$-10^{{{tic:.1f}}}$" if type(tic) == Decimal else tic for tic in new_negx_vals]
                ax[ind_split, ax_neg].set_xticklabels(new_negx_ticks)

        # Set common y values and label to both axes
        plt.yscale(y_scale)
        if ax_neg == 0:
            y_lim = ax[ind_split, ax_pos].get_ylim()
            ax[ind_split, ax_neg].set_ylim(y_lim)
        if y_scale == 'linear':
            ax[ind_split, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[ind_split, 0].tick_params(axis='both', which='major')
        ax[ind_split, 0].set_ylabel('counts')
        if ax_neg == 0:
            ax[ind_split, ax_pos].tick_params(axis='both', which='major')
            ax[ind_split, ax_pos].grid(axis='both', which='major', linestyle='--', alpha=alpha)
        ax[ind_split, 0].grid(axis='both', which='major', linestyle='--', alpha=alpha)

        if ind_split == n_temporal_splits - 1:
            xlabels = f"{ds_profile.long_name}\n[{ds_profile.units}]"
            if ax_neg == 0:
                ax[ind_split, 0].set_xlabel(xlabels, position=(1.05, 1e6),
                                            horizontalalignment='center')
            else:
                ax[ind_split, 0].set_xlabel(xlabels)

        if ind_split == 0:  # Show legend only once (for all times)
            ax[ind_split, ax_pos].legend(loc='upper right')

        # Logging histogram results
        df_stats = df_stats.set_index('wavelength [nm]')
        logger.info(f"Done calculating daily histogram")
        logger.info(df_stats)

        # Plot stats table on figure for each time split
        if vis_stats_table:
            stats_table = ax[ind_split, ax_pos].table(cellText=df_stats.values,
                                                      colWidths=[0.09] * 3,
                                                      rowLabels=df_stats.index.tolist(),
                                                      colLabels=df_stats.columns.tolist(),
                                                      cellLoc='center',
                                                      loc='upper left')
            stats_table.scale(1.0, 1.2)
            # stats_table.set_fontsize(14)

        # Create xr.DataSet of statistics containing stats table for each time split
        ds_stats = xr.Dataset(
            data_vars={'stats': (('Wavelength', 'Stats'), df_stats.values),
                       'index': ind_split,
                       'start_time': time_split[0].values,
                       'end_time': time_split[-1].values
                       },
            coords={'Stats': ['zero', 'positive', 'negative'],
                    'Wavelength': df_stats.index.to_list(),
                    })
        list_ds_stats.append(ds_stats)

    stitle = f"Histogram of {ds_profile.info.lower()} " \
             f"\n {dataset.attrs['location']} {date_datetime.strftime('%Y-%m-%d')} " \
             f" for {dataset.Height[0].values:.2f} - {dataset.Height[-1].values:.2f} km"
    if show_title:
        fig.suptitle(stitle)
    plt.tight_layout()
    plt.show()

    if SAVE_FIG:
        fpath = save_fig(fig=fig, fig_name=stitle, folder_name=dst_folder, format_fig=format_fig, dpi=dpi)
    else:
        fpath = None

    return fig, axes, list_ds_stats, fpath


def save_fig(fig, fig_name: str = '',
             folder_name: os.path = os.path.join(PKG_ROOT_DIR, 'figures'),
             format_fig='png', dpi: int = SAVEFIG_DPI):
    logger = logging.getLogger()
    if fig_name == '':
        try:
            stitle = fig._suptitle.get_text()
        except AttributeError as e:
            # There is no sup title to the figure, try to get ax title
            try:
                stitle = fig.axes[0].get_title()
            except:
                # There is no ax title to the figure, set it empty
                stitle = ''
            else:
                logger.debug('Setting name by ax title')
        else:
            logger.debug('Setting name by figure suptitle')
        # if stitle is empty, set a generic fig name by time of creation
        fig_name = stitle if stitle != '' else f"figure_{datetime.now().strftime('%Y_%m_%d %H_%M')}"

    fname = stitle2figname(fig_name, format_fig)
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name, exist_ok=True)
            logger.debug(f"Creating folder: {folder_name}")
        except Exception:
            raise OSError(f"Failed to create folder: {folder_name}")
    fpath = os.path.join(folder_name, fname)
    fig.savefig(fpath, bbox_inches='tight', format=format_fig.replace('.', ''), dpi=dpi)
    logger.debug(f"Figure saved at {fpath}")
    return fpath
