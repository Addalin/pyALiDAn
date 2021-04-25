import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
import seaborn as sns

import calendar
import matplotlib.dates as mdates

import learning_lidar.utils.global_settings as gs
from learning_lidar.generation.daily_signals_generations_utils import explore_orig_day
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.generation.daily_signals_generations_utils import calc_total_optical_density, \
    calc_lidar_signal, calc_daily_measurement

eps = np.finfo(np.float).eps
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
TIMEFORMAT = mdates.DateFormatter('%H:%M')
colors = ["darkblue", "darkgreen", "darkred"]
sns.set_palette(sns.color_palette(colors))
customPalette = sns.set_palette(sns.color_palette(colors))
PLOT_RESULTS = False


def generate_daily_lidar_measurment(station, day_date, SAVE_DS=True):
    total_ds = calc_total_optical_density(station=station, day_date=day_date)
    signal_ds = calc_lidar_signal(station=station, day_date=day_date, total_ds=total_ds)
    measure_ds = calc_daily_measurement(station=station, day_date=day_date, signal_ds=signal_ds)
    # TODO: merge signal_ds and measure_ds to lidar_ds
    lidar_gen_ds = xr.Dataset().assign_attrs(signal=signal_ds, mesure=measure_ds)
    if SAVE_DS:
        gen_utils.save_generated_dataset(station, measure_ds, data_source='lidar',
                                         save_mode='sep')  # TODO: add profile name for seperates as 'range_corr'
        gen_utils.save_generated_dataset(station, lidar_gen_ds, data_source='lidar', save_mode='single')

    return


if __name__ == '__main__':
    """
     1. Set parameters
        1. Station & general params
        2. Set Height index
        3. Choose a day & Set Time index
"""
    # %% 1. Station & general params
    main_folder = r'C:\Users\addalin\Dropbox\Lidar\lidar_learning'
    station_name = 'haifa'
    station = gs.Station(station_name=station_name)
    wavelengths = gs.LAMBDA_nm().get_elastic()  # [355,532,1064]

    # %% 2. Set Height index
    heights = station.get_height_bins_values()

    # %% 3. Choose a day & Set Time index

    month = 9
    year = 2017
    last = (calendar.monthrange(year, month)[1])
    month_start_day = datetime(year, month, 1)
    month_end_day = datetime(year, month, last)

    day_number = 2
    cur_day = month_start_day + timedelta(days=day_number)
    day_str = cur_day.strftime('%Y-%m-%d')

    # %%
    DISP_FIG = False

    start_date = datetime(2017, 9, 1)
    end_date = datetime(2017, 10, 31)
    ds_total = calc_total_optical_density(station=station, day_date=cur_day)
    signal_ds = calc_lidar_signal(station, cur_day, ds_total)
    measure_ds = calc_daily_measurement(station, cur_day, signal_ds)
    # TODO: Continue from here...
    """
    # %% 6. Save lidar and range corrected signal , and measurements
        1. Creating Daily Lidar Signals dataset
        2. Saving dataset of range_corr_p per wavelength, and lidar_gen_ds
    """

    # %%1. Creating Daily Lidar Signals dataset

    lidar_gen_ds = xr.Dataset()
    lidar_gen_ds = lidar_gen_ds.assign(range_corr=pr2_ds,
                                       range_corr_p=pr2n_ds,
                                       lidar_sig=p_ds,
                                       lidar_sig_p=pn_ds)
    lidar_gen_ds['date'] = cur_day
    lidar_gen_ds.attrs = {'location': station.location,
                          'info': 'Daily generated lidar signals',
                          'source_file': os.path.basename(__file__)}
    # %% 2. Saving dataset of range_corr_p per wavelength, and lidar_gen_ds
    # save_gen_lidar_dataset(station, lidar_gen_ds, save_mode='both')
    gen_utils.save_generated_dataset(station, lidar_gen_ds, data_source='lidar', save_mode='both')

    """
    ### 6. Exploring 1D generated profiles of:
        - $p$
        - $p\cdot r^2$
        - $p_{\rm Poiss}$
        - $p_{\rm Poiss}\cdot r^2$
        > TODO: This section in Analysis notebook

    """

    EXPLORE_GEN_DAY = False
    if EXPLORE_GEN_DAY:
        # 1D generated signals exploration
        DISP_FIG = True
        if DISP_FIG:
            Times = cur_day + np.array([timedelta(hours=5), timedelta(hours=10), timedelta(hours=15)])
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
            for t, ax in zip(Times, axes.ravel()):
                pr2_ds.where(p_mean < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linestyle='--',
                                                                                linewidth=0.8)
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
            for t, ax in zip(Times, axes.ravel()):
                pn_ds.where(p_mean < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linewidth=0.5,
                                                                               alpha=0.6)
            plt.tight_layout()
            plt.show()

            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
            for t, ax in zip(Times, axes.ravel()):
                pr2n_ds.where(p_mean < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linewidth=0.5,
                                                                                 alpha=0.6)
            plt.tight_layout()
            plt.show()

    """
    # 7. Exploring up-sampled and down-sampled data to check the "stairs" effect of multiplying / dividing by $r^2$
        1. Testing: $p = \frac {LC_{\rm generated}\cdot\beta_{\rm attbsc}}{r^2}$
         - Upsampling / downsampling ${LC_{\rm generated}\cdot\beta_{\rm attbsc}}$
         - Calculating $p = \frac {LC_{\rm generated}\cdot\beta_{\rm attbsc}}{\hat r^2}$
         - Where:
            - for up-sample case: $\hat r = \frac{1}{2}\cdot r$
            - for down-sample case: $\hat r = 2\cdot r$
        > TODO: This section in Analysis notebook
    
    """
    if EXPLORE_GEN_DAY:
        # plot 3D surf figure
        X = np.arange(0, station.total_time_bins, 1)
        th = 12.0
        Y = pr2_ds.Height.sel(Height=slice(0, th)).values
        X, Y = np.meshgrid(X, Y)
        Z = p_ds.sel(Wavelength=wavelength, Height=slice(0, th)).values
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='turbo',
                               linewidth=0.1, rstride=5, alpha=.5,
                               cstride=5, antialiased=False)
        ax.set_ylabel('Height')
        ax.set_xlabel('Time')
        ax.set_zlabel('Density')
        # ax.set_ylim([0,5.0])
        fig.colorbar(surf, shrink=0.8, aspect=5)
        ax.view_init(30, 10)
        plt.show()

    if EXPLORE_GEN_DAY:
        # up-sampled signal exploration
        start_h = heights[0]
        end_h = heights[-1]
        scale = 0.25
        new_height_size = int(station.n_bins / scale)
        new_heights = np.linspace(start=start_h, stop=end_h, num=new_height_size, endpoint=True)

        temp = xr.DataArray(np.zeros((new_height_size, station.total_time_bins, len(wavelengths))),
                            [("Height", new_heights), ("Time", pr2_ds.Time.values),
                             ("Wavelength", pr2_ds.Wavelength.values)])
        up_pr2d = pr2_ds.interp_like(temp, method='linear')

        # 2D plot of up-sampled range corrected
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            up_pr2d.sel(Wavelength=wavelength).plot(cmap='turbo', ax=ax)
        plt.tight_layout()
        plt.show()

    if EXPLORE_GEN_DAY:
        # plot functions exploration
        new_rr_im = np.tile(temp.Height.values.reshape(new_height_size, 1),
                            (pr2_ds.Wavelength.size, 1, pr2_ds.Time.size)) ** 2
        da_rr = xr.DataArray(dims=('Wavelength', 'Height', 'Time'), data=new_rr_im)
        up_pd = (up_pr2d / da_rr).astype(int)

        # %%1D plot of up-sampled signal
        Times = cur_day + np.array([timedelta(hours=5), timedelta(hours=10), timedelta(hours=15)])
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
        for t, ax in zip(Times, axes.ravel()):
            p_ds.where(p_ds < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linewidth=0.8)
            up_pd.where(up_pd < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linewidth=0.8,
                                                                          linestyle='--')
        plt.tight_layout()
        plt.show()

        # %Testing changes in 2D plotting for several plot functions
        # %% 2D plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            # log_p = xr.apply_ufunc(lambda p:np.log(p),up_pd.sel(Wavelength = wavelength),keep_attrs = True)
            # log_p.plot(cmap='turbo',ax=ax)
            up_pd.where(up_pd < 50).sel(Wavelength=wavelength).plot(cmap='turbo', ax=ax)
            # xr.plot.contourf(p_ds.where(p_ds<50).sel(Wavelength = wavelength),p_ds.Time)
            # .plot.counturf(cmap='turbo',ax =ax)
        plt.tight_layout()
        plt.show()

        # %% contourf
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            xr.plot.contourf(up_pd.where(up_pd < 50).sel(Wavelength=wavelength),
                             x='Time', y='Height', cmap='turbo', ax=ax, levels=50)
        plt.tight_layout()
        plt.show()

        # %% pcolormesh
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            xr.plot.pcolormesh(up_pd.where(up_pd < 50).sel(Wavelength=wavelength),
                               x='Time', y='Height', cmap='turbo',
                               ax=ax, levels=25, rasterized=True,
                               vmin=eps, vmax=up_pd.where(up_pd < 50).sel(Wavelength=wavelength).max().values)
        plt.tight_layout()
        plt.show()

        # %%pcolormesh with different color division
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            xr.plot.pcolormesh(up_pd.where(up_pd < 50).sel(Wavelength=wavelength),
                               x='Time', y='Height', cmap='turbo',
                               ax=ax, levels=300, rasterized=True,
                               vmin=up_pd.where(up_pd < 50).sel(Wavelength=wavelength).min().values,
                               vmax=up_pd.where(up_pd < 50).sel(Wavelength=wavelength).max().values)
        plt.tight_layout()
        plt.show()

    """
    # 2. Explore measurements and parameters for the chosen day
        1. Load extended calibration database for signal exploration
        2. Create daily calibration dataset
        3. Visualise typical parameters of measured day as: $LC$, $r_m$ ,$r_0$,$r_1$
        > TODO: This section in Analysis notebook
    """
    EXPLORE_ORIG_DAY = False
    if EXPLORE_ORIG_DAY:
        time_indx = station.calc_daily_time_index(cur_day)
        explore_orig_day(main_folder=main_folder, station_name=station_name,
                         start_date=start_date, end_date=end_date,
                         cur_day=cur_day, timedelta=timedelta, wavelengths=wavelengths, time_indx=time_indx)
