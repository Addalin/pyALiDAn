import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import seaborn as sns
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat

# Local modules
import learning_lidar.utils.global_settings as gs
from learning_lidar.utils.global_settings import TIMEFORMAT
from learning_lidar.generation.daily_signals_generations_utils import explore_orig_day
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.generation.daily_signals_generations_utils import calc_total_optical_density, \
    calc_lidar_signal, calc_daily_measurement,custom_plot_xr
import learning_lidar.generation.daily_signals_generations_utils as gen_sig_utils
from learning_lidar.utils.utils import create_and_configer_logger
import pandas as pd
from learning_lidar.utils.global_settings import eps


def generate_daily_lidar_measurement(station, day_date, SAVE_DS=True):
    ds_total = calc_total_optical_density(station=station, day_date=day_date)
    signal_ds = calc_lidar_signal(station, day_date, ds_total)
    measure_ds = calc_daily_measurement(station, day_date, signal_ds)

    if SAVE_DS:
        gen_utils.save_generated_dataset(station, measure_ds, data_source='lidar', save_mode='both',
                                         profiles=['range_corr'])
        gen_utils.save_generated_dataset(station, measure_ds, data_source='bg', save_mode='sep', profiles=['p_bg'])
        gen_utils.save_generated_dataset(station, signal_ds, data_source='signal', save_mode='single')

    return measure_ds, signal_ds


if __name__ == '__main__':
    gs.set_visualization_settings()
    gen_sig_utils.PLOT_RESULTS = False
    logging.getLogger('PIL').setLevel(logging.ERROR)                # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)         # Fix annoying matplotlib logs
    logger = create_and_configer_logger('generate_density.log', level=logging.DEBUG)
    station = gs.Station(station_name='haifa')
    start_date = datetime(2017, 10, 1)
    end_date = datetime(2017, 10, 31)
    days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    num_days = len(days_list)
    num_processes = min((cpu_count() - 1, num_days))
    with Pool(num_processes) as p:
        p.starmap(generate_daily_lidar_measurement, zip(repeat(station), days_list))

    RUN_NEXT = False
    if RUN_NEXT:
        for cur_day in days_list:
            measure_ds, signal_ds = generate_daily_lidar_measurement(station, cur_day)

        # TODO:
        #  1. create test_daily_signal_generation.ipynb  under 'Analysis' folder
        #  2. call to : measure_ds, signal_ds = generate_daily_lidar_measurement(station, cur_day)
        #  3. move the EXPLORE_GEN_DAY (part below) to this file -> (test_daily_signal_generation.ipynb)
        #  4. Following that for comparison add explore_orig_day for cur_day (The part of EXPLORE_ORIG_DAY)
        #  5. Delete: daily_signals_generation.ipynb

        """
        ### 6. Exploring 1D generated profiles of:
            - $p$
            - $p\cdot r^2$
            - $p_{\rm Poiss}$
            - $p_{\rm Poiss}\cdot r^2$
            > TODO: This section in Analysis notebook
    
        """

        EXPLORE_GEN_DAY = False
        DISP_FIG = False
        if EXPLORE_GEN_DAY:
            # 1D generated signals exploration
            DISP_FIG = True
            if DISP_FIG:
                Times = cur_day + np.array([timedelta(hours=5), timedelta(hours=10), timedelta(hours=15)])
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
                for t, ax in zip(Times, axes.ravel()):
                    measure_ds.range_corr.where(measure_ds.p_mean < 50). \
                        sel(Time=t, Height=slice(0, 10)).drop('date').plot(ax=ax, hue='Wavelength', linestyle='--', linewidth=0.8)
                plt.tight_layout()
                plt.show()

                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
                for t, ax in zip(Times, axes.ravel()):
                    measure_ds.p.where(measure_ds.p_mean < 50).sel(Time=t, Height=slice(0, 10)). \
                        plot(ax=ax, hue='Wavelength', linewidth=0.5, alpha=0.6)
                plt.tight_layout()
                plt.show()

                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
                for t, ax in zip(Times, axes.ravel()):
                    measure_ds.range_corr.where(measure_ds.p_mean < 50).sel(Time=t, Height=slice(0, 10)). \
                        plot(ax=ax, hue='Wavelength', linewidth=0.5, alpha=0.6)
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
            # TODO: create a surf function to a given profile (e.g. measure_ds.range_corr, 'range_corr' is the name of
            #  profile),and wavelength . e.g, (532) plot 3D surf figure
            # TODO: xarray didn't implemented surf plot -
            wavelength = 532
            X = np.arange(0, station.total_time_bins, 1)
            th = 12.0
            Y = measure_ds.range_corr.Height.sel(Height=slice(0, th)).values
            X, Y = np.meshgrid(X, Y)
            Z = measure_ds.range_corr.sel(Wavelength=wavelength, Height=slice(0, th)).values
            ax = plt.axes(projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='turbo',
                                   linewidth=0.1, rstride=5, alpha=.5,
                                   cstride=5, antialiased=False)
            ax.set_ylabel('Height')
            ax.set_xlabel('Time')
            ax.set_zlabel('Density')
            plt.colorbar(surf, ax=ax, shrink=0.9, aspect=5)
            ax.view_init(30, 10)
            plt.show()

        if EXPLORE_GEN_DAY:
            wavelengths = gs.LAMBDA_nm().get_elastic()
            heights = station.get_height_bins_values()
            # up-sampled signal exploration
            start_h = heights[0]
            end_h = heights[-1]
            scale = 0.25
            new_height_size = int(station.n_bins / scale)
            new_heights = np.linspace(start=start_h, stop=end_h, num=new_height_size, endpoint=True)

            temp = xr.DataArray(np.zeros((new_height_size, station.total_time_bins, len(wavelengths))),
                                [("Height", new_heights), ("Time", signal_ds.range_corr.Time.values),
                                 ("Wavelength", signal_ds.range_corr.Wavelength.values)])
            up_pr2d = signal_ds.range_corr.interp_like(temp, method='linear')

            # 2D plot of up-sampled range corrected
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
            for wavelength, ax in zip(wavelengths, axes.ravel()):
                up_pr2d.sel(Wavelength=wavelength).plot(cmap='turbo', ax=ax)
            plt.tight_layout()
            plt.show()

        if EXPLORE_GEN_DAY:
            # plot functions exploration
            new_rr_im = np.tile(temp.Height.values.reshape(new_height_size, 1),
                                (len(wavelengths), 1, signal_ds.range_corr.Time.size)) ** 2
            da_rr = xr.DataArray(dims=('Wavelength', 'Height', 'Time'), data=new_rr_im)
            up_pd = (up_pr2d / da_rr).astype(int)

            # %%1D plot of up-sampled signal
            Times = cur_day + np.array([timedelta(hours=5), timedelta(hours=10), timedelta(hours=15)])
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
            for t, ax in zip(Times, axes.ravel()):
                signal_ds.p.where(signal_ds.p < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength',
                                                                                          linewidth=0.8)
                up_pd.where(up_pd < 50).sel(Time=t, Height=slice(0, 10)).plot(ax=ax, hue='Wavelength', linewidth=0.8,
                                                                              linestyle='--')
            plt.tight_layout()
            plt.show()

            # %Testing changes in 2D plotting for several plot functions
            # %% 2D plot
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
            for wavelength, ax in zip(wavelengths, axes.ravel()):
                up_pd.where(up_pd < 50).sel(Wavelength=wavelength).plot(cmap='turbo', ax=ax)
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
            explore_orig_day(main_folder=main_folder, station_name=station.name,
                             start_date=start_date, end_date=end_date,
                             cur_day=cur_day, timedelta=timedelta, wavelengths=wavelengths, time_indx=time_indx)
