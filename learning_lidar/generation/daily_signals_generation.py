import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pytictoc import TicToc

import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import seaborn as sns

import calendar
import matplotlib.dates as mdates

import learning_lidar.global_settings as gs
from learning_lidar.generation.daily_signals_generations_utils import explore_orig_day
from learning_lidar.generation.generation_utils import save_generated_dataset
from learning_lidar.preprocessing import preprocessing as prep
from learning_lidar.utils.miscLidar import calc_tau, generate_poisson_signal_STEP

eps = np.finfo(np.float).eps
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
timeformat = mdates.DateFormatter('%H:%M')
colors = ["darkblue", "darkgreen", "darkred"]
sns.set_palette(sns.color_palette(colors))
customPalette = sns.set_palette(sns.color_palette(colors))


def get_gen_dataset_file_name(station, day_date, wavelength='*', data_source='lidar', file_type='range_corr'):
    """
     Retrieves file pattern name of generated lidar dataset according to:
      station, date, wavelength.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the generated date
    :param wavelength: wavelength [nm] e.g., for the green channel 532 [nm] or all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'aerosol' or 'lidar'
    :param file_type: string object: e.g., 'range_corr'/'lidar_power' for separated files per wavelength (355,532, or 1064) or 'lidar'/'aerosol' for all wavelengths

    :return: dataset file name (netcdf) file of the data_type required per given day and wavelength, data_source and file_type
    """
    if wavelength == '*':
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{data_source}.nc"
    else:
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{file_type}_{wavelength}_{data_source}.nc"

    return file_name


def save_gen_lidar_dataset(station, dataset, save_mode='both'):
    """
    Save the input dataset to netcdf file
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily generated lidar signal, holding 5 data variables:
             4 daily dataset, with dimensions of : Height, Time, Wavelength.
              name of profiles: 'range_corr','range_corr_p', 'lidar_sig','lidar_sig_p'
             1 shared variable: date
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' -saving both options
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    date_datetime = prep.get_daily_ds_date(dataset)
    month_folder = prep.get_month_folder_name(station.gen_lidar_dataset, date_datetime)

    prep.get_daily_ds_date(dataset)
    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []

    # NOTE: Currently saving to separated profiles is only for `range_corr_p` - used in the learning phase.cur_day
    # if one needs other separated profile, add it as an an input term.
    profile = list(dataset.data_vars)[1]
    if save_mode in ['both', 'sep']:
        for wavelength in dataset.Wavelength.values:
            ds_profile = dataset.sel(Wavelength=wavelength)[profile]
            ds_profile['date'] = date_datetime
            file_name = get_gen_dataset_file_name(station, date_datetime, data_source='lidar',
                                                  wavelength=wavelength, file_type=profile)
            ncpath = prep.save_dataset(ds_profile, month_folder, file_name)
            if ncpath:
                ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_gen_dataset_file_name(station, date_datetime, data_source='lidar', wavelength='*')
        ncpath = prep.save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


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
    km_scale = 1e-3
    min_height = station.altitude + station.start_bin_height
    top_height = station.altitude + station.end_bin_height
    heights = np.linspace(min_height * km_scale, top_height * km_scale, station.n_bins)
    dr = heights[1] - heights[0]
    total_height_bins = heights.size

    # %% 3. Choose a day & Set Time index

    month = 9
    year = 2017
    last = (calendar.monthrange(year, month)[1])
    month_start_day = datetime(year, month, 1)
    month_end_day = datetime(year, month, last)

    day_number = 2
    cur_day = month_start_day + timedelta(days=day_number)
    day_str = cur_day.strftime('%Y-%m-%d')

    min_t = datetime.combine(cur_day.date(), cur_day.time().min)
    max_t = cur_day + timedelta(hours=24) - timedelta(seconds=30)

    time_indx = pd.date_range(start=min_t, end=max_t, freq='30S')
    total_time_bins = time_indx.size
    # %%
    DISP_FIG = False

    """
    # 2. Explore measurements and parameters for the chosen day
        1. Load extended calibration database for signal exploration
        2. Create daily calibration dataset
        3. Visualise typical parameters of measured day as: $LC$, $r_m$ ,$r_0$,$r_1$
        > TODO: This section in Analysis notebook
    """

    start_date = datetime(2017, 9, 1)
    end_date = datetime(2017, 10, 31)

    EXPLORE_ORIG_DAY = False
    if EXPLORE_ORIG_DAY:
        explore_orig_day(main_folder=main_folder, station_name=station_name,
                         start_date=start_date, end_date=end_date,
                         min_t=min_t, cur_day=cur_day, timedelta=timedelta, wavelengths=wavelengths,
                         time_indx=time_indx)

    """
    # 3. Generate total backscatter and extinction distribution
        1. Load generated aerosol profiles
        2. Load molecular profiles
        3. Calculate total distribution: Summing aerosol and molecular profiles
            - $\beta = \beta_{aer}+\beta_{mol}$
            - $\sigma = \sigma_{aer}+\sigma_{mol}$
            > The molecular profiles were calculated from radiosondes measurements from NOA
            > TODO: Blend or "fade" the aerosol layer before adding to molecular density
    """

    # %% 1. Load generated aerosol profiles
    month_folder = prep.get_month_folder_name(station.gen_aerosol_dataset, month_start_day)
    nc_aer = f"{cur_day.strftime('%Y_%m_%d')}_{station_name.capitalize()}_aerosol.nc"
    ds_aer = prep.load_dataset(os.path.join(month_folder, nc_aer))
    sigma_aer = ds_aer.sigma
    beta_aer = ds_aer.beta


    def custom_plot_xr(data, height_slice=None, figsize=(16, 6)):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        for wavelength, ax in zip(wavelengths, axes.ravel()):
            if height_slice:
                data.sel(Height=height_slice, Wavelength=wavelength).plot(cmap='turbo', ax=ax)
            else:
                data.sel(Wavelength=wavelength).plot(cmap='turbo', ax=ax)
        plt.tight_layout()
        plt.show()


    if DISP_FIG:
        height_slice = slice(0.0, 15)
        custom_plot_xr(data=sigma_aer, height_slice=height_slice)
        custom_plot_xr(data=beta_aer, height_slice=height_slice)

    # %% 2. Load molecular profiles
    month_folder = prep.get_month_folder_name(station.molecular_dataset, month_start_day)
    nc_name = f"{cur_day.strftime('%Y_%m_%d')}_{station_name.capitalize()}_molecular.nc"
    ds_mol = prep.load_dataset(os.path.join(month_folder, nc_name))

    if DISP_FIG:
        custom_plot_xr(data=ds_mol.sigma, height_slice=height_slice)
        custom_plot_xr(data=ds_mol.beta, height_slice=height_slice)

    # %% 3. Calculate total distribution

    ds_total = xr.zeros_like(ds_mol)
    ds_total = ds_total.assign(date=ds_mol.date)
    ds_total.attrs = {'info': 'Daily generated profiles',
                      'sources_file': 'generate_density.ipynb',
                      'location': station.name}
    ds_total.lambda_nm.loc[:] = wavelengths

    ds_total.sigma.loc[:] = ds_mol.sigma + sigma_aer
    ds_total.beta.loc[:] = ds_mol.beta + beta_aer

    if DISP_FIG:
        custom_plot_xr(data=ds_total.sigma)
        custom_plot_xr(data=ds_total.beta)

        """
        3. Generate lidar signal
            1. Calculating the attenuated backscatter $\beta_{\rm attbsc} = \beta\exp(-2\tau)$
            2. Load daily generated Lidar power factor
            3. Calculating range corrected signal $p\cdot r^2 =LC_{generated}\cdot\beta\exp(-2\tau)$
                - Calculate a new Lidar signal $p$,based on LC generated p_day_gen.
                $p\cdot r^2 = {LC_{\rm generated}}\cdot\beta\exp(-2\tau) = {LC_{\rm generated}}\cdot\beta_{\rm attbsc}$
                > see  generation process at: `generate_LC_pattern.ipynb`
            4. Calculating lidar signal $p = \frac{LC_{generated}\cdot \beta_{\rm attbsc}}{r^2}$
        """

        # %% 1. Calculating the attenuated backscatter
        exp_tau_c = []
        for wavelength in wavelengths:
            exp_tau_t = []
            for t in tqdm(ds_total.Time):
                sigma_t = ds_total.sigma.sel(Time=t)
                e_tau = xr.apply_ufunc(lambda x: np.exp(-2 * calc_tau(x, heights)),
                                       sigma_t.sel(Wavelength=wavelength), keep_attrs=True)
                e_tau.name = r'$\exp(-2 \tau)$'
                exp_tau_t.append(e_tau)

            exp_tau_c.append(xr.concat(exp_tau_t, dim='Time'))
        exp_tau_d = xr.concat(exp_tau_c, dim='Wavelength')
        exp_tau_d = exp_tau_d.transpose('Wavelength', 'Height', 'Time')
        ds_total.attbsc.loc[:] = exp_tau_d.loc[:] * ds_total.beta

        for profile in ['beta', 'sigma', 'attbsc']:
            ds_total[profile].attrs['info'] = ds_total[profile].attrs['info'].replace('Molecular', 'Total')

        if DISP_FIG:
            custom_plot_xr(data=ds_total.attbsc, figsize=(16, 8))

        # %%

        # %% 2. Load daily generated Lidar power factor
        nc_power = f"generated_LC_{station.name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.nc"
        ds_gen_p = prep.load_dataset(os.path.join(station.generation_folder, nc_power))
        day_slice = slice(cur_day, cur_day + timedelta(hours=24) - timedelta(seconds=30))
        p_day_gen = ds_gen_p.p.sel(Time=day_slice)

        DISP_FIG = True
        if DISP_FIG:
            p_day_gen.plot(hue='Wavelength', linewidth=0.8)
            # ds_gen_p.sel(Time=tday).plot.scatter(y='p_new',x='Time',hue='Wavelength',s=15,hue_style ='discrete',edgecolor='w')
            plt.title(fr"B\'ezier interpolation of {ds_gen_p.p.long_name} - for {cur_day.strftime('%d/%m/%Y')}")
            plt.tight_layout()
            plt.show()

        # %% 3. Calculating daily range corrected signal
        pr2_c = []
        for wavelength in wavelengths:
            pr2_t = []
            p_t = []
            for t in tqdm(ds_total.Time, desc=f'Time loop for {wavelength}'):
                attbsc_t = ds_total.attbsc.sel(Time=t)
                LC_t = p_day_gen.sel(Time=t)
                pr2 = xr.apply_ufunc(lambda x, y: (x * y),
                                     attbsc_t.sel(Wavelength=wavelength),
                                     LC_t.sel(Wavelength=wavelength), keep_attrs=True)
                pr2.name = r'$pr2$'
                pr2_t.append(pr2)

            pr2_c.append(xr.concat(pr2_t, dim='Time'))

        pr2_ds = xr.concat(pr2_c, dim='Wavelength')
        pr2_ds = pr2_ds.transpose('Wavelength', 'Height', 'Time')
        pr2_ds.attrs = {'info': 'Generated Range Corrected Lidar Signal',
                        'long_name': r'$LC \beta \cdot \exp(-2\tau)$',
                        'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$'}

        DISP_FIG = True
        if DISP_FIG:
            custom_plot_xr(data=pr2_ds, figsize=(16, 8))

        # %% 4. Calculating daily lidar signal
        rr_im = np.tile(heights.reshape(total_height_bins, 1), (len(wavelengths), 1, total_time_bins)) ** 2

        ds_total = ds_total.assign(rr=xr.Variable(dims=('Wavelength', 'Height', 'Time'), data=rr_im))
        p_ds = (pr2_ds / ds_total.rr)
        p_ds.attrs = {'info': 'Generated Lidar Signal',
                      'long_name': r'$p$',
                      'units': r'$\rm$' + r'$photons$'}
        p_ds.name = r'$p$'

        EXPLORE_GEN_DAY = True
        if EXPLORE_GEN_DAY:
            # Check if the "stairs like" signal - caused by int casting
            p_d_int = p_ds.astype(int)
            Times = cur_day + np.array([timedelta(hours=5), timedelta(hours=10), timedelta(hours=15)])
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 5))
            for t, ax in zip(Times, axes.ravel()):
                p_d_int.where(p_d_int < 50).sel(Time=t, Height=slice(0, 10)). \
                    plot(ax=ax, hue='Wavelength', linewidth=0.8)
                p_ds.where(p_ds < 50).sel(Time=t, Height=slice(0, 10)). \
                    plot(ax=ax, hue='Wavelength', linewidth=0.8, linestyle='--')
            plt.tight_layout()
            plt.show()

        if DISP_FIG:
            custom_plot_xr(data=p_ds, height_slice=slice(0, 5), figsize=(16, 8))

            """
            5. Generate Lidar measurement
                1. Load bg mean signal per month
                >  Created daily mean background signal in: generate_bg_signals.ipynb
                2. Slice the relevant day of the monthly signal
                3. Calculate mean signal: $\mu_{p} =\mathbb{E}(p) + <p_{bg}>$
                4. Calculate signal measure: $P_{measure}\sim Poiss (\mu_{p} ) $
                    - Note:  for $\mu_{p}>50$: $Poiss (\mu_{p}) = \mu_{p} + \sqrt{\mu_{p}}\cdot  \mathcal{N}(0,1)$
                    > This is to save time and power of computations
                    >
                    > The poisson distribution calculated only for values lower then 50 - to assure we dont get negative values
                5. Calculate generated range corrected signal $p_{measure}\cdot r^2$
            """

            # %% 1. Load bg mean signal per month
            nc_name = f"generated_bg_{station.name}_{month_start_day.strftime('%Y-%m-%d')}_{month_end_day.strftime('%Y-%m-%d')}.nc"
            bg_folder = os.path.join(station.generation_folder, 'bg_dataset')
            ds_bg = prep.load_dataset(ncpath=os.path.join(bg_folder, nc_name))

            # %% 2. Slice the relevant day of the monthly signal
            day_slice = slice(cur_day, cur_day + timedelta(days=1) - timedelta(seconds=30))
            ds_total = ds_total.assign(bgmean=ds_bg.sel(Time=day_slice).bg)

            if DISP_FIG:
                fig, ax = plt.subplots(ncols=1, nrows=1)
                ds_total.bgmean.sel(Time=day_slice).plot(hue='Wavelength', ax=ax, linewidth=0.8)
                ax.set_xlim([day_slice.start, day_slice.stop])
                ax.set_title(f"{ds_bg.bg.info} - {cur_day.strftime('%d/%m/%Y')}")
                plt.xticks(rotation=0)
                ax.set_ybound([-.01, 2])
                plt.tight_layout()
                plt.show()

            # %% 3. Calculate mean signal
            p_mean = p_ds + ds_total.bgmean
            if DISP_FIG:
                # plot 2D figure
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
                for wavelength, ax in zip(wavelengths, axes.ravel()):
                    p_mean.where(p_mean < 20).sel(Wavelength=wavelength, Height=slice(0, 10)). \
                        plot(ax=ax, cmap='turbo')
                plt.tight_layout()
                plt.show()

            # %% 4. Calculate signal measure
            tic0 = TicToc()
            tic0.tic()
            pn_h = xr.apply_ufunc(
                lambda mu: (mu + (np.sqrt(mu)) * np.random.normal(loc=0, scale=1.0, size=mu.shape)).astype(int),
                p_mean.where(p_mean >= 50).fillna(0), keep_attrs=True)
            pn_l = xr.apply_ufunc(lambda mu: generate_poisson_signal_STEP(mu),
                                  p_mean.where(p_mean < 50).fillna(0), keep_attrs=True, dask='parallelized')
            tic0.toc()
            pn_ds = pn_h + pn_l
            pn_ds.attrs = {'info': 'Generated Poisson Lidar Signal',
                           'long_name': r'$p$',
                           'units': r'$\rm$' + r'$photons$'}

            if DISP_FIG:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
                for wavelength, ax in zip(wavelengths, axes.ravel()):
                    pn_ds.where(pn_ds >= 3).sel(Wavelength=wavelength,
                                                Height=slice(0, 10)) \
                        .plot(cmap='turbo', ax=ax)
                plt.tight_layout()
                plt.show()

            # %% 5. Calculate generated range corrected signal
            pr2n_ds = (pn_ds.copy(deep=True) * ds_total.rr)
            pr2n_ds.attrs = {'info': 'Generated Poisson Range Corrected Lidar Signal',
                             'long_name': r'$p$' + r'$\cdot r^2$',
                             'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$'}

            if DISP_FIG:
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
                for wavelength, ax in zip(wavelengths, axes.ravel()):
                    pr2n_ds.where(pn_ds >= 3). \
                        sel(Wavelength=wavelength,
                            Height=slice(0, 10)).plot(ax=ax, cmap='turbo', )
                plt.tight_layout()
                plt.show()

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
    #save_gen_lidar_dataset(station, lidar_gen_ds, save_mode='both')
    save_generated_dataset(station, lidar_gen_ds, data_source='lidar', save_mode='both')

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
        X = np.arange(0, total_time_bins, 1)
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
        new_height_size = int(total_height_bins / scale)
        new_heights = np.linspace(start=start_h, stop=end_h, num=new_height_size, endpoint=True)

        temp = xr.DataArray(np.zeros((new_height_size, total_time_bins, len(wavelengths))),
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
