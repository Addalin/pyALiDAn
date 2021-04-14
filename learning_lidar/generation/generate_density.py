import numpy as np
import pandas as pd

import learning_lidar.generation.generate_density_utils
import learning_lidar.global_settings as gs
from datetime import datetime, timedelta
import os

from learning_lidar.generation.generate_density_utils import create_ratio, set_gaussian_grid, \
    create_Z_level2, create_blur_features, create_sampled_level_interp, create_ds_density, TIMEFORMAT, \
    create_atmosphere_ds, create_sigma, calc_aod, calculate_LRs_and_ang, calc_tau_ir_uv, calc_normalized_density, \
    plot_max_density_per_time, calc_normalized_tau, convert_sigma, get_params

from learning_lidar.preprocessing import preprocessing as prep
import xarray as xr
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

if __name__ == '__main__':
    PLOT_RESULTS = True
    colors = ["darkblue", "darkgreen", "darkred"]

    # Set location and density parameters
    month = 9
    year = 2017
    cur_day = datetime(2017, 9, 2, 0, 0)
    station = gs.Station(station_name='haifa')

    dr, heights, ds_day_params = get_params(station=station, year=year, month=month, cur_day=cur_day)

    LR_tropos = 55
    ref_height = np.float(ds_day_params.rm.sel(Time=cur_day).values)
    ref_height_bin = np.int(ref_height / dr)
    sigma_532_max = np.float(ds_day_params.beta532.sel(Time=cur_day).values) * LR_tropos
    # TODO not in use. Delete?
    # ang_532_10264 = np.float(ds_day_params.ang5321064.sel(Time=cur_day).values)
    # ang_355_532 = np.float(ds_day_params.ang355532.sel(Time=cur_day).values)

    LR = np.float(ds_day_params.LR.sel(Time=cur_day).values)

    end_t = cur_day + timedelta(hours=24) - timedelta(seconds=30)
    time_index = pd.date_range(start=cur_day, end=end_t, freq='30S')
    total_time_bins = len(time_index)  # 2880

    total_bins = station.n_bins
    x = np.arange(total_time_bins)
    y = np.arange(total_bins)
    X, Y = np.meshgrid(x, y, indexing='xy')
    grid = np.dstack((X, Y))

    # %% set ratio

    smooth_ratio = create_ratio(station=station, ref_height=ref_height, ref_height_bin=ref_height_bin,
                                total_bins=total_bins, y=y,
                                plot_results=PLOT_RESULTS)

    # Set a grid of Gaussian's - component 0
    Z_level0 = set_gaussian_grid(nx=5, ny=1, cov_size=1E+6, choose_ratio=.95, std_ratio=.25, cov_r_lbounds=[.8, .1],
                                 grid=grid, x=x, y=y, start_bin=0, top_bin=int(0.5 * ref_height_bin),
                                 plot_results=PLOT_RESULTS)

    # Set a grid of gaussians - component 1
    Z_level1 = set_gaussian_grid(nx=6, ny=2, cov_size=5 * 1E+4, choose_ratio=.9, std_ratio=.15, cov_r_lbounds=[.8, .1],
                                 grid=grid, x=x, y=y, start_bin=int(0.1 * ref_height_bin),
                                 top_bin=int(0.8 * ref_height_bin), plot_results=PLOT_RESULTS)

    Z_level2 = create_Z_level2(grid=grid, x=x, y=y, grid_cov_size=1E+4, ref_height_bin=ref_height_bin,
                               plot_results=PLOT_RESULTS)

    blur_features = create_blur_features(Z_level2=Z_level2, nsamples=int(total_bins * total_time_bins * .0005),
                                         plot_results=PLOT_RESULTS)

    # Subsample & interpolation of 1/4 part of the component (stretching to one day of measurments)
    # %% # TODO: create 4 X 4 X 4 combinations per evaluation , save for each

    indexes = np.round(np.linspace(0, 720, 97)).astype(int)
    target_indexes = [i * 30 for i in range(97)]
    target_indexes[-1] -= 1
    tt_index = time_index[target_indexes]

    # %% trying to set different sizes of croping : 6,8,12 or 24 hours . This is not finished yet, thus commented

    """
    interval_size = np.random.choice([6,8,12,24])
    bins_interval = 120*interval_size
    bins_interval,interval_size, bins_interval/30
    2880/30+1 , len(target_indexes),96*30, source_indexes"""

    sampled_level0_interp = create_sampled_level_interp(Z_level=Z_level0, k=np.random.uniform(0.5, 2.5),
                                                        indexes=indexes, tt_index=tt_index)

    sampled_level1_interp = create_sampled_level_interp(Z_level=Z_level1, k=np.random.uniform(0, 3),
                                                        indexes=indexes, tt_index=tt_index)

    sampled_level2_interp = create_sampled_level_interp(Z_level=blur_features, k=np.random.uniform(0, 3),
                                                        indexes=indexes, tt_index=tt_index)

    ds_density, times = create_ds_density(sampled_level0_interp=sampled_level0_interp,
                                          sampled_level1_interp=sampled_level1_interp,
                                          sampled_level2_interp=sampled_level2_interp,
                                          heights=heights, time_index=time_index,
                                          plot_results=PLOT_RESULTS)

    atmosphere_ds = create_atmosphere_ds(ds_density=ds_density, smooth_ratio=smooth_ratio, plot_results=PLOT_RESULTS)

    sigma_g, sigma_ratio = create_sigma(atmosphere_ds=atmosphere_ds, sigma_532_max=sigma_532_max,
                                        times=times, plot_results=PLOT_RESULTS)

    tau_g = calc_aod(dr=dr, sigma_g=sigma_g, plot_results=PLOT_RESULTS)
    """ 
    Angstrom Exponent
    1. To convert $\sigma_{aer}$ from $532[nm]$ to $355[nm]$ and $1064[nm]$
    2. Typical values of angstrom exponent are from `20170901_20170930_haifa_ang.nc`
    3. Sample procedure is done in :`KDE_estimation_sample.ipynb`, and data is loaded from `ds_month_params`
    """

    RUN_SINGLE_SAMPLE = False
    if RUN_SINGLE_SAMPLE:
        nc_name_aeronet = f"{month_start_day.strftime('%Y%m%d')}_{month_end_day.strftime('%Y%m%d')}_haifa_ang.nc"
        ds_ang = prep.load_dataset(os.path.join(station.aeronet_folder, nc_name_aeronet))

        t_slice = slice(cur_day, cur_day + timedelta(days=1))
        means = []
        for wavelengths in ds_ang.Wavelengths:
            angstrom_mean = learning_lidar.generation.generate_density_utils.angstrom.sel(Wavelengths=wavelengths,
                                                                                          Time=t_slice).mean().item()
            angstrom_std = learning_lidar.generation.generate_density_utils.angstrom.sel(Wavelengths=wavelengths,
                                                                                         Time=t_slice).std().item()

            textstr = ' '.join((
                r'$\mu=%.2f$, ' % (angstrom_mean,),
                r'$\sigma=%.2f$' % (angstrom_std,)))
            learning_lidar.generation.generate_density_utils.angstrom.sel(Wavelengths=wavelengths, Time=t_slice). \
                plot(x='Time', label=fr"$ \AA \, {wavelengths.item()}$, " + textstr)
            means.append(angstrom_mean)
        plt.legend()
        plt.show()
        ang_532_10264 = means[2]
        ang_355_532 = means[0]

    LRs, ang_355_532, ang_532_10264 = calculate_LRs_and_ang(ds_day_params=ds_day_params, time_index=time_index,
                                                            plot_results=PLOT_RESULTS)

    tau_ir, tau_uv = calc_tau_ir_uv(tau_g=tau_g, ang_355_532=ang_355_532, ang_532_10264=ang_532_10264,
                                    plot_results=PLOT_RESULTS)

    sigma_normalized = calc_normalized_density(sigma_ratio=sigma_ratio, plot_results=PLOT_RESULTS)

    if PLOT_RESULTS:
        plot_max_density_per_time(sigma_ratio)

    tau_normalized = calc_normalized_tau(dr=dr, sigma_normalized=sigma_normalized, plot_results=PLOT_RESULTS)

    sigma_ir = convert_sigma(tau=tau_ir, wavelen=1064, tau_normalized=tau_normalized, sigma_normalized=sigma_normalized,
                             plot_results=PLOT_RESULTS)
    sigma_uv = convert_sigma(tau=tau_uv, wavelen=355, tau_normalized=tau_normalized, sigma_normalized=sigma_normalized,
                             plot_results=PLOT_RESULTS)

    # Extinction profiles of $\sigma_{aer}$ at different times

    # Set your custom color palette
    import seaborn as sns

    sns.set_palette(sns.color_palette(colors))
    customPalette = sns.set_palette(sns.color_palette(colors))

    if PLOT_RESULTS:
        t_index = [500, 1500, 2500]
        times = [ds_density.Time[ind].values for ind in t_index]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)
        for t, ax in zip(times, axes.ravel()):
            sigma_uv.sel(Time=t).plot.line(ax=ax, y='Height', label=sigma_uv.Wavelength.item())
            sigma_ir.sel(Time=t).plot.line(ax=ax, y='Height', label=sigma_ir.Wavelength.item())
            sigma_g.sel(Time=t).plot.line(ax=ax, y='Height', label=r'$532$')

            ax.set_title(t)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Calculate $\beta_{aer}$ assuming the lidar ratio $LR=60[sr]$
    beta_uv = sigma_uv / LR
    beta_uv.attrs = {'long_name': r'$\beta$', 'units': r'$1/km \cdot sr$'}
    beta_ir = sigma_ir / LR
    beta_ir.attrs = {'long_name': r'$\beta$', 'units': r'$1/km \cdot sr$'}
    beta_g = sigma_g / LR
    beta_g.attrs = {'long_name': r'$\beta$', 'units': r'$1/km \cdot sr$'}

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        ax = axes.ravel()
        beta_uv.plot(ax=ax[0], cmap='turbo')
        beta_ir.plot(ax=ax[2], cmap='turbo')
        beta_g.plot(ax=ax[1], cmap='turbo')
        plt.tight_layout()
        plt.show()

    mol_month_folder = prep.get_month_folder_name(station.molecular_dataset, cur_day)
    nc_mol = fr"{cur_day.strftime('%Y_%m_%d')}_{station.location}_molecular.nc"
    ds_mol = prep.load_dataset(os.path.join(mol_month_folder, nc_mol))

    """
    Generation parameters
    1. $\sigma_{532}^{max}$ - max value from Tropos retrievals calculated as $\beta_{532}^{max}\cdot LR$, $LR=55sr$  (Tropos assumption)
    2. $A_{532,1064}$ - Angstrom exponent of 532-1064, as a daily mean value calculated from AERONET
    3. $A_{355,532}$ - Angstrom exponent of 355-532, as a daily mean value calculated from AERONET
    4. $LR$ - Lidar ratio, corresponding to Angstroms values (based on literature and TROPOS)
    5. $r_{max}$ - top height of aerosol layer. Taken as $\sim1.25\cdot r_{max}$, $s.t.\; r_{max}$ is the maximum value of the reference range from TROPOS retrievals of that day.
    
    Source files:
    1. nc_name_aeronet - netcdf file post-processed from AERONET retrivals, using: read_AERONET_dat.py ( for angstrom values)
    2. ds_extended - calibration dataset processed from TROPOS retrivals, using dataseting.py (for r_mx, sigma_max values)
    
    Create the aerosol dataset
    """

    # initializing the dataset
    ds_aer = xr.zeros_like(ds_mol)
    ds_aer = ds_aer.drop('attbsc')
    ds_aer = ds_aer.assign(date=ds_mol.date)
    ds_aer.attrs = {'info': 'Daily generated aerosol profiles',
                    'source_file': 'generate_density.ipynb',
                    'location': station.name,
                    }
    ds_aer.lambda_nm.loc[:] = [355, 532, 1064]

    # adding generation info
    ds_aer = ds_aer.assign(max_sigm_g=xr.Variable(dims=(), data=sigma_532_max,
                                                  attrs={'long_name': r'$\sigma_{532}^{max}$', 'units': r'$1/km$',
                                                         'info': r'A generation parameter. The maximum extinction '
                                                                 r'value from '
                                                                 r'Tropos retrievals calculated as $\beta_{532}^{'
                                                                 r'max}\cdot LR$, $LR=55sr$'}),
                           ang_532_1064=xr.Variable(dims='Time', data=ang_532_10264,
                                                    attrs={'long_name': r'$A_{532,1064}$',
                                                           'info': r'A generation parameter. Angstrom exponent of '
                                                                   r'532-1064. '
                                                                   r'The daily mean value calculated from AERONET '
                                                                   r'level 2.0'}),
                           ang_355_532=xr.Variable(dims='Time', data=ang_355_532,
                                                   attrs={'long_name': r'$A_{355,532}$',
                                                          'info': r'A generation parameter. Angstrom exponent of '
                                                                  r'355-532. '
                                                                  r'The daily mean value calculated from AERONET '
                                                                  r'level 2.0'}),
                           LR=xr.Variable(dims='Time', data=LR, attrs={'long_name': r'$\rm LR$', 'units': r'$sr$',
                                                                       'info': r'A generation parameter. A lidar '
                                                                               r'ratio, corresponds to Angstroms '
                                                                               r'values (based on literature and '
                                                                               r'TROPOS)'}),
                           r_max=xr.Variable(dims=(), data=ref_height,
                                             attrs={'long_name': r'$r_{max}$', 'units': r'$km$',
                                                    'info': r'A generation parameter. Top height of aerosol layer.'
                                                            r'Taken as $\sim1.25\cdot r_{max}$, $s.t.\; r_{max}$ is '
                                                            r'the maximum value of '
                                                            r'the reference range from TROPOS retrievals, for the '
                                                            r'date.'}),
                           params_source=xr.Variable(dims=(), data=os.path.join(gen_source_path),
                                                     attrs={
                                                         'info': 'netcdf file name, containing generated density '
                                                                 'parameters, '
                                                                 ' using: KDE_estimation_sample.ipynb .'})
                           )
    # TODO
    """ 
    1. Addapt the folowing variable to have dimention of Time : ang_532_1064, ang_355_532, LR
    2. fix 5,10,15,20,25,30,28 18 of september ang and LR
    """

    # assign $\beta$ and $\sigma$ values
    ds_aer.sigma.attrs['info'] = 'Aerosol attenuation coefficient'
    ds_aer.sigma.loc[dict(Wavelength=532)] = sigma_g.values
    ds_aer.sigma.loc[dict(Wavelength=355)] = sigma_uv.values
    ds_aer.sigma.loc[dict(Wavelength=1064)] = sigma_ir.values
    ds_aer.beta.attrs['info'] = 'Aerosol backscatter coefficient'
    ds_aer.beta.loc[dict(Wavelength=532)] = beta_g.values
    ds_aer.beta.loc[dict(Wavelength=355)] = beta_uv.values
    ds_aer.beta.loc[dict(Wavelength=1064)] = beta_ir.values

    # TODO: create ds_aer from scratch (without loading ds_mol)

    # Save the aerosols dataset
    month_str = f'0{month}' if month < 10 else f'{month}'
    nc_aer = fr"{cur_day.strftime('%Y_%m_%d')}_Haifa_aerosol_check.nc"
    nc_aer_folder = fr'D:\data_haifa\GENERATION\aerosol_dataset\{month_str}'
    prep.save_dataset(ds_aer, nc_aer_folder, nc_aer)

    # Show relative ratios between aerosols and molecular backscatter
    ratio_beta = ds_aer.beta / (ds_mol.beta + ds_aer.beta)
    ratio_beta.where(ratio_beta < 0.1).plot(x='Time', y='Height', row='Wavelength',
                                            cmap='turbo_r', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()

    ratio_sigma = ds_aer.sigma / (ds_mol.sigma + ds_aer.sigma)
    ratio_sigma.where(ratio_sigma < 0.1).plot(x='Time', y='Height', row='Wavelength',
                                              cmap='turbo_r', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()

    atmosphere_ds.density.plot(x='Time', y='Height', row='Component',
                               cmap='turbo', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()
