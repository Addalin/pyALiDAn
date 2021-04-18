from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import learning_lidar.global_settings as gs
from learning_lidar.generation.generate_density_utils import explore_gen_day, PLOT_RESULTS, wrap_dataset, \
    generate_aerosol, calc_time_index, get_ds_day_params_and_path,  get_dr_and_heights
from learning_lidar.generation.generation_utils import save_generated_dataset

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

colors = ["darkblue", "darkgreen", "darkred"]
sns.set_palette(sns.color_palette(colors))
customPalette = sns.set_palette(sns.color_palette(colors))


def generate_density(station, cur_day, month, year):
    """
    Creating Daily Lidar Aerosols' dataset with:
    - beta
    - sigma
    - Generation parameters
        1. $\sigma_{532}^{max}$ - max value from Tropos retrievals calculated as $\beta_{532}^{max}\cdot LR$, $LR=55sr$  (Tropos assumption)
        2. $A_{532,1064}$ - Angstrom exponent of 532-1064, as a daily mean value calculated from AERONET
        3. $A_{355,532}$ - Angstrom exponent of 355-532, as a daily mean value calculated from AERONET
        4. $LR$ - Lidar ratio, corresponding to Angstroms values (based on literature and TROPOS)
        5. $r_{max}$ - top height of aerosol layer. Taken as $\sim1.25\cdot r_{max}$, $s.t.\; r_{max}$ is the maximum value of the reference range from TROPOS retrievals of that day.
    - Source files:
        1. nc_name_aeronet - netcdf file post-processed from AERONET retrivals, using: read_AERONET_dat.py ( for angstrom values)
        2. ds_extended - calibration dataset processed from TROPOS retrivals, using dataseting.py (for r_mx, sigma_max values)
    """

    # Get and compute the different basic parameters
    dr, heights = get_dr_and_heights(station)
    ds_day_params, gen_source_path = get_ds_day_params_and_path(station=station, year=year, month=month,
                                                                cur_day=cur_day)
    ref_height = np.float(ds_day_params.rm.sel(Time=cur_day).values)
    time_index = calc_time_index(cur_day)
    start_height = 1e-3 * (station.start_bin_height + station.altitude)

    # Generate the aerosol
    sigma_ds, beta_ds, sigma_532_max, ang_532_10264, ang_355_532, LR = generate_aerosol(ds_day_params=ds_day_params,
                                                                                        dr=dr, heights=heights,
                                                                                        total_bins=station.n_bins,
                                                                                        ref_height=ref_height,
                                                                                        time_index=time_index,
                                                                                        start_height=start_height,
                                                                                        cur_day=cur_day)

    # Creating Daily Lidar Aerosols' dataset
    ds_aer = wrap_dataset(sigma_ds=sigma_ds, beta_ds=beta_ds, sigma_532_max=sigma_532_max, ang_532_10264=ang_532_10264,
                          ang_355_532=ang_355_532, LR=LR, ref_height=ref_height, station_name=station.name,
                          gen_source_path=gen_source_path, cur_day=cur_day)

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        for wavelength, ax in zip(ds_aer.Wavelength.values, axes.ravel()):
            ds_aer.beta.sel(Wavelength=wavelength).plot(ax=ax, cmap='turbo')
        plt.tight_layout()
        plt.show()

    # Save the aerosols dataset
    save_generated_dataset(station, ds_aer, data_source='aerosol', save_mode='single')

    return ds_aer


if __name__ == '__main__':
    station = gs.Station(station_name='haifa')

    days_list = [datetime(2017, 9, 2, 0, 0)]
    for cur_day in days_list:
        ds_aer = generate_density(station, cur_day=cur_day, month=9, year=2017)

        EXPLORE_GEN_DAY = False
        if EXPLORE_GEN_DAY:
            explore_gen_day(station, ds_aer, cur_day)