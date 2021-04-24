import os
import pandas as pd
import learning_lidar.preprocessing.preprocessing as prep
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter

def explore_orig_day(main_folder, station_name, start_date, end_date, cur_day, timedelta, wavelengths, time_indx):
    day_str = cur_day.strftime('%Y-%m-%d')
    ds_path_extended = os.path.join(main_folder, 'data',
                                    f"dataset_{station_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.nc")
    csv_path_extended = os.path.join(main_folder, 'data',
                                     f"dataset_{station_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.csv")
    df = pd.read_csv(csv_path_extended)
    ds_extended = prep.load_dataset(ds_path_extended)

    # %%
    day_slice = slice(cur_day, cur_day.date() + timedelta(days=1))
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
    ds_extended.sel(Time=slice(cur_day, cur_day + timedelta(hours=24))).LC.plot(hue='Wavelength', linewidth=0.8)
    ds_extended.sel(Time=slice(cur_day, cur_day + timedelta(hours=24))). \
        plot.scatter(y='LC', x='Time', hue='Wavelength', s=30, hue_style='discrete', edgecolor='w')

    ax.set_title(fr"{ds_day.LC.long_name} for {day_str}")
    plt.tight_layout()
    plt.show()
