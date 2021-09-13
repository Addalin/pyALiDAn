import calendar
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.utils.data
import xarray as xr

import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.utils import utils, xr_utils, vis_utils, proc_utils, global_settings as gs

eps = np.finfo(np.float).eps
torch.manual_seed(8318)


# TODO:  add 2 flags - Debug and save figure.
# TODO : organize main() to functions & comments
# %% Helper Functions
def decay_p(t, t0, p0, days_decay):
    return p0 * (np.exp(-(t - t0) / days_decay))


def var_p(p, maxv, minv, val):
    return 0.25 * (1 - p / maxv) / ((maxv - minv) / maxv) + 0.05 * (1 - (1 - val / maxv) / ((maxv - minv) / maxv))


sns.set_theme()
vis_utils.set_visualization_settings()


def generate_LC_pattern_main(params):
    station_name = params.station_name
    start_date = params.start_date
    end_date = params.end_date

    # Load extended calibration database
    station = gs.Station(station_name=station_name)
    wavelengths = gs.LAMBDA_nm().get_elastic()

    data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(os.curdir))), 'data')

    ds_extended_name = f"dataset_{station_name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.nc"
    ds_path_extended = os.path.join(data_folder, ds_extended_name)
    ds_extended = xr_utils.load_dataset(ds_path_extended)

    if params.plot_results:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        ds_extended.LC.plot(ax=ax, hue='Wavelength', linewidth=0.8)
        ds_extended.plot.scatter(ax=ax, y='LC', x='Time',
                                 hue='Wavelength',
                                 s=8, hue_style='discrete', edgecolor='w')
        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        title = fr"{ds_extended.LC.long_name} for {start_date.strftime('%d/%m/%Y')}--{end_date.strftime('%d/%m/%Y')}"
        clean_title = f"pLC_PICASSO_{start_date.strftime('%m')}-{end_date.strftime('%m_%Y')}"
        # fig_path = os.path.join('figures', clean_title)
        # print(f"Saving fig to {fig_path}")
        # plt.savefig(fig_path + '.jpeg')
        # plt.savefig(fig_path + '.svg')
        ax.set_title(title)
        plt.show()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        # requires - conda install -c conda-forge nc-time-axis
        ds_extended.sel(Time=slice(start_date, start_date + timedelta(hours=24))).LC.plot(ax=ax, hue='Wavelength',
                                                                                          linewidth=0.5)
        ds_extended.sel(Time=slice(start_date, start_date + timedelta(hours=24))).plot.scatter(ax=ax, y='LC', x='Time',
                                                                                               hue='Wavelength', s=15,
                                                                                               hue_style='discrete',
                                                                                               edgecolor='w')
        ax.set_title(fr"{ds_extended.LC.long_name} for {start_date.strftime('%d/%m/%Y')}")
        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

    # ### Creating pattern of laser power through days
    # #### 1. The lidar factor is dependent on optical and geometrical values of the system.
    # #### 2. From the LC retrieved by TROPOS it seems as it has an exponential decay through the period.
    # #### 3. Therefore, first generating decay power $p(t)=p_0\cdot\exp(-\frac{t-t_0}{t_{decay}})$
    # #### 4. Then calculating upper and lower bounding curves of interval of confidence.
    # The interval of confidence is  set as $[\pm5\%,\pm25\% ]$.
    # Higher confidence is for higher power values (meaning small interval of confidence).
    # #### 5. Then the new power is randomly generated withing the interval of confidence per time $t$

    # Set the times for generating random powers.
    freq_H = 5  # choose some hourly frequency e.g. 3,4,7 hrs...
    time_indx = pd.date_range(start=start_date, end=end_date, freq=f'{freq_H}H')
    if time_indx[-1] < end_date:
        # This is to make sure the closing time is at least the end_date.
        # If not then taking another period of freq_H, to make sure the final interpolation
        # will have generated power values for the whole period.
        final_dt = time_indx[-1] + timedelta(hours=freq_H)
        final_dtidx = pd.DatetimeIndex(data=[final_dt])
        time_indx = time_indx.append(final_dtidx)
    elif time_indx[-1] == end_date:
        final_dt = end_date
    df_times = pd.DataFrame(time_indx, columns=['date'])
    bins_per_day = timedelta(days=1) / timedelta(hours=freq_H)
    df_times['t_day'] = (df_times.index / bins_per_day).values

    # Set parameters for generating a decay power pattern p(t)
    days_decay = 70
    peak_days = np.array([-1, 50])
    period1 = (df_times.t_day >= peak_days[0]) & (df_times.t_day < peak_days[1])
    period2 = (df_times.t_day >= peak_days[1])
    max_powers = [25000, 70000, 60000]
    ds_chans = []
    for wavelength, p0 in zip(wavelengths, max_powers):
        c1 = df_times[period1].apply(lambda row: decay_p(row.t_day, peak_days[0], p0, days_decay), axis=1,
                                     result_type='expand')
        c2 = df_times[period2].apply(lambda row: decay_p(row.t_day, peak_days[1], p0, days_decay), axis=1,
                                     result_type='expand')
        ds_chans.append(xr.Dataset(
            data_vars={'p': ('Time', pd.concat([c1, c2])),
                       'lambda_nm': ('Wavelength', np.uint16([wavelength]))
                       },
            coords={'Time': df_times.date.values,
                    'Wavelength': np.uint16([wavelength])
                    }))
    ds_gen_p = xr.concat(ds_chans, dim='Wavelength')
    ds_gen_p.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$\rm nm$'}
    ds_gen_p.p.attrs = {'units': r'$\rm{photons\,sr\,km^3}$', 'long_name': r'$\rm{ LC_{generated}}$',
                        'info': 'LC - Lidar constant - from generation'}
    ds_gen_p = ds_gen_p.assign(
        p_ubound=
        xr.apply_ufunc(lambda p, maxv, minv: p + p * var_p(p, maxv, minv, ds_gen_p.p),
                       ds_gen_p.p, ds_gen_p.p.max(dim='Time'),
                       ds_gen_p.p.min(dim='Time'), keep_attrs=True),
        p_lbound=
        xr.apply_ufunc(lambda p, maxv, minv: p - p * var_p(p, maxv, minv, ds_gen_p.p),
                       ds_gen_p.p, ds_gen_p.p.max(dim='Time'),
                       ds_gen_p.p.min(dim='Time'), keep_attrs=True))

    ds_gen_p = ds_gen_p.assign(p_new=xr.apply_ufunc(lambda lbound, ubound, rand: lbound + (ubound - lbound) * rand,
                                                    ds_gen_p.p_lbound, ds_gen_p.p_ubound,
                                                    np.random.rand(3, ds_gen_p.Time.size), keep_attrs=True))

    if params.plot_results:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        ds_gen_p.p.plot(ax=ax, hue='Wavelength', linewidth=0.8)
        ax.set_title(fr"{ds_gen_p.p.long_name} for {start_date.strftime('%d/%m/%Y')}--{end_date.strftime('%d/%m/%Y')}")
        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        ds_gen_p.p.plot(ax=ax, hue='Wavelength', linewidth=0.5)
        for wavelength, c in zip(wavelengths, vis_utils.COLORS):
            ax.fill_between(ds_gen_p.Time.values,
                            ds_gen_p.p_lbound.sel(Wavelength=wavelength).values,
                            ds_gen_p.p_ubound.sel(Wavelength=wavelength).values,
                            color=c, alpha=.1)
        ax.set_title(fr"{ds_gen_p.p.long_name} for {start_date.strftime('%d/%m/%Y')}--{end_date.strftime('%d/%m/%Y')}")
        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        ds_gen_p.p.plot(ax=ax, hue='Wavelength', linewidth=0.5)
        ds_gen_p.plot.scatter(ax=ax, y='p_new', x='Time', hue='Wavelength', s=8, hue_style='discrete', edgecolor='w')
        for wavelength, c in zip(wavelengths, vis_utils.COLORS):
            ax.fill_between(ds_gen_p.Time.values,
                            ds_gen_p.p_lbound.sel(Wavelength=wavelength).values,
                            ds_gen_p.p_ubound.sel(Wavelength=wavelength).values,
                            color=c, alpha=.1)
        ax.set_title(
            fr"Random {ds_gen_p.p.long_name} for {start_date.strftime('%d/%m/%Y')}--{end_date.strftime('%d/%m/%Y')}")
        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.show()

    # #### 5. Calculate interpolated lidar power per each wavelength for the period from the randomized powers.
    # - The calculation is based on a fit of randomised powers $p_t, t \in [0,t_{np}]$, with Bezier interpolation.
    # - The interpolation uses $dn$ bins (points) between each consecutive per of points $p_t, p_{t+1}$.
    # - $dn$ is set according to lidar measuring frequency of $\delta_t = 30[s]$
    # - For a period starting at $t_0$, ending at $t_{np}$, there are $np$ randomised points
    # 	-  The total measurements bins will be $n_{total}=\frac {t_1 - t_0 [s]}{\delta_t[s]}$
    # 	-  The amount of interpolated bins between each couple is $dn = \frac{n_{total}}{np-1}$,
    # 	where $np-1$ is the number of cubic curves to evaluate.
    #
    # #### 6. Converting Bezier paths to LC(t) and creating dataset of generated lidar power.

    # Set the period for calculating bezier fitting
    end_time = final_dt  # start_time +timedelta(hours=freq_H)# final_dt# datetime(2017,10,31)
    tslice = slice(start_date, end_time)
    p_slice = ds_gen_p.p_new.sel(Wavelength=wavelength, Time=tslice)
    n_pts = p_slice.Time.size
    t0 = p_slice.Time[0].values
    t1 = p_slice.Time[-1].values
    dt0 = utils.dt64_2_datetime(t0)
    dt1 = utils.dt64_2_datetime(t1)
    difft = (dt1 - dt0)

    n_total = difft.days * station.total_time_bins + difft.seconds / station.freq
    dn_t = np.int(n_total / (n_pts - 1))
    # initialize the points at times of n_pts
    points = np.empty((n_pts, 2))
    points[:, 0] = np.array([n * dn_t for n in range(n_pts)])
    # Set the time index in which the interpolation is calculated.
    power_time_index = pd.date_range(start=start_date, end=final_dt, freq=f'{station.freq}S')
    paths_chan = []
    for wavelength in wavelengths:
        points[:, 1] = ds_gen_p.p_new.sel(Wavelength=wavelength, Time=tslice).values
        path = proc_utils.Bezier.evaluate_bezier(points, dn_t)
        paths_chan.append(xr.Dataset(
            data_vars={'p': ('Time', path[:, 1]),
                       'lambda_nm': ('Wavelength', np.uint16([wavelength]))
                       },
            coords={'Time': power_time_index.values,
                    'Wavelength': np.uint16([wavelength])
                    }))
    new_p = xr.concat(paths_chan, dim='Wavelength')
    new_p.p.attrs = {'units': r'$\rm{photons\,sr\,km^3}$',
                     'long_name': r'$\rm{ LC_{generated}}$',
                     'info': 'LC - Lidar constant - from generation'}
    new_p.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$\rm nm$'}


    if params.plot_results:
        # Single plot of generated bezier interpolation
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
        for wavelength, c in zip(wavelengths, vis_utils.COLORS):
            ax.fill_between(ds_gen_p.Time.values,
                            ds_gen_p.p_lbound.sel(Wavelength=wavelength).values,
                            ds_gen_p.p_ubound.sel(Wavelength=wavelength).values,
                            color=c, alpha=.1)
        ds_gen_p.plot.scatter(ax=ax, y='p_new', x='Time', hue='Wavelength', s=10, hue_style='discrete', edgecolor='w')
        new_p.p.plot(ax=ax, hue='Wavelength', linewidth=0.8)
        title = fr"B\'ezier interpolation of {new_p.p.long_name} for " \
                fr"{start_date.strftime('%d/%m/%Y')}-- {end_date.strftime('%d/%m/%Y')}"
        clean_title = f"pLC_gen_{start_date.strftime('%m')}-{end_date.strftime('%m_%Y')}"
        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%m-%d')
        ax.xaxis.set_major_formatter(myFmt)

        ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        # fig_path = os.path.join('figures', clean_title)
        # print(f"Saving fig to {fig_path}")
        # plt.savefig(fig_path + '.jpeg')
        # plt.savefig(fig_path + '.svg')
        ax.set_title(title)
        plt.show()



        curdays = [] # [start_date + timedelta(days=1 * n * 8) for n in range(8)] # Uncomment to plot daily Bezier inter
        for cur_day in curdays:
            day_slice = slice(cur_day, cur_day + timedelta(hours=24) - timedelta(seconds=30))
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 5))
            new_p.p.sel(Time=day_slice).plot(ax=ax, hue='Wavelength', linewidth=0.8)
            ds_gen_p.sel(Time=day_slice).plot.scatter(ax=ax, y='p_new', x='Time', hue='Wavelength', s=15,
                                                      hue_style='discrete',
                                                      edgecolor='w')
            ax.ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
            ax.set_title(fr"B\'ezier interpolation of {new_p.p.long_name} - for {cur_day.strftime('%d/%m/%Y')}")
            plt.tight_layout()
            plt.show()


    if params.plot_results:
        # Plot both true and generated on two subplots
        import matplotlib.dates as mdates
        myFmt = mdates.DateFormatter('%m-%d')

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(14, 5), sharey=True)
        ds_extended.LC.plot(ax=ax[0], hue='Wavelength', linewidth=0.8)
        ds_extended.plot.scatter(ax=ax[0], y='LC', x='Time',
                                 hue='Wavelength',
                                 s=8, hue_style='discrete', edgecolor='w')
        ax[0].ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        plt.tight_layout()
        title = fr"{ds_extended.LC.long_name} for {start_date.strftime('%d/%m/%Y')}--{end_date.strftime('%d/%m/%Y')}"
        ax[0].set_title(title)
        ax[0].set_ylabel(r'${\rm P}_{\rm LC}[{\rm photons} \cdot {\rm km}^3]$')
        ax[0].xaxis.set_major_formatter(myFmt)
        ax[0].legend(markerscale=6)

        for wavelength, c in zip(wavelengths, vis_utils.COLORS):
            ax[1].fill_between(ds_gen_p.Time.values,
                            ds_gen_p.p_lbound.sel(Wavelength=wavelength).values,
                            ds_gen_p.p_ubound.sel(Wavelength=wavelength).values,
                            color=c, alpha=.1)
        ds_gen_p.plot.scatter(ax=ax[1], y='p_new', x='Time', hue='Wavelength', s=10, hue_style='discrete', edgecolor='w')
        new_p.p.plot(ax=ax[1], hue='Wavelength', linewidth=0.8)
        title = fr"B\'ezier interpolation of {new_p.p.long_name} for " \
                fr"{start_date.strftime('%d/%m/%Y')}-- {end_date.strftime('%d/%m/%Y')}"

        ax[1].xaxis.set_major_formatter(myFmt)
        ax[1].get_legend().remove()
        ax[1].ticklabel_format(axis='y', style="sci", scilimits=(0, 0))
        # ax[1].axes.get_yaxis().set_visible(False)
        ax[1].set_ylabel("")
        fig.autofmt_xdate(rotation=0)

        plt.tight_layout()
        clean_title = f"pLC_gen_PICASSO_{start_date.strftime('%m')}-{end_date.strftime('%m_%Y')}"
        fig_path = os.path.join('figures', clean_title)
        print(f"Saving fig to {fig_path}")
        plt.savefig(fig_path + '.jpeg')
        plt.savefig(fig_path + '.svg')
        ax[1].set_title(title)

        plt.show()

    # %% Save monthly LC dataset
    if params.save_ds:
        for month in range(start_date.month, end_date.month + 1):
            gen_utils.save_monthly_params_dataset(station, start_date.year, month, new_p, type_="LC")

        gen_utils.save_full_params_dataset(station, start_date, end_date, new_p, type_="LC")


if __name__ == '__main__':
    parser = utils.get_base_arguments()
    args = parser.parse_args()

    generate_LC_pattern_main(args)
