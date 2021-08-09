import os
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sns
import xarray as xr
from scipy import stats
from scipy.stats import multivariate_normal

import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.utils import utils, xr_utils, vis_utils, proc_utils, global_settings as gs

vis_utils.set_visualization_settings()


# TODO: add debug and save of figures option
# TODO : organize main() to functions & comments
# TODO Highlight rejected sampling

def valid_box_domain(x, y, bounds_x, bounds_y):
    return bounds_x[0] <= x <= bounds_x[1] and bounds_y[0] <= y <= bounds_y[1]


def plot_angstrom_exponent_distribution(x, y, x_label, y_label, date_):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(x=x, y=y, s=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Angstrom Exponent distribution {date_}")
    plt.tight_layout()
    plt.show()


def kde_estimation_main(args, month, year, DATA_DIR):
    #  Load measurements from AERONET for current month
    station = gs.Station(station_name=args.station_name)
    start_date, end_date = args.start_date, args.end_date

    folder_name = station.aeronet_folder
    monthdays = (date(year, month + 1, 1) - date(year, month, 1)).days
    start_day = datetime(year, month, 1, 0, 0)
    end_day = datetime(year, month, monthdays, 0, 0)
    nc_aeronet_name = f"{start_day.strftime('%Y%m%d')}_{end_day.strftime('%Y%m%d')}_{station.location.lower()}_ang.nc"
    ds_ang = xr_utils.load_dataset(os.path.join(folder_name, nc_aeronet_name))

    # ## Angstrom Exponent
    # ### $A_{355,532}$ vs. $A_{532,1064 }$ for the current month
    # 1. Perform a kernel density estimation on the data
    t_slice = slice(start_day, end_day + timedelta(days=1) - timedelta(seconds=30))
    couple_0 = f"{355}-{532}"
    couple_1 = f"{532}-{1064}"

    x, y = ds_ang.angstrom.sel(Wavelengths=couple_0).values, ds_ang.angstrom.sel(Wavelengths=couple_1).values

    if args.plot_results:
        plot_angstrom_exponent_distribution(x, y, x_label=couple_0, y_label=couple_1, date_=t_slice.start.strftime('%Y-%m'))

    # 2. Perform a kernel density estimation on the data
    # 3. Resample the estimated density generate new values for $A_{355,532}$ & $A_{532,1064 }$, per each day

    # Remove nan values
    valid_ind = np.where(~np.isnan(x) & ~np.isnan(y))[0]  # or np.where(y==np.nan) # and y~np.nan)
    x, y = x[valid_ind], y[valid_ind]
    values = np.vstack([x, y])

    # Estimate kernel
    kernel = stats.gaussian_kde(values)

    # Sample new points
    [x1, y1] = kernel.resample(2 * monthdays)
    scores_new = kernel(np.vstack([x1, y1]))
    max_ind = np.argpartition(scores_new, -2 * monthdays)[-2 * monthdays:]
    ang_355_532, ang_532_1064 = x1[max_ind], y1[max_ind]

    # Calc 2D function of the density
    xmin, xmax = min(x.min(), x1.min()), max(x.max(), x1.max())
    ymin, ymax = min(y.min(), y1.min()), max(y.max(), y1.max())
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    # Show density and the new chosen samples
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        ax.scatter(x=x, y=y, s=1, c='k', label='AERONET')
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax])
        ax.plot(ang_355_532, ang_532_1064, 'k*', markersize=6)
        ax.plot(ang_355_532, ang_532_1064, 'w*', markersize=4, label='new samples')
        ax.set_xlabel(couple_0)
        ax.set_ylabel(couple_1)
        ax.set_title(f"Sampling from Angstrom Exponent distribution {t_slice.start.strftime('%Y-%m')}")
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ## Angstrom - Lidar Ratio

    df_a_lr = pd.read_csv(station.Angstrom_LidarRatio)
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        df_a_lr[df_a_lr['type'] == 'red'].plot.scatter(x='x', y='y',
                                                       label=df_a_lr[df_a_lr['type'] == 'red']['name'].unique()[0],
                                                       c='r',
                                                       ax=ax)
        df_a_lr[df_a_lr['type'] == 'black'].plot.scatter(x='x', y='y',
                                                         label=df_a_lr[df_a_lr['type'] == 'black']['name'].unique()[0],
                                                         c='b', ax=ax)
        df_a_lr[df_a_lr['type'] == 'green'].plot.scatter(x='x', y='y',
                                                         label=df_a_lr[df_a_lr['type'] == 'green']['name'].unique()[0],
                                                         c='g', ax=ax)
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([25, 125])
        plt.ylim([0, 4])
        plt.show()

    # ### Creating joint probability $P(x=LR,y=A)$
    # 1 . Calculating multivariate normal distribution for each type in the dataset
    xmin, xmax = [25, 125]
    ymin, ymax = [0, 4]
    Z_types = []
    weight_types = []
    for type in ['red', 'black', 'green']:
        df_type = df_a_lr[df_a_lr['type'] == type]
        LR_type = df_type['x']
        A_type = df_type['y']
        std_x = df_type['dx'] * .5
        std_y = df_type['dy'] * .5

        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid = np.dstack((X, Y))
        Z_type = np.zeros((grid.shape[0], grid.shape[1]))

        for x0, y0, stdx, stdy in zip(LR_type, A_type, std_x, std_y):
            cov = np.diag((stdx, stdy))
            rv = multivariate_normal((x0, y0), cov)
            Z_i = np.reshape(rv.pdf(grid), X.shape)
            Z_type += Z_i
        Z_types.append(Z_type)
        weight_types.append(len(df_type))

    # 2 . Calculating the joint distribution by:
    # - Normalizing each to 1
    # - Weighted sum of distributions.
    # > Note: The weights are set according to the relative portion of the type in the original dataset.

    # Joint distribution according to the dataset
    weights = np.array(weight_types)
    weights = weights / weights.sum()
    normal_Z = np.zeros((grid.shape[0], grid.shape[1]))
    for z_type, weight in zip(Z_types, weights):
        normal_Z += weight * z_type / z_type.sum()

    # Sampling the grid , with the weights set by the joint distribution inorder to generate a kernel distribution
    xy = np.vstack([X.reshape(X.size), Y.reshape(Y.size)])
    kernel_LR_A = stats.gaussian_kde(xy, weights=normal_Z.reshape(normal_Z.size))
    Z = np.reshape(kernel_LR_A(xy).T, X.shape)
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        for type in ['red', 'black', 'green']:
            x_type, y_type, label_type, x_err, y_err = df_a_lr[df_a_lr['type'] == type]['x'], \
                                                       df_a_lr[df_a_lr['type'] == type]['y'], \
                                                       df_a_lr[df_a_lr['type'] == type]['name'].unique()[0], \
                                                       df_a_lr[df_a_lr['type'] == type]['dx'] * .5, \
                                                       df_a_lr[df_a_lr['type'] == type]['dy'] * .5
            ax.errorbar(x_type, y_type, xerr=x_err, yerr=y_err, markersize=0, fmt='o', c='k', lw=.5)
            ax.plot(x_type, y_type, '.k', markersize=8)
            ax.plot(x_type, y_type, '.' + type[0], markersize=5, label=label_type)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        ax.set_title(f"Angstrom Exponent - Lidar Ratio distribution {t_slice.start.strftime('%Y-%m')}")
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Joint distribution weighted in favor of urban industrial and desert dust
    weights = np.array([.05, .75, .20])
    normal_Z = np.zeros((grid.shape[0], grid.shape[1]))
    for z_type, weight in zip(Z_types, weights):
        normal_Z += weight * z_type / z_type.sum()

    # Sampling the grid , with the weights set by the joint distribution inorder to generate a kernel distribution
    xy = np.vstack([X.reshape(X.size), Y.reshape(Y.size)])
    kernel_LR_A = stats.gaussian_kde(xy, weights=normal_Z.reshape(normal_Z.size))
    Z = np.reshape(kernel_LR_A(xy).T, X.shape)
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        for type in ['red', 'black', 'green']:
            x_type, y_type, label_type, x_err, y_err = df_a_lr[df_a_lr['type'] == type]['x'], \
                                                       df_a_lr[df_a_lr['type'] == type]['y'], \
                                                       df_a_lr[df_a_lr['type'] == type]['name'].unique()[0], \
                                                       df_a_lr[df_a_lr['type'] == type]['dx'] * .5, \
                                                       df_a_lr[df_a_lr['type'] == type]['dy'] * .5
            ax.errorbar(x_type, y_type, xerr=x_err, yerr=y_err, markersize=0, fmt='o', c='k', lw=.5)
            ax.plot(x_type, y_type, '.k', markersize=8)
            ax.plot(x_type, y_type, '.' + type[0], markersize=5, label=label_type)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        ax.set_title(f"Angstrom Exponent - Lidar Ratio distribution {t_slice.start.strftime('%Y-%m')}")
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3. Sampling $LR$ from 1D conditioned probability $P(x=LR|y=A)$
    LR_samp = []
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for ang in ang_355_532:
        # calc conditioned density for each value of Angstrom Exponent list
        X_, Y_ = np.mgrid[xmin:xmax:200j, ang:ang:1j]
        positions_ = np.vstack([X_.ravel(), Y_.ravel()])
        Z_ = np.reshape(kernel_LR_A(positions_).T, X_.shape)
        yy_i = Z_.reshape(Z_.size)
        xx_i = X_.reshape(X_.size)
        kernel_LR_cond_A_i = stats.gaussian_kde(yy_i)
        random_state = sns.utils.check_random_state(None)
        weights = kernel_LR_cond_A_i.dataset[0, :]
        maxv = weights.max()
        minv = weights.min()
        weights = (weights - minv) / (maxv - minv)
        weights /= weights.sum()
        indx = random_state.choice(kernel_LR_cond_A_i.n, size=1, p=weights)
        ax.plot(xx_i, yy_i, linewidth=0.8)
        ax.scatter(x=xx_i[indx], y=yy_i[indx], s=10)
        LR_samp.append(xx_i[indx])
    plt.xlabel(r'$\rm \, LR_{355[nm]}$')
    plt.ylabel(r'$\rm A 355-532$')

    ax.set_title(f"Sampling from conditioned distribution $P(x=LR|y=A)$ {t_slice.start.strftime('%Y-%m')}")
    plt.tight_layout()
    ax.grid(color='darkgray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, Z.max()])
    plt.show()

    LR_samp = np.array(LR_samp).reshape(2 * monthdays)

    if args.plot_results:
        # 4. Show the joint density, and the new samples of LR
        # Show density, and the new chosen samples
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for type in ['red', 'black', 'green']:
            x_type, y_type, label_type, x_err, y_err = df_a_lr[df_a_lr['type'] == type]['x'], \
                                                       df_a_lr[df_a_lr['type'] == type]['y'], \
                                                       df_a_lr[df_a_lr['type'] == type]['name'].unique()[0], \
                                                       df_a_lr[df_a_lr['type'] == type]['dx'] * .5, \
                                                       df_a_lr[df_a_lr['type'] == type]['dy'] * .5
            ax.errorbar(x_type, y_type, xerr=x_err, yerr=y_err, markersize=0, fmt='o', c='k', lw=.5)
            ax.plot(x_type, y_type, '.k', markersize=8)
            ax.plot(x_type, y_type, '.' + type[0], markersize=5, label=label_type)
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        ax.plot(LR_samp, ang_355_532, 'k*', markersize=6)
        ax.plot(LR_samp, ang_355_532, 'w*', markersize=4, label='new samples')
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        ax.set_title(f"Sampling from $P(x=LR|y=A)$ {t_slice.start.strftime('%Y-%m')}")
        plt.legend()
        fig.colorbar(im, ax=ax)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ### Sampling $r_m$ and $\beta_{532}^{max}$ for current month
    # 1 . Load database relevant to current month
    km_scale = 1E+3

    monthdays = (date(year, month + 1, 1) - date(year, month, 1)).days
    csv_name = f"dataset_{station.name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.csv"
    csv_path_extended = os.path.join(DATA_DIR, csv_name)
    df_extended = pd.read_csv(csv_path_extended)
    df_extended['date'] = pd.to_datetime(df_extended['date'], format='%Y-%m-%d')
    grps_month = df_extended.groupby(df_extended['date'].dt.month).groups
    key_month = month
    df_month = df_extended.iloc[grps_month.get(key_month).values].reset_index()
    rm = []
    beta_532 = []
    grps_files = df_month.groupby(df_month.profile_path).groups
    for key, v in zip(grps_files.keys(), grps_files.values()):
        ds_profile = xr_utils.load_dataset(key)
        max_beta_532 = ds_profile.aerBsc_klett_532.values.max()
        max_beta_532 *= km_scale  # converting 1/(sr m) to  1/(sr km)
        cur_rm = df_month['rm'].iloc[v.values[0]]
        beta_532.append(max_beta_532)
        rm.append(cur_rm)

    # 2 . Estimate kernel density for $r_{m}$ vs. $\beta_{532}^{max}$
    df_rm_beta = pd.DataFrame(columns=['rm', 'beta-532'], data=np.array([rm, beta_532]).T)
    # Remove irrelevant values
    index_to_remove = df_rm_beta[(df_rm_beta['beta-532'] > 1.0) | (df_rm_beta['beta-532'] < 0.0)].index
    df_rm_beta.drop(index=index_to_remove, inplace=True)
    df_rm_beta.dropna(inplace=True)
    x = df_rm_beta['rm'].T
    y = df_rm_beta['beta-532'].T
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        df_rm_beta.plot.scatter(x='rm', y='beta-532', ax=ax, s=10)
        xmin, xmax = [x.min(), x.max()]
        ymin, ymax = [y.min(), y.max()]
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.show()

    # 3. Sample new values for  $r_{m}$ , $\beta_{532}^{max}$ per each day

    # Estimate kernel
    values = np.vstack([x, y])

    kernel_rm_beta = stats.gaussian_kde(values)

    # Sample new points
    rm_bounds = [np.round_(df_rm_beta['rm'].min()),
                 np.ceil(df_rm_beta['rm'].max())]
    beta_bounds = [0.0, 1.0]

    rm_v = []
    beta_v = []
    for day in range(monthdays):
        valid_domain = False
        while ~valid_domain:
            sample_rm, sample_beta = kernel_rm_beta.resample(1)[:, 0]
            valid_domain = valid_box_domain(sample_rm, sample_beta,
                                            rm_bounds, beta_bounds)
        rm_v.append(sample_rm)
        beta_v.append(sample_beta)
    print(rm_v, beta_v)
    rm_new, beta_532_new = np.array(rm_v), np.array(beta_v)
    # x_,y_ = kernel_rm_beta.resample(monthdays+10)
    # scores_new = kernel_rm_beta([x_, y_])
    # max_ind = np.argpartition(scores_new, -monthdays)[-monthdays:]
    # rm_new,beta_532_new = x_[max_ind],y_[max_ind]

    # Calc 2D function of the density
    xmin, xmax = [min(x.min(), rm_new.min()), max(x.max(), rm_new.max())]
    ymin, ymax = [min(y.min(), beta_532_new.min()), max(y.max(), beta_532_new.max())]
    X, Y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
    xy = np.vstack([X.reshape(X.size), Y.reshape(Y.size)])
    Z = np.reshape(kernel_rm_beta(xy).T, X.shape)

    if args.plot_results:
        # Show density and the new chosen samples
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
        df_rm_beta.plot.scatter(x='rm', y='beta-532', ax=ax, c='k')
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        ax.plot(rm_new, beta_532_new, 'k*', markersize=6)
        ax.plot(rm_new, beta_532_new, 'w*', markersize=4, label='new samples')
        ax.set_xlabel(r'$r_m$')
        ax.set_ylabel(r'$\beta_{532}^{max}$')
        ax.set_title(r"Sampling from $r_m$ - $ \beta_{532}^{max}$  " + f"{t_slice.start.strftime('%Y-%m')}")
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Create dataset of parameters for generating month signals

    # resizing beta and rm
    listr = [[r, r] for r in rm_new]
    rm_new = np.array(listr).reshape(monthdays * 2)
    listb = [[b, b] for b in beta_532_new]
    beta_532_new = np.array(listb).reshape(monthdays * 2)

    start_day = datetime(year, month, 1, 0, 0)
    end_day = datetime(year, month, monthdays, 0, 0)
    days = pd.date_range(start_day, end_day + timedelta(hours=12), freq='12H')

    ds_month = create_density_params_ds(station, rm_new=rm_new, ang_355_532=ang_355_532, ang_532_1064=ang_532_1064,
                                        LR_samp=LR_samp, beta_532_new=beta_532_new, times=days.values,
                                        nc_aeronet_name=nc_aeronet_name)
    # save the dataset
    if args.save_ds:
        gen_source_path = gen_utils.get_month_gen_params_path(station, start_day, type_='density_params')
        print(gen_source_path)
        xr_utils.save_dataset(ds_month, os.path.dirname(gen_source_path), os.path.basename(gen_source_path))

    if args.extended_smoothing_bezier:
        # #### 6. Converting Bezier paths to LC(t) and creating dataset of generated lidar power.
        # Set the period for calculating bezier fitting
        start_time = start_day
        end_time = end_day + timedelta(
            days=1)  # - timedelta(seconds = 30) #start_time +timedelta(hours=freq_H)# final_dt# datetime(2017,10,31)
        tslice = slice(start_time, end_time)
        p_slice = ds_month.ang355532.sel(Time=tslice)
        n_pts = p_slice.Time.size
        t0 = p_slice.Time[0].values
        t1 = p_slice.Time[-1].values
        dt0 = utils.dt64_2_datetime(t0)
        dt1 = utils.dt64_2_datetime(t1)
        difft = (end_time - start_time)

        n_total = difft.days * station.total_time_bins + difft.seconds / station.freq
        dn_t = np.int(n_total / (n_pts))
        # initialize the points at times of n_pts
        points = np.empty((n_pts + 1, 2))
        points[:, 0] = np.array([n * dn_t for n in range(n_pts + 1)])
        points[0:monthdays * 2, 1] = p_slice.values
        points[-1, 1] = p_slice.values[-1]
        # calc bezier
        path_ang355532 = proc_utils.Bezier.evaluate_bezier(points, int(dn_t))
        # Set the time index in which the interpolation is calculated.
        time_index = pd.date_range(start=start_date, end=end_time, freq=f'30S')
        ds_bezier = xr.Dataset(
            data_vars={'ang355532': (('Time'), path_ang355532[0:-1, 1])},
            coords={'Time': time_index.values[0:-1]})

        ds_bezier.ang355532.plot()
        ds_month.plot.scatter(x='Time', y='ang355532')
        plt.show()


def create_density_params_ds(station, rm_new, ang_355_532, ang_532_1064, LR_samp, beta_532_new, times, nc_aeronet_name)\
        -> xr.Dataset:
    """
    Wraps the variables into a xr.Dataset
    """

    ds_month = xr.Dataset(data_vars={'rm': ('Time', rm_new),
                                     'ang355532': ('Time', ang_355_532),
                                     'ang5321064': ('Time', ang_532_1064),
                                     'LR': ('Time', LR_samp),
                                     'beta532': ('Time', beta_532_new)},
                          coords={'Time': times})

    ds_month = ds_month.assign(
        aeronet_source=xr.Variable(dims=(), data=os.path.join(station.aeronet_folder, nc_aeronet_name),
                                   attrs={'info': 'netcdf file name, processed from AERONET retrievals,'
                                                  ' using: read_AERONET_data.py.'}))
    ds_month.rm.attrs = {'units': r'$km$', 'long_name': r'$r_m$',
                         'info': 'Reference range'}
    ds_month.ang355532.attrs = {'long_name': r'$\AA_{355,532}$',
                                'info': 'Angstrom Exponent 355,532'}
    ds_month.ang5321064.attrs = {'long_name': r'$\AA_{532,1064}$',
                                 'info': 'Angstrom Exponent 532,1064'}
    ds_month.LR.attrs = {'units': r'$sr$', 'long_name': r'$LR$',
                         'info': 'Lidar Ratio'}
    ds_month.beta532.attrs = {'units': r'$km^{{-1}} sr^{-1}$',
                              'long_name': r'$\beta$',
                              'info': '$Aerosol Backscatter'}

    return ds_month


if __name__ == '__main__':

    HOME_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    DATA_DIR = os.path.join(HOME_DIR, 'data')

    parser = utils.get_base_arguments()
    parser.add_argument('--save_ds', action='store_true',
                        help='Whether to save the datasets')
    parser.add_argument('--extended_smoothing_bezier', action='store_true',
                        help='Whether to do extended smoothing bezier')
    parser.add_argument('--plot_results', action='store_true',
                        help='Whether to plot graphs')
    args = parser.parse_args()

    # start_date and end_date should correspond to the extended csv!
    # months to run KDE on, one month at a time.
    for date_ in pd.date_range(args.start_date, args.end_date, freq='MS'):
        kde_estimation_main(args, date_.month, date_.year, DATA_DIR)
