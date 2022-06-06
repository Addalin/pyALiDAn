import os
from datetime import datetime, timedelta, date

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import xarray as xr
from scipy import stats
from scipy.stats import multivariate_normal

from learning_lidar.utils import vis_utils

vis_utils.set_visualization_settings()

import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.utils import utils, xr_utils, vis_utils, proc_utils, global_settings as gs
from learning_lidar.generation.generate_density_utils import LR_tropos
vis_utils.set_visualization_settings()


# TODO: add debug and save of figures option
# TODO : organize main() to functions & comments

def plot_2D_KDE(sample_orig, sample_new, Z,
                label_orig: str = 'orig', label_new: str = 'new',
                label_x: str = 'x_label', label_y: str = 'y_label', s_title: str = '',
                x_lim=[0, 1], y_lim=[0, 1], figsize=(7, 5), fig_type: str = 'svg'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.scatter(x=sample_orig[0], y=sample_orig[1], s=1, c='k', label=label_orig)
    im = ax.imshow(np.rot90(Z), cmap='turbo',
                   extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
    ax.plot(sample_new[0], sample_new[1], 'k*', markersize=6)
    ax.plot(sample_new[0], sample_new[1], 'w*', markersize=4, label=label_new)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    fig.colorbar(im, ax=ax)
    plt.legend()
    ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', s_title + '.' + fig_type), bbox_inches='tight')
    ax.set_title(s_title)
    plt.show()
    return fig, ax


def gaussian2density(means_x, means_y, stds_x, stds_y, grid):
    density = np.zeros(grid.shape[0:2])
    for mu_x, mu_y, std_x, std_y in zip(means_x, means_y, stds_x, stds_y):
        cov = np.diag((std_x, std_y))
        mu = (mu_x, mu_y)
        rv = multivariate_normal(mu, cov)
        Z_i = np.reshape(rv.pdf(grid), grid.shape[0:2])
        density += Z_i
    return density


def plot_angstrom_exponent_distribution(x, y, x_label, y_label, date_):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.scatter(x=x, y=y, s=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = f"Angstrom Exponent distribution {date_}"
    plt.tight_layout()
    # plt.savefig(os.path.join('figures', title+'.pdf'))
    ax.set_title(title)
    plt.show()


def kde_estimation_main(args, month, year, data_folder):
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

    # if args.plot_results:
    # plot_angstrom_exponent_distribution(x, y, x_label=couple_0, y_label=couple_1,
    #                                    date_=t_slice.start.strftime('%Y-%m'))

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
    # TODO: the argpartition was to make sure values are within limits .
    #  so make sure the usage of rejection sampling is done correctly
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
        fig, ax = plot_2D_KDE(sample_orig=[x, y], sample_new=[ang_355_532, ang_532_1064], Z=Z,
                              label_orig='AERONET', label_new='New samples',
                              label_x=r"${\rm \AA}_{355, 532}$",
                              label_y=r"${\rm \AA}_{532, 1064}$",
                              s_title=f"pdf_angstrom_{t_slice.start.strftime('%B_%Y')}",
                              x_lim=[xmin, xmax], y_lim=[ymin, ymax])

    # ## Angstrom - Lidar Ratio

    df_a_lr = pd.read_csv(station.Angstrom_LidarRatio)

    # ### Creating joint probability $P(x=LR,y=A)$
    # 1 . Calculating multivariate normal distribution for each type in the dataset
    xmin, xmax = [25, 125]
    ymin, ymax = [0, 4]
    Z_types = []
    weight_types = []
    types = df_a_lr.type.unique().tolist()
    colors = df_a_lr.color.unique().tolist()
    grps_type = df_a_lr.groupby(df_a_lr.type).groups
    for type in types:
        df_type = df_a_lr[df_a_lr['type'] == type]
        LR_type = df_type['x']
        A_type = df_type['y']
        std_x = df_type['std_x']
        std_y = df_type['std_y']

        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid = np.dstack((X, Y))
        Z_type = gaussian2density(LR_type, A_type, std_x, std_y, grid)
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
        normal_Z += weight * z_type / z_type.sum()  # TODO: check that the sum of total density is 1

    # Sampling the grid , with the weights set by the joint distribution inorder to generate a kernel distribution
    xy = np.vstack([X.reshape(X.size), Y.reshape(Y.size)])
    kernel_LR_A = stats.gaussian_kde(xy, weights=normal_Z.reshape(normal_Z.size))
    Z = np.reshape(kernel_LR_A(xy).T, X.shape)
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        im = ax.imshow(np.rot90(Z), cmap='turbo', extent=[xmin, xmax, ymin, ymax], aspect="auto")
        sns.scatterplot(data=df_a_lr, x='x', y='y', hue='type', ax=ax)
        ax.errorbar(x=df_a_lr.x, y=df_a_lr.y, xerr=df_a_lr.std_x, yerr=df_a_lr.std_y,
                    markersize=0, fmt='o', c='k', lw=.5)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        title = f"Angstrom Exponent - Lidar Ratio distribution {t_slice.start.strftime('%Y-%m')}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Joint distribution weighted in favor of urban industrial and desert dust
    weights = np.array(
        [.05, .20, .75])  # weights for: ['biomass burning', 'biomass burning and desert dust', 'urban industrial']
    normal_Z = np.zeros(grid.shape[0:2])
    for z_type, weight in zip(Z_types, weights):
        normal_Z += weight * z_type / z_type.sum()  # TODO: check that the sum of total density is 1

    # Sampling the grid , with the weights set by the joint distribution inorder to generate a kernel distribution
    xy = np.vstack([X.reshape(X.size), Y.reshape(Y.size)])
    kernel_LR_A = stats.gaussian_kde(xy, weights=normal_Z.reshape(normal_Z.size))
    Z = np.reshape(kernel_LR_A(xy).T, X.shape)
    if args.plot_results:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        sns.scatterplot(data=df_a_lr, x='x', y='y', hue='type', ax=ax)
        ax.errorbar(x=df_a_lr.x, y=df_a_lr.y, xerr=df_a_lr.std_x, yerr=df_a_lr.std_y, markersize=0, fmt='o', c='k',
                    lw=.5)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        title = f"Angstrom Exponent - Lidar Ratio distribution {t_slice.start.strftime('%Y-%m')}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 3. Sampling $LR$ from 1D conditioned probability $P(x=LR|y=A)$
    LR_samp = []
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    for ang in ang_355_532:
        # calc conditioned density for each value of Angstrom Exponent list
        X_, Y_ = np.mgrid[xmin:xmax:200j, ang:ang:1j]
        positions_ = np.vstack([X_.ravel(), Y_.ravel()])
        Z_ = np.reshape(kernel_LR_A(positions_).T, X_.shape)
        yy_i = Z_.reshape(Z_.size)
        xx_i = X_.reshape(X_.size)
        kernel_LR_cond_A_i = stats.gaussian_kde(yy_i)
        random_state = sk.utils.check_random_state(None)
        weights = kernel_LR_cond_A_i.dataset[0, :]
        maxv = weights.max()
        minv = weights.min()
        weights = (weights - minv) / (maxv - minv)
        weights /= weights.sum()
        # TODO: sample from conditioned distribution and make sure that samples correlate with 95%.
        indx = random_state.choice(kernel_LR_cond_A_i.n, size=1, p=weights)
        ax.plot(xx_i, yy_i, linewidth=0.8)
        ax.scatter(x=xx_i[indx], y=yy_i[indx], s=10)
        LR_samp.append(xx_i[indx])

    if args.plot_results:
        plt.xlabel(r'$\rm \, LR_{355[nm]}$')
        plt.ylabel(r'$\rm A 355-532$')

        ax.grid(color='darkgray', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, Z.max()])

        title = f"Sampling from conditioned distribution $P(x=LR|y=A)$ {t_slice.start.strftime('%Y-%m')}"
        ax.set_title(title)
        plt.tight_layout()
        # plt.savefig(os.path.join('figures', title+'.pdf'))
        plt.show()

    LR_samp = np.array(LR_samp).reshape(2 * monthdays)

    if args.plot_results:
        # 4. Show the joint density, and the new samples of LR
        # Show density, and the new chosen samples
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[xmin, xmax, ymin, ymax], aspect="auto")
        sns.scatterplot(data=df_a_lr, x='x', y='y', hue='type', ax=ax)
        ax.errorbar(x=df_a_lr.x, y=df_a_lr.y, xerr=df_a_lr.std_x, yerr=df_a_lr.std_y, markersize=0, fmt='o', c='k',
                    lw=.5)
        ax.plot(LR_samp, ang_355_532, 'k*', markersize=6)
        ax.plot(LR_samp, ang_355_532, 'w*', markersize=4, label='new samples')
        plt.xlabel(r'$\rm \, LR[sr]$')
        plt.ylabel(r'${\rm \AA}$')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.legend()
        fig.colorbar(im, ax=ax)
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()

        title = f"Sampling from $P(x=LR|y=A)$ {t_slice.start.strftime('%Y-%m')}"
        clean_title = f"pdf_LR_angstrom_" + f"{t_slice.start.strftime('%B_%Y')}"
        plt.savefig(os.path.join('figures', clean_title + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join('figures', clean_title + '.jpeg'), bbox_inches='tight')
        ax.set_title(title)
        plt.show()

    # ### Sampling $r_m$ and $\beta_{532}^{max}$ for current month
    # 1 . Load database relevant to current month
    km_scale = 1E+3

    monthdays = (date(year, month + 1, 1) - date(year, month, 1)).days
    # TODO: use the new dataset which already includes the required statistics
    #  remember also to convert backscatter to km units
    csv_name = f"dataset_{station.name}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_extended.csv"
    csv_path_extended = os.path.join(data_folder, csv_name)
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

    # Sample new points # TODO: drop rm =0
    rm_bounds = [np.round_(df_rm_beta['rm'].min()),
                 np.ceil(df_rm_beta['rm'].max())]
    beta_bounds = [0.0, 1.0]

    rm_v = []
    beta_v = []
    for day in range(monthdays):
        valid_domain = False
        while ~valid_domain:
            sample_rm, sample_beta = kernel_rm_beta.resample(1)[:, 0]
            valid_domain = gen_utils.valid_box_domain(sample_rm, sample_beta,
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
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        df_rm_beta.plot.scatter(x='rm', y='beta-532', ax=ax, c='k', label='PICASO')
        im = ax.imshow(np.rot90(Z), cmap='turbo',
                       extent=[round(xmin), round(xmax), 0, ymax], aspect="auto")
        ax.plot(rm_new, beta_532_new, 'k*', markersize=6)
        ax.plot(rm_new, beta_532_new, 'w*', markersize=4, label='new samples')
        ax.set_xlabel(r'$r_{\rm ref} [{\rm km}]$')
        ax.set_ylabel(r'$\alpha_{532}^{\rm max} \left[\frac{1}{\rm km}\right]$')
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        # Set new y_ticks - Convert Beta values to alpha!
        y_ticks = [round(tick_label, 1) for tick_label in ax.get_yticks() * LR_tropos]
        ax.set_yticklabels(y_ticks)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
        fig.colorbar(im, ax=ax)
        plt.legend()
        ax.grid(color='w', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        title = r"Sampling from $r_m$ - $ \alpha_{532}^{max}$  " + f"{t_slice.start.strftime('%Y-%m')}"
        clean_title = r"pdf_alpha_refHeight_" + f"{t_slice.start.strftime('%B_%Y')}"
        plt.savefig(os.path.join('figures', clean_title + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join('figures', clean_title + '.jpeg'), bbox_inches='tight')
        ax.set_title(title)
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


def create_density_params_ds(station, rm_new, ang_355_532, ang_532_1064, LR_samp, beta_532_new, times, nc_aeronet_name) \
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
    ds_month.rm.attrs = {'units': r'$\rm km$', 'long_name': r'$r_m$',
                         'info': 'Reference range'}
    ds_month.ang355532.attrs = {'long_name': r'$\AA_{355,532}$',
                                'info': 'Angstrom Exponent 355,532'}
    ds_month.ang5321064.attrs = {'long_name': r'$\AA_{532,1064}$',
                                 'info': 'Angstrom Exponent 532,1064'}
    ds_month.LR.attrs = {'units': r'$\rm sr$', 'long_name': r'$LR$',
                         'info': 'Lidar Ratio'}
    ds_month.beta532.attrs = {'units': r'$\rm km^{{-1}} sr^{-1}$',
                              'long_name': r'$\beta$',
                              'info': '$Aerosol Backscatter'}

    return ds_month


if __name__ == '__main__':

    data_folder = gs.PKG_DATA_DIR

    parser = utils.get_base_arguments()

    parser.add_argument('--extended_smoothing_bezier', action='store_true',
                        help='Whether to do extended smoothing bezier')

    args = parser.parse_args()

    # start_date and end_date should correspond to the extended csv!
    # months to run KDE on, one month at a time.
    for date_ in pd.date_range(args.start_date, args.end_date, freq='MS'):
        kde_estimation_main(args, date_.month, date_.year, data_folder)
