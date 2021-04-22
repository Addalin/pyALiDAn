import os
from datetime import datetime, date, timedelta
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import griddata, CubicSpline
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from learning_lidar.preprocessing import preprocessing as prep
import learning_lidar.generation.generation_utils as gen_utils

# TODO: move plot settings to a general utils, this is being used throughout the module
TIMEFORMAT = mdates.DateFormatter('%H:%M')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Profiles at different times
t_index = [500, 1500, 2500]  # TODO param for generate_density somehow

PLOT_RESULTS = False  # TODO param for generate_density somehow
LR_tropos = 55


# TODO print --> logger.debug
# TODO: Save the logs to a dedicated logdir

# %% Functions of Daily Aerosols' Density Generation

def dt2binscale(dt_time, res_sec=30):
    """
    TODO move this function to Station
    Returns the bin index corresponds to dt_time
    binscale - is the time scale [0,2880], of a daily lidar bin index from 00:00:00 to 23:59:30.
    The lidar has a bin measure every 30 sec, in total 2880 bins per day.
    :param dt_time: datetime.datetime object
    :return: binscale - float in [0,2880]
    """
    res_minute = 60 / res_sec
    res_hour = 60 * res_minute
    res_musec = (1e-6) / res_sec
    tind = dt_time.hour * res_hour + dt_time.minute * res_minute + dt_time.second / res_sec + dt_time.microsecond * res_musec
    return tind


def get_random_sample_grid(nx, ny, orig_x, orig_y, std_ratio=.125):
    """
    TODO: add usage
    :param nx:
    :param ny:
    :param orig_x:
    :param orig_y:
    :param std_ratio:
    :return:
    """
    delta_x = (orig_x[-1] - orig_x[0]) / nx
    delta_y = (orig_y[-1] - orig_y[0]) / ny

    # generate a new grid (points are set to be in the middle of each patch)
    center_x = .5 * delta_x + (np.arange(nx) * delta_x).astype(int).reshape((1, nx)).repeat(ny, axis=0) + orig_x[0]
    center_y = .5 * delta_y + (np.arange(ny) * delta_y).astype(int).reshape((ny, 1)).repeat(nx, axis=1) + orig_y[0]

    # set random distances from centers of the new grid
    dx = (std_ratio * delta_x * np.random.randn(nx, ny)).astype(int).T
    dy = (std_ratio * delta_y * np.random.randn(nx, ny)).astype(int).T

    # set random point in each patch of the new grid
    points_x = center_x + dx
    points_y = center_y + dy

    new_grid = {'x': center_x.flatten(), 'y': center_y.flatten()}
    sample_points = {'x': points_x.flatten(), 'y': points_y.flatten()}
    return new_grid, sample_points


def get_random_cov_mat(lbound_x=.5, lbound_y=.1):
    """
    TODO: add usage
    :param lbound_x:
    :param lbound_y:
    :return:
    """
    # generating covariance matrix with higher x diagonal of gaussian
    # set : lbound_x< std_x <= 1
    std_x = 1 - lbound_x * np.random.rand()
    # set : lbound_y < std_y <= std_x
    std_y = std_x - (std_x - lbound_y) * np.random.rand()
    # %%
    # generate random correlation [-1,1]
    # this is to make sure that the covariance matrix is PSD : std_x*std_y - std_xy*std_xy >= 0
    rho = -1 + 2 * np.random.rand()
    std_xy = rho * std_y * std_x
    cov = np.array([[std_x, std_xy], [std_xy, std_y]])
    return cov


def make_interpolated_image(n_samples, im):
    """
    Randomly sampe and interpolate an image.
    :param n_samples: int object. Number of samples to make.
    :param im: np.array. Input 2D image
    :return: interp_im : np.array. A 2D image interpolated from a random selection of pixels, of the original image im.
    """
    nx, ny = im.shape[1], im.shape[0]
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    ix = np.random.randint(im.shape[1], size=n_samples)
    iy = np.random.randint(im.shape[0], size=n_samples)
    samples = im[iy, ix]
    interp_im = griddata((iy, ix), samples, (Y, X), method='nearest', fill_value=0)
    return interp_im


def create_multi_gaussian_density(grid, nx, ny, grid_x, grid_y, std_ratio=.125, choose_ratio=1.0,
                                  cov_size=1E-5, cov_r_lbounds=[.8, .1]):
    """
    TODO: add usage
    :param grid:
    :param nx:
    :param ny:
    :param grid_x:
    :param grid_y:
    :param std_ratio:
    :param choose_ratio:
    :param cov_size:
    :param cov_r_lbounds:
    :return:
    """
    # Set a grid of Gaussian's
    # 1. Define centers of Gaussians:
    new_grid, sample_points = get_random_sample_grid(nx, ny, grid_x, grid_y, std_ratio)
    if choose_ratio < 1.0:
        center_x, _, center_y, _ = train_test_split(sample_points['x'], sample_points['y'],
                                                    train_size=choose_ratio, shuffle=True)
    else:
        center_x = sample_points['x']
        center_y = sample_points['y']

    # 2. Define covariance and distribution to each gaussian, and calculated the total density
    density = np.zeros((grid.shape[0], grid.shape[1]))
    for x0, y0 in zip(center_x, center_y):
        cov = cov_size * get_random_cov_mat(lbound_x=cov_r_lbounds[0], lbound_y=cov_r_lbounds[1])
        rv = multivariate_normal((x0, y0), cov)
        density += rv.pdf(grid)
    # normalizing:
    density = normalize(density)
    return density


def get_sub_sample_level(density, source_indexes, target_indexes):
    """
    TODO: add usage
    :param density:
    :param source_indexes:
    :param target_indexes:
    :return:
    """
    density_samples = density[:, source_indexes]
    df_sigma = pd.DataFrame(density_samples, columns=target_indexes)
    interp_sigma_df = (df_sigma.T.resample('30S').interpolate(method='linear')).T
    sampled_interp = interp_sigma_df.values
    sampled_interp = normalize(sampled_interp)
    return sampled_interp


def normalize(x, max_value=1):
    """
    TODO : move to util of lidar (as miscLidar)
    :param x: np.array. Input signal to normalize. can be 1D, 2D ,3D ...
    :param max_value: np.float. The max number to normalize the signal
    :return: Normalized signal
    """
    return max_value * (x - x.min()) / (x.max() - x.min())


def create_ratio(start_height, ref_height, ref_height_bin, total_bins):
    """
    TODO: add usage
    :param start_height:
    :param ref_height:
    :param ref_height_bin:
    :param total_bins:
    :return:
    """
    t_start = start_height / ref_height
    r_start = 0.7
    t_r = np.array(
        [0, 0.125 * t_start, 0.25 * t_start, .5 * t_start, t_start, 0.05, 0.1, .3, .4, .5, .6, .7, .8, .9,
         1.0]) * np.float(ref_height_bin)
    ratios = np.array([1, 1, 1, 1, 1, 1.0, 1, 1, 1, 1, 0.95, 0.85, 0.4, .3, 0.2])
    t_o = np.array(
        [0, 0.125 * t_start, 0.25 * t_start, .5 * t_start, t_start, 0.05, 0.1, .3, .4, .5, .6, .7, .8, .9,
         1.0]) * np.float(ref_height_bin)
    overlaps = np.array([.0, .01, 0.02, .1, r_start, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    t_interp = np.arange(start=1, stop=total_bins + 1, step=1)
    ratio_interp = np.interp(t_interp, t_r, ratios)
    overlap_interp = np.interp(t_interp, t_o, overlaps)

    smooth_ratio = gaussian_filter1d(ratio_interp, sigma=40)  # TODO apply smooth ratio as overlap function
    smooth_overlap = gaussian_filter1d(overlap_interp, sigma=20)
    y = np.arange(total_bins)
    smooth_ratio = np.ones_like(y)

    if PLOT_RESULTS:
        plt.figure(figsize=(3, 3))
        plt.plot(smooth_ratio, t_interp, label='density ratio')
        plt.plot(smooth_overlap, t_interp, linestyle='--', label='overlap')
        plt.ylabel('Height bins')
        plt.xlabel('ratio')
        plt.show()

    return smooth_ratio


def set_gaussian_component(nx, ny, cov_size, choose_ratio, std_ratio, cov_r_lbounds, grid,
                           x, y, start_bin, top_bin):
    """
    TODO: add usage
    :param nx:
    :param ny:
    :param cov_size:
    :param choose_ratio:
    :param std_ratio:
    :param cov_r_lbounds:
    :param start_bin: setting height bounds for randomizing Gaussians
    :param top_bin: setting height bounds for randomizing Gaussians
    :param PLOT_RESULTS:
    :return:
    """

    grid_y = y[start_bin:top_bin]
    grid_x = x

    density = create_multi_gaussian_density(grid, nx, ny, grid_x, grid_y, std_ratio,
                                            choose_ratio, cov_size, cov_r_lbounds)
    if PLOT_RESULTS:
        plt.figure()
        im = plt.imshow(density, cmap='turbo')
        plt.colorbar(im)
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.show()

    return density


def set_gaussian_grid_features(nx, ny, x, y, start_bin, top_bin):
    """
    TODO: add usage
    :param nx:
    :param ny:
    :param x:
    :param y:
    :param start_bin: setting height bounds for randomizing Gaussians
    :param top_bin: setting height bounds for randomizing Gaussians
    :return:
    """
    grid_x = x
    grid_y = y[start_bin:top_bin]
    _, sample_points = get_random_sample_grid(nx, ny, grid_x, grid_y, std_ratio=.25)
    center_x = sample_points['x']
    center_y = sample_points['y']
    return center_x, center_y


def set_features_component(grid, x, y, grid_cov_size, ref_height_bin):
    """
    TODO: add usage
    :param grid:
    :param x:
    :param y:
    :param grid_cov_size:
    :param ref_height_bin:
    :return:
    """
    density = create_Z_level2(grid, x, y, grid_cov_size, ref_height_bin)

    blur_features = create_blur_features(density=density, n_samples=int(grid.shape[0] * grid.shape[1] * .0005))
    return blur_features


def create_Z_level2(grid, x, y, grid_cov_size, ref_height_bin):
    """
    TODO: add usage and rename
    :param grid:
    :param x:
    :param y:
    :param grid_cov_size:
    :param ref_height_bin:
    :return:
    """
    # Create Z_level2

    # Set a grid of Gaussians - component 2 - for features
    center_x, center_y = set_gaussian_grid_features(nx=9, ny=1, x=x, y=y, start_bin=int(0 * ref_height_bin),
                                                    top_bin=int(.3 * ref_height_bin))
    center_x1, center_y1 = set_gaussian_grid_features(nx=8, ny=1, x=x, y=y, start_bin=int(.2 * ref_height_bin),
                                                      top_bin=int(.5 * ref_height_bin))
    center_x2, center_y2 = set_gaussian_grid_features(nx=7, ny=1, x=x, y=y, start_bin=int(.4 * ref_height_bin),
                                                      top_bin=int(.7 * ref_height_bin))

    center_x_split_1, center_x_split_2, center_y_split_1, center_y_split_2 = \
        train_test_split(np.concatenate((center_x, center_x1, center_x2), axis=0),
                         np.concatenate((center_y, center_y1, center_y2), axis=0),
                         train_size=.5)

    density = np.zeros((grid.shape[0], grid.shape[1]))

    for x0, y0 in zip(center_x_split_1, center_y_split_1):
        cov = grid_cov_size * get_random_cov_mat(lbound_x=.7, lbound_y=.01)
        rv = multivariate_normal((x0, y0), cov)
        r = 1
        density += r * rv.pdf(grid)

    for x0, y0 in zip(center_x_split_2, center_y_split_2):
        cov = grid_cov_size * get_random_cov_mat(lbound_x=.7, lbound_y=.01)
        rv = multivariate_normal((x0, y0), cov)
        r = -1
        density += r * rv.pdf(grid)

    density = normalize(density)

    if PLOT_RESULTS:
        plt.figure()
        im = plt.imshow(density, cmap='turbo')
        plt.colorbar(im)
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.show()

    return density


def create_blur_features(density, n_samples):
    """
    TODO: add usage
    :param density:
    :param n_samples:
    :return:
    """
    # Gradients of component 2
    g_filter = np.array([[0, -1, 0],
                         [-1, 0, 1],
                         [0, 1, 0]])
    grad = signal.convolve2d(density, g_filter, boundary='symm', mode='same')
    grad_norm = normalize(grad)
    grad_amplitude = np.absolute(grad)
    grad_norm_amplitude = normalize(grad_amplitude)

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        ax = axes.ravel()
        ax_i = ax[0]

        im = ax_i.imshow(grad_norm, cmap='turbo')
        plt.colorbar(im, ax=ax_i)
        ax_i.set_aspect('equal')
        ax_i.invert_yaxis()
        ax_i.set_title('normalized gradient')

        ax_i = ax[1]
        im = ax_i.imshow(grad_norm_amplitude, cmap='turbo')
        plt.colorbar(im, ax=ax_i)
        ax_i.set_aspect('equal')
        ax_i.invert_yaxis()
        ax_i.set_title('gradient - magnitude normalized')

        ax_i = ax[2]
        im = ax_i.imshow(np.angle(grad), cmap='hsv')
        plt.colorbar(im, ax=ax_i)
        ax_i.set_aspect('equal')
        ax_i.invert_yaxis()
        ax_i.set_title('gradient - angle')
        plt.show()

    # Subsample and interpolation of the absolute of gradients - component 2
    interp_features = make_interpolated_image(n_samples, grad_norm_amplitude)
    blur_features = gaussian_filter(interp_features, sigma=(21, 61))
    blur_features = normalize(blur_features)

    if PLOT_RESULTS:
        plt.figure()
        im = plt.imshow(blur_features, cmap='turbo')
        plt.colorbar(im)
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.title('gradient - absolut interpolated')
        plt.show()

    return blur_features


def random_subsampled_density(density, k, time_index, level_id):
    """
    TODO: add usage
    :param density:
    :param k:
    :param time_index:
    :param level_id:
    :return:
    """
    # Subsample & interpolation of 1/4 part of the component (stretching to one day of measurments)
    # TODO: create 4 X 4 X 4 combinations per evaluation , save for each

    indexes = np.round(np.linspace(0, 720, 97)).astype(int)
    target_indexes = [i * 30 for i in range(97)]
    target_indexes[-1] -= 1
    tt_index = time_index[target_indexes]

    # trying to set different sizes of cropping : 6,8,12 or 24 hours . This is not finished yet, thus commented
    """
    interval_size = np.random.choice([6,8,12,24])
    bins_interval = 120*interval_size
    bins_interval,interval_size, bins_interval/30
    2880/30+1 , len(target_indexes),96*30, source_indexes
    """
    shift_bins = int(720 * k)
    print(f'component {level_id} shift bins {shift_bins}')
    source_indexes = indexes + shift_bins
    sampled_level_interp = get_sub_sample_level(density, source_indexes, tt_index)
    return sampled_level_interp


def generate_daily_density(sampled_level0_interp, sampled_level1_interp, sampled_level2_interp, heights, time_index):
    """
    TODO: add usage
    :param sampled_level0_interp:
    :param sampled_level1_interp:
    :param sampled_level2_interp:
    :param heights:
    :param time_index:
    :return:
    """
    components = []
    for indl, component in enumerate([sampled_level0_interp, sampled_level1_interp, sampled_level2_interp]):
        ds_component = xr.Dataset(
            data_vars={'density': (('Height', 'Time'), component),
                       'component': ('Component', [indl])},
            coords={'Height': heights,
                    'Time': time_index.tolist(),
                    'Component': [indl]})
        components.append(ds_component)
    ds_density = xr.concat(components, dim='Component')
    ds_density.Height.attrs = {'units': 'km'}

    if PLOT_RESULTS:

        ds_density.density.plot(x='Time', y='Height', row='Component', cmap='turbo', figsize=(8, 10), sharex=True)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)
        times = [ds_density.Time[ind].values for ind in t_index]
        for t, ax in zip(times,
                         axes.ravel()):
            ds_density.density.sel(Time=t).plot.line(ax=ax, y='Height')
        # plt.tight_layout()
        plt.show()

    return ds_density


def merge_density_components(ds_density):
    """
    TODO: add usage
    :param ds_density:
    :return:
    """
    # Random Merge of  density components
    # Built as a linear combination of density components
    weight_0 = 0.6 + 0.25 * np.random.randn()
    if weight_0 < 0.3:
        weight_0 = 0.5 + abs(weight_0)
    weight_1 = 0.3 + 0.25 * np.random.randn()
    if weight_1 < 0:
        weight_1 = 0.3 + abs(weight_1)
    weight_2 = 0.4 + 0.25 * np.random.randn()
    if weight_2 < 0:
        weight_2 = 0.4 + abs(weight_2)
    print("weights:", [weight_0, weight_1, weight_2])
    ds_density = ds_density.assign({'weights': ('Component', [weight_0, weight_1, weight_2])})
    merged = xr.zeros_like(ds_density.density[0])

    # Summing up the components to an aerosol density $\rho_{aer}$
    for l in ds_density.Component:
        merged += ds_density.density.sel(Component=l) * ds_density.weights.sel(Component=l)

    # Normalizing and smooth the final density
    rho = xr.apply_ufunc(lambda x, r: normalize(r * x),
                         merged, ds_density.ratio, keep_attrs=False)

    rho.attrs = {'info': 'Normalized Merged density', 'name': 'Density',
                 'long_name': r'$\rho$', 'units': r'$A.U.$'}

    ds_density = ds_density.assign(rho=rho)

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ds_density.rho.plot(cmap='turbo', ax=ax)
        ax.set_title(ds_density.rho.attrs['info'])
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    return ds_density


def calc_temporal_normalized_density(rho):
    """

    :param rho: xarray.Dataset(). A 2D normalized density with dimensions of : Height, Time
    :return: Normalizing the original density of sigma per time measurement
    """
    rho_norm_t = []
    for t in rho.Time:
        rho_norm_t.append(xr.apply_ufunc(lambda x: normalize(x), rho.sel(Time=t), keep_attrs=True))

    rho_temp_norm = xr.concat(rho_norm_t, dim='Time')
    rho_temp_norm = rho_temp_norm.transpose('Height', 'Time')
    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        rho_temp_norm.plot(cmap='turbo')
        plt.title('Temporally Normalized density')
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    return rho_temp_norm


def plot_max_density_per_time(rho):
    # maximum density values per time
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    rho.max(dim='Height').plot(ax=ax)
    ax.set_title(r'$\rho_{aer}^{max}(t)$')
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()


def generate_density_components(total_time_bins, total_height_bins, time_index, heights, ref_height_bin):
    """
    TODO: add usage
    :param total_time_bins:
    :param total_height_bins:
    :param time_index:
    :param heights:
    :param ref_height_bin:
    :return:
    """
    x = np.arange(total_time_bins)
    y = np.arange(total_height_bins)
    X, Y = np.meshgrid(x, y, indexing='xy')
    grid = np.dstack((X, Y))

    # Set component 0
    component_0 = set_gaussian_component(nx=5, ny=1, cov_size=1E+6, choose_ratio=.95, std_ratio=.25,
                                         cov_r_lbounds=[.8, .1], grid=grid, x=x, y=y, start_bin=0,
                                         top_bin=int(0.5 * ref_height_bin))

    # Set component 1
    component_1 = set_gaussian_component(nx=6, ny=2, cov_size=5 * 1E+4, choose_ratio=.9, std_ratio=.15,
                                         cov_r_lbounds=[.8, .1], grid=grid, x=x, y=y,
                                         start_bin=int(0.1 * ref_height_bin), top_bin=int(0.8 * ref_height_bin))

    # Set component 2
    component_2 = set_features_component(grid=grid, x=x, y=y, grid_cov_size=1E+4, ref_height_bin=ref_height_bin)

    # Randomly subsampled components
    subsamp_component_0 = random_subsampled_density(density=component_0, k=np.random.uniform(0.5, 2.5),
                                                    time_index=time_index, level_id='0')

    subsamp_component_1 = random_subsampled_density(density=component_1, k=np.random.uniform(0, 3),
                                                    time_index=time_index, level_id='1')

    subsamp_component_2 = random_subsampled_density(density=component_2, k=np.random.uniform(0, 3),
                                                    time_index=time_index, level_id='2')

    ds_density = generate_daily_density(sampled_level0_interp=subsamp_component_0,
                                        sampled_level1_interp=subsamp_component_1,
                                        sampled_level2_interp=subsamp_component_2, heights=heights,
                                        time_index=time_index)

    return ds_density


def generate_density(station, day_date, ds_day_params):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param ds_day_params:
    :return:
    """
    # TODO: add log.debug "start generating density for day_date..."

    # Set generation parameters of density
    ref_height = np.float(ds_day_params.rm.sel(Time=day_date).values)
    time_index = station.calc_daily_time_index(day_date)
    heights = station.get_height_bins_values()
    dr = heights[1] - heights[0]
    total_bins = heights.size
    ref_height_bin = np.int(ref_height / dr)

    # Create density components
    ds_density = generate_density_components(total_time_bins=station.total_time_bins, total_height_bins=station.n_bins,
                                             time_index=time_index, heights=heights, ref_height_bin=ref_height_bin)

    # set ratio - this is mainly for overlap function and/ or applying differnet endings on the daily profile.
    # currently the ratio is "1" for all bins.
    ratio = create_ratio(start_height=heights[0], ref_height=ref_height, ref_height_bin=ref_height_bin,
                         total_bins=total_bins)
    ds_density = ds_density.assign({'ratio': ('Height', ratio)})

    # Merge density components
    ds_density = merge_density_components(ds_density)

    # TODO: add log.debug "finished generating density for day_date..."
    return ds_density


# %%
# %% Function of Daily Aerosols' Optical Density Generation

def calc_beta(sigma_uv, sigma_ir, sigma_g, LR):
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

    return beta_uv, beta_ir, beta_g


def angstrom(tau_1, tau_2, lambda_1, lambda_2):
    """
    TODO : move to util of lidar (as miscLidar)
    calculates angstrom exponent
    :param tau_1: AOD Aerosol optical depth at wavelength lambda_1
    :param tau_2: AOD Aerosol optical depth at wavelength lambda_2
    :param lambda_1: wavelength lambda_1 , lambda_1<lambda_2 (e.g. 355 nm)
    :param lambda_2: wavelength lambda_2 , lambda_1<lambda_2 (e.g. 532 nm)
    :return: angstrom exponent A_1,2
    """
    return -np.log(tau_1 / tau_2) / np.log(lambda_1 / lambda_2)


def convert_sigma(tau, wavelength, tau_normalized, sigma_normalized):
    """convert $\sigma_{X}$"""
    # X = 1064 or 355
    # $\sigma_{X}^{max}(t) = \frac{\tau_{X}(t)}{\tau_N(t)}, \;\forall\, t \in Time_{day} $
    sigma_max = tau / tau_normalized
    sigma_ir = sigma_normalized * sigma_max

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        sigma_max.plot()
        title = r'$\sigma^{max}_{' + str(wavelength) + '}(t) $'
        ax.set_title(title)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        sigma_ir.plot(cmap='turbo', ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    return sigma_ir


def create_sigma(ds_density, sigma_532_max):
    # Creating $\sigma_{532}$
    # To create the aerosol, the density is:
    """1. Normalized
    2. Corrected according to ratio above
    3. multiplied with a typical $\sigma_{aer, 532}$, e.g.$\sigma_{max}=0.025[1 / km]$"""

    sigma_g = xr.apply_ufunc(lambda x: normalize(x, sigma_532_max),
                             ds_density.rho.copy(deep=True), keep_attrs=False)
    sigma_g['Wavelength'] = 532

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        sigma_g.plot(cmap='turbo', ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 6))
        times = [ds_density.Time[ind].values for ind in t_index]
        for t in times:
            sigma_g.sel(Time=t).plot(ax=ax, y='Height')
        plt.tight_layout()
        plt.show()

    return sigma_g


def calc_aod_ds(sigma_ds):
    """
    TODO: add usage
    :param sigma_ds:
    :return:
    """
    # Calculate Aerosol Optical Depth (AOD)
    # $\tau_{aer,\lambda} = \int \sigma_{aer,\lambda} (r) dr\;\; \forall \, r \leq r_{ref} $
    dr_vec = xr.apply_ufunc(lambda x: np.insert(x[1:] - x[0:-1], 0, x[0]), sigma_ds.Height, keep_attrs=True)
    tau = dr_vec * sigma_ds
    aod = tau.sum(dim='Height')
    aod.attrs = {'name': 'aod', 'long_name': r'$\tau$', 'info': 'Aerosol Optical Depth'}
    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        aod.plot(ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    return aod


def calc_normalized_tau(dr, sigma_normalized):
    # normalized tau
    tau_normalized = dr * sigma_normalized.sum(dim='Height')
    tau_normalized.name = r'$\tau_N$'
    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        plt.title('Normalized AOD')
        tau_normalized.plot(ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    return tau_normalized


def calculate_LR_and_ang(ds_day_params, time_index):
    # Estimate AOD of $\lambda=1064nm$ and  $\lambda=355nm$
    """
        Angstrom Exponent
        1. To convert $\sigma_{aer}$ from $532[nm]$ to $355[nm]$ and $1064[nm]$
        2. Typical values of angstrom exponent are from `20170901_20170930_haifa_ang.nc`
        3. Sample procedure is done in :`KDE_estimation_sample.ipynb`, and data is loaded from `ds_month_params`

        $\tau_{1064} = \frac{\tau_{532}}{(532 / 1064) ^ {-A_{532, 1064}}}$

        $\tau_{355} =\tau_{532} \cdot(355 / 532) ^ {-A_{355, 532}} $
    """

    # Cubic spline interpolation of A_355,532, A_532,1064 and LR  during the time bins of the day

    # 1. Setting values to interpolate
    ang355532s = ds_day_params.ang355532.values
    ang5321064s = ds_day_params.ang5321064.values
    LRs = ds_day_params.LR.values

    # 2. Setting the time parameter of the curve - tbins
    tbins = np.round([int(dt2binscale(datetime.utcfromtimestamp(dt.tolist() / 1e9))) for
                      dt in ds_day_params.ang355532.Time.values])

    # The last bin is added or updated to 2880 artificially.
    # If it was zero, it represents the time 00:00 of the next day.
    # Thus forcing 2800 to have a sequence over an entire day or avoiding a circular curve.
    # This is a temporary solution.
    # TODO: Create the full day interpolation in:KDE_estimation_sample.ipynb.Then load the daily values from the parameters csv. Similar to the process done for the background signal.
    if tbins[-1] > 0:
        tbins = np.append(tbins, 2880)
        ang355532s = np.append(ang355532s, ang355532s.mean())
        ang5321064s = np.append(ang5321064s, ang5321064s.mean())
        LRs = np.append(LRs, LRs.mean())
    else:
        tbins[2] = 2880

    # Fit of the splines
    cs_355532 = CubicSpline(tbins, ang355532s)
    cs_5321064 = CubicSpline(tbins, ang5321064s)
    cs_LR = CubicSpline(tbins, LRs)  # TODO replace to bazier interpolation

    # Calculate the spline for day bins: meaning: 2880 bins + 1 for 00:00 on the next day
    LR = cs_LR(np.arange(time_index.size))
    ang_355_532 = cs_355532(np.arange(time_index.size))
    ang_532_10264 = cs_5321064(np.arange(time_index.size))

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ax.plot(time_index.tolist(), LR)
        ds_day_params.plot.scatter(y='LR', x='Time', ax=ax)  # .plot.(ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ds_day_params.plot.scatter(y='ang355532', x='Time', ax=ax, label=ds_day_params.ang355532.long_name)
        ax.plot(time_index.tolist(), ang_355_532)
        ds_day_params.plot.scatter(y='ang5321064', x='Time', ax=ax, label=ds_day_params.ang5321064.long_name)
        ax.plot(time_index.tolist(), ang_532_10264)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        ax.set_ylabel(r'$\AA$')
        plt.legend()
        plt.show()

    return LR, ang_355_532, ang_532_10264


def calc_tau_ir_uv(tau_g, ang_355_532, ang_532_10264):
    tau_ir = (tau_g / ((532 / 1064) ** (-ang_532_10264))).copy(deep=True)
    tau_ir['Wavelength'] = 1064
    tau_uv = (tau_g * ((355 / 532) ** (-ang_355_532))).copy(deep=True)
    tau_uv['Wavelength'] = 355

    if PLOT_RESULTS:
        tau_ir.plot(label=r'$1064nm$')
        tau_g.plot(label=r'$532nm$')
        tau_uv.plot(label=r'$355nm$')
        plt.title('AOD')
        plt.legend()
        plt.show()

    return tau_ir, tau_uv


def generate_aerosol(station, day_date, ds_day_params, ds_density):
    # TODO: add log.debug "start generating aerosol optical density for day_date..."

    sigma_532_max = np.float(ds_day_params.sel(Time=day_date).beta532.values) * LR_tropos

    sigma_g = create_sigma(ds_density=ds_density, sigma_532_max=sigma_532_max)

    tau_g = calc_aod_ds(sigma_ds=sigma_g)

    time_index = station.calc_daily_time_index(day_date)
    LR, ang_355_532, ang_532_10264 = calculate_LR_and_ang(ds_day_params=ds_day_params, time_index=time_index)

    tau_ir, tau_uv = calc_tau_ir_uv(tau_g=tau_g, ang_355_532=ang_355_532, ang_532_10264=ang_532_10264)

    rho_normalized = calc_temporal_normalized_density(rho=ds_density.rho)

    if PLOT_RESULTS:
        plot_max_density_per_time(ds_density.rho)

    tau_normalized = calc_aod_ds(sigma_ds=rho_normalized)

    sigma_ir = convert_sigma(tau=tau_ir, wavelength=1064, tau_normalized=tau_normalized,
                             sigma_normalized=rho_normalized)
    sigma_uv = convert_sigma(tau=tau_uv, wavelength=355, tau_normalized=tau_normalized, sigma_normalized=rho_normalized)

    if PLOT_RESULTS:
        plot_extinction_profiles_sigme_diff_times(sigma_uv=sigma_uv, sigma_ir=sigma_ir, sigma_g=sigma_g)

    # Creating Daily Lidar Aerosols' sigma dataset
    sigma_ds = xr.concat([sigma_uv, sigma_g, sigma_ir], dim='Wavelength')
    sigma_ds.attrs = {'info': "Daily aerosols' generated extinction coefficient",
                      'long_name': r'$\sigma$', 'units': r'$1/km$'}

    # Creating Daily Lidar Aerosols' beta dataset
    beta_ds = xr.apply_ufunc(lambda sigma, LR: sigma / LR, sigma_ds, LR, keep_attrs=True)
    beta_ds.attrs = {'info': "Daily aerosols' generated backscatter coefficient",
                     'long_name': r'$\beta$', 'units': r'$1/km \cdot sr$'}

    # Wrap Daily Lidar Aerosols' dataset
    ds_aer = wrap_aerosol_dataset(station=station, day_date=day_date, ds_day_params=ds_day_params, sigma_ds=sigma_ds,
                                  beta_ds=beta_ds, sigma_532_max=sigma_532_max, ang_532_10264=ang_532_10264,
                                  ang_355_532=ang_355_532, LR=LR)

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        for wavelength, ax in zip(ds_aer.Wavelength.values, axes.ravel()):
            ds_aer.beta.sel(Wavelength=wavelength).plot(ax=ax, cmap='turbo')
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        for wavelength, ax in zip(ds_aer.Wavelength.values, axes.ravel()):
            ds_aer.sigma.sel(Wavelength=wavelength).plot(ax=ax, cmap='turbo')
        plt.tight_layout()
        plt.show()

    # TODO: add log.debug "finished generating aerosol optical density for day_date..."
    return ds_aer


def wrap_aerosol_dataset(station, day_date, ds_day_params, sigma_ds, beta_ds, sigma_532_max,
                         ang_532_10264, ang_355_532, LR):
    """
    Wrapping the Daily Lidar Aerosols' dataset with the followings:
    - beta
    - sigma
    - Generation parameters
        1. $\sigma_{532}^{max}$ - max value from Tropos retrievals calculated as $\beta_{532}^{max}\cdot LR$, $LR=55sr$  (Tropos assumption)
        2. $A_{532,1064}$ - Angstrom exponent of 532-1064, as a daily mean value calculated from AERONET
        3. $A_{355,532}$ - Angstrom exponent of 355-532, as a daily mean value calculated from AERONET
        4. $LR$ - Lidar ratio, corresponding to Angstroms values (based on literature and TROPOS)
        5. $r_{max}$ - top height of aerosol layer. Taken as $\sim1.25\cdot r_{max}$, $s.t.\; r_{max}$ is the maximum value of the reference range from TROPOS retrievals of that day.
    """

    ds_aer = xr.Dataset()
    ds_aer = ds_aer.assign(
        sigma=sigma_ds, beta=beta_ds,
        max_sigma_g=xr.Variable(dims=(), data=sigma_532_max,
                                attrs={'long_name': r'$\sigma_{532}^{max}$', 'units': r'$1/km$',
                                       'info': r'A generation parameter. The maximum extinction value from '
                                               r'Tropos retrievals calculated as $\beta_{532}^{max}\cdot LR$, $LR=55sr$'}),
        ang_532_1064=xr.Variable(dims=('Time'), data=ang_532_10264, attrs={'long_name': r'$A_{532,1064}$',
                                                                           'info': r'A generation parameter. Angstrom '
                                                                                   r'exponent of 532-1064. '
                                                                                   r'The daily mean value calculated '
                                                                                   r'from AERONET level 2.0'}),
        ang_355_532=xr.Variable(dims=('Time'), data=ang_355_532, attrs={'long_name': r'$A_{355,532}$',
                                                                        'info': r'A generation parameter. Angstrom '
                                                                                r'exponent of 355-532. '
                                                                                r'The daily mean value calculated '
                                                                                r'from AERONET level 2.0'}),
        LR=xr.Variable(dims=('Time'), data=LR, attrs={'long_name': r'$\rm LR$', 'units': r'$sr$',
                                                      'info': r'A generation parameter. The lidar ratio, corresponds to '
                                                              r'Angstroms values (based on literature and TROPOS)'}),
        r_max=xr.Variable(dims=(), data=np.float(ds_day_params.rm.sel(Time=day_date).values),
                          attrs={'long_name': r'$r_{max}$', 'units': r'$km$',
                                 'info': r'A generation parameter. The top height of aerosol layer.'}),
        params_source=xr.Variable(dims=(), data=gen_utils.get_month_density_gen_fname(station, day_date),
                                  attrs={'info': 'netcdf file name, containing generated density parameters,'
                                                 ' using: KDE_estimation_sample.ipynb .'}))
    ds_aer.attrs = {'info': 'Daily generated aerosol profiles',
                    'source_file': 'generate_density.ipynb',  # for python module: os.path.basename(__file__)
                    'location': station.location, }
    ds_aer.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    ds_aer.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    ds_aer['date'] = day_date
    return ds_aer


def plot_extinction_profiles_sigme_diff_times(sigma_uv, sigma_ir, sigma_g):
    # Extinction profiles of $\sigma_{aer}$ at different times
    times = [sigma_uv.Time[ind].values for ind in t_index]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)
    for t, ax in zip(times, axes.ravel()):
        sigma_uv.sel(Time=t).plot.line(ax=ax, y='Height', label=sigma_uv.Wavelength.item())
        sigma_ir.sel(Time=t).plot.line(ax=ax, y='Height', label=sigma_ir.Wavelength.item())
        sigma_g.sel(Time=t).plot.line(ax=ax, y='Height', label=r'$532$')

        ax.set_title(t)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% General Helper functions

def explore_gen_day(station, day_date, ds_aer, ds_density):
    # Show relative ratios between aerosols and molecular backscatter

    mol_month_folder = prep.get_month_folder_name(station.molecular_dataset, day_date)
    nc_mol = fr"{day_date.strftime('%Y_%m_%d')}_{station.location}_molecular.nc"
    ds_mol = prep.load_dataset(os.path.join(mol_month_folder, nc_mol))
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

    ds_density.density.plot(x='Time', y='Height', row='Component',
                            cmap='turbo', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()


def get_daily_gen_param_ds(station, day_date):
    """
    Returns the daily parameters of density creation as a dataset.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :return: ds_day_params: xarray.Dataset(). Daily dataset of generation parameters.
    """
    gen_source_path = gen_utils.get_month_density_gen_fname(station, day_date)
    ds_month_params = prep.load_dataset(gen_source_path)
    ds_day_params = ds_month_params.sel(Time=slice(day_date, day_date + timedelta(days=1)))

    return ds_day_params
