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


def random_subsampled_density(density, k, time_index, level_id) -> object:
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


def merge_density_components(density_ds):
    """
    TODO: add usage
    :param density_ds:
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
    density_ds = density_ds.assign({'weights': ('Component', [weight_0, weight_1, weight_2])})
    merged = xr.zeros_like(density_ds.density[0])

    # Summing up the components to an aerosol density $\rho_{aer}$
    for l in density_ds.Component:
        merged += density_ds.density.sel(Component=l) * density_ds.weights.sel(Component=l)

    merged.attrs = {'info': 'Merged density', 'name': 'Density',
                    'long_name': r'$\rho$', 'units': r'$A.U.$'}

    density_ds = density_ds.assign(merged=merged)

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        density_ds.merged.plot(cmap='turbo', ax=ax)
        ax.set_title(density_ds.merged.info)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()

    return density_ds


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

    # Randomly subsample components
    subsample_component_0 = random_subsampled_density(density=component_0, k=np.random.uniform(0.5, 2.5),
                                                    time_index=time_index, level_id='0')

    subsample_component_1 = random_subsampled_density(density=component_1, k=np.random.uniform(0, 3),
                                                      time_index=time_index, level_id='1')

    subsample_component_2 = random_subsampled_density(density=component_2, k=np.random.uniform(0, 3),
                                                      time_index=time_index, level_id='2')

    components = [subsample_component_0, subsample_component_1, subsample_component_2]
    components_ds = []
    for ind, component in enumerate(components):
        component_ds = xr.Dataset(
            data_vars={'density': (('Height', 'Time'), component),
                       'component': ('Component', [ind])},
            coords={'Height': heights,
                    'Time': time_index.tolist(),
                    'Component': [ind]})
        components_ds.append(component_ds)
    density_ds = xr.concat(components_ds, dim='Component')
    density_ds.Height.attrs = {'units': 'km'}

    if PLOT_RESULTS:

        density_ds.density.plot(x='Time', y='Height', row='Component', cmap='turbo', figsize=(8, 10), sharex=True)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)
        times = [density_ds.Time[ind].values for ind in t_index]
        for t, ax in zip(times,
                         axes.ravel()):
            density_ds.density.sel(Time=t).plot.line(ax=ax, y='Height')
            ax.set_title(pd.to_datetime(str(t)).strftime('%H:%M:%S'))
        plt.tight_layout()
        plt.show()

    return density_ds


def normalize_density_ds(density_ds):
    """
    TODO add usage
    :param density_ds:
    :return:
    """
    # Generating rho - by normalizing and smooth the merged density
    rho = xr.apply_ufunc(lambda x, r: normalize(r * x),
                         density_ds.merged, density_ds.ratio, keep_attrs=False)

    rho.attrs = {'info': 'Normalized density', 'name': 'Density',
                 'long_name': r'$\rho$', 'units': r'$A.U.$'}

    # Normalizing rho per time measurement
    rho_norm_t = []
    for t in rho.Time:
        rho_norm_t.append(xr.apply_ufunc(lambda x: normalize(x), rho.sel(Time=t), keep_attrs=True))

    rho_temp_norm = xr.concat(rho_norm_t, dim='Time')
    rho_temp_norm = rho_temp_norm.transpose('Height', 'Time')
    rho_temp_norm.attrs = {'info': 'Temporally Normalized density', 'name': 'Density',
                           'long_name': r'$\rho$', 'units': r'$A.U.$'}

    density_ds = density_ds.assign(rho=rho, rho_tnorm=rho_temp_norm)

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        rho_temp_norm.plot(cmap='turbo')
        plt.title('Temporally Normalized density')
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.show()

    if PLOT_RESULTS:
        plot_max_density_per_time(density_ds.rho)
    return density_ds


def generate_density(station, day_date, day_params_ds):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param day_params_ds:
    :return:
    """
    # TODO: add log.debug "start generating density for day_date..."

    # Set generation parameters of density
    ref_height = np.float(day_params_ds.rm.sel(Time=day_date).values)
    time_index = station.calc_daily_time_index(day_date)
    heights = station.get_height_bins_values()
    dr = heights[1] - heights[0]
    total_bins = heights.size
    ref_height_bin = np.int(ref_height / dr)

    # Create density components
    density_ds = generate_density_components(total_time_bins=station.total_time_bins, total_height_bins=station.n_bins,
                                             time_index=time_index, heights=heights, ref_height_bin=ref_height_bin)

    # set ratio - this is mainly for overlap function and/ or applying differnet endings on the daily profile.
    # currently the ratio is "1" for all bins.
    ratio = create_ratio(start_height=heights[0], ref_height=ref_height, ref_height_bin=ref_height_bin,
                         total_bins=total_bins)
    density_ds = density_ds.assign({'ratio': ('Height', ratio)})

    # Merge density components
    density_ds = merge_density_components(density_ds)

    # Normalize density
    density_ds = normalize_density_ds(density_ds)

    density_ds.attrs['source_file'] = os.path.basename(__file__)
    density_ds.attrs['station'] = station.location
    density_ds['date'] = day_date

    # TODO: add log.debug "finished generating density for day_date..."
    return density_ds


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
    TODO : move to miscLidar
    calculates angstrom exponent
    :param tau_1: AOD Aerosol optical depth at wavelength lambda_1
    :param tau_2: AOD Aerosol optical depth at wavelength lambda_2
    :param lambda_1: wavelength lambda_1 , lambda_1<lambda_2 (e.g. 355 nm)
    :param lambda_2: wavelength lambda_2 , lambda_1<lambda_2 (e.g. 532 nm)
    :return: angstrom exponent A_1,2
    """
    return -np.log(tau_1 / tau_2) / np.log(lambda_1 / lambda_2)


def sigma2aod_ds(sigma_ds):
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


def get_LR(day_params_ds, time_index):
    """
    TODO add usags
    """

    # Cubic spline interpolation of A_355,532, A_532,1064 and LR  during the time bins of the day

    # 1. Setting values to interpolate
    LRs = day_params_ds.LR.values

    # 2. Setting the time parameter of the curve - tbins
    tbins = np.round([int(dt2binscale(datetime.utcfromtimestamp(dt.tolist() / 1e9))) for
                      dt in day_params_ds.ang355532.Time.values])

    # The last bin is added or updated to 2880 artificially.
    # If it was zero, it represents the time 00:00 of the next day.
    # Thus forcing 2800 to have a sequence over an entire day or avoiding a circular curve.
    # This is a temporary solution.
    # TODO: Create the full day interpolation in:KDE_estimation_sample.ipynb.Then load the daily values from the parameters csv. Similar to the process done for the background signal.
    if tbins[-1] > 0:
        tbins = np.append(tbins, 2880)
        LRs = np.append(LRs, LRs.mean())
    else:
        tbins[2] = 2880

    # Fit of the splines
    cs_LR = CubicSpline(tbins, LRs)  # TODO replace to bazier interpolation

    # Calculate the spline for day bins: meaning: 2880 bins + 1 for 00:00 on the next day
    LR = cs_LR(np.arange(time_index.size))

    LR_ds = xr.Dataset()
    LR_ds = LR_ds.assign(
        LR=xr.Variable(dims='Time', data=LR,
                       attrs={'long_name': r'$\rm LR$', 'units': r'$sr$',
                              'info': r'A generation parameter. The lidar ratio, corresponds to Angstroms values'})). \
        assign_coords({'Time': time_index})

    if PLOT_RESULTS:
        tbins[2] -= 1
        times = LR_ds.Time[tbins].values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        LR_ds.plot(ax=ax)
        ax.scatter(times, LRs, label=LR_ds.long_name)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()

    return LR_ds


def get_angstrom(day_params_ds, tau_g):
    # Estimate AOD of $\lambda=1064nm$ and  $\lambda=355nm$
    """
        Angstrom Exponent
        1. To convert $\sigma_{aer}$ from $532[nm]$ to $355[nm]$ and $1064[nm]$
        2. Typical values of angstrom exponent are from `20170901_20170930_haifa_ang.nc`
        3. Sample procedure is done in :`KDE_estimation_sample.ipynb`, and data is loaded from `month_params_ds`

        $\tau_{1064} = \frac{\tau_{532}}{(532 / 1064) ^ {-A_{532, 1064}}}$

        $\tau_{355} =\tau_{532} \cdot(355 / 532) ^ {-A_{355, 532}} $
    """

    nbins = tau_g.Time.size
    # Cubic spline interpolation of A_355,532, A_532,1064 and LR  during the time bins of the day

    # 1. Setting values to interpolate
    ang355532s = day_params_ds.ang355532.values
    ang5321064s = day_params_ds.ang5321064.values

    # 2. Setting the time parameter of the curve - tbins
    tbins = np.round([int(dt2binscale(datetime.utcfromtimestamp(dt.tolist() / 1e9))) for
                      dt in day_params_ds.ang355532.Time.values])

    # The last bin is added or updated to 2880 artificially.
    # If it was zero, it represents the time 00:00 of the next day.
    # Thus forcing 2800 to have a sequence over an entire day or avoiding a circular curve.
    # This is a temporary solution.
    # TODO: Create the full day interpolation in:KDE_estimation_sample.ipynb.Then load the daily values from the parameters csv. Similar to the process done for the background signal.
    if tbins[-1] > 0:
        tbins = np.append(tbins, nbins)
        ang355532s = np.append(ang355532s, ang355532s.mean())
        ang5321064s = np.append(ang5321064s, ang5321064s.mean())
    else:
        tbins[2] = nbins

    # Fit of the splines
    cs_355532 = CubicSpline(tbins, ang355532s)
    cs_5321064 = CubicSpline(tbins, ang5321064s)

    # Calculate the spline for day bins: meaning: 2880 bins + 1 for 00:00 on the next day
    ang_355_532 = cs_355532(np.arange(nbins))
    ang_532_10264 = cs_5321064(np.arange(nbins))
    ang_ds = xr.Dataset()
    ang_ds = ang_ds.assign(
        ang_532_1064=xr.Variable(dims=({'Time'}), data=ang_532_10264,
                                 attrs={'long_name': r'$A_{532,1064}$',
                                        'info': r'A generation parameter. Angstrom exponent of 532-1064.'}),
        ang_355_532=xr.Variable(dims=({'Time'}), data=ang_355_532,
                                attrs={'long_name': r'$A_{355,532}$',
                                       'info': r'A generation parameter. Angstrom exponent of 355-532.'})) \
        .assign_coords({'Time': tau_g.Time.values})

    if PLOT_RESULTS:
        tbins[2] -= 1
        times = tau_g.Time[tbins].values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        ang_ds.ang_355_532.plot(ax=ax)
        ax.scatter(times, ang355532s, label=ang_ds.ang_355_532.long_name)
        ang_ds.ang_532_1064.plot(ax=ax)
        ax.scatter(times, ang5321064s, label=ang_ds.ang_532_1064.long_name)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        ax.set_ylabel(r'$\AA$')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ang_ds


def calc_ang_tau_ds(day_params_ds, sigma_g):
    tau_g = sigma2aod_ds(sigma_ds=sigma_g)
    ang_ds = get_angstrom(day_params_ds, tau_g)
    wavelength_0 = 532
    wavelength = 1064
    tau_ir = xr.apply_ufunc(lambda x, ang: (x / ((wavelength_0 / wavelength) ** (-ang))), tau_g,
                            ang_ds.ang_532_1064.values,
                            keep_attrs=True).assign_coords({'Wavelength': wavelength})

    wavelength = 355
    tau_uv = xr.apply_ufunc(lambda x, ang: (x * ((wavelength / wavelength_0) ** (-ang))), tau_g,
                            ang_ds.ang_355_532.values,
                            keep_attrs=True).assign_coords({'Wavelength': wavelength})

    tau_ds = xr.concat([tau_uv, tau_g, tau_ir], dim='Wavelength')

    if PLOT_RESULTS:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        tau_ds.plot(hue='Wavelength', ax=ax)
        ax.set_title(tau_ds.info)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()

    return tau_ds, ang_ds


def generate_sigma_ds(station, day_date, day_params_ds, density_ds):
    sigma_532_max = np.float(day_params_ds.sel(Time=day_date).beta532.values) * LR_tropos
    sigma_g = xr.apply_ufunc(lambda rho: normalize(rho, sigma_532_max),
                             density_ds.rho.copy(deep=True), keep_attrs=False).assign_coords({'Wavelength': 532})
    tau_ds, ang_ds = calc_ang_tau_ds(day_params_ds=day_params_ds, sigma_g=sigma_g)

    tau_normalized = sigma2aod_ds(sigma_ds=density_ds.rho_tnorm)

    # convert sigma to 1064, 355
    sigma_max = xr.apply_ufunc(lambda tau, tau_norm: tau / tau_norm, tau_ds, tau_normalized, keep_attrs=True)
    sigma_ir = density_ds.rho_tnorm * sigma_max.sel(Wavelength=1064)
    sigma_uv = density_ds.rho_tnorm * sigma_max.sel(Wavelength=355)

    # Creating Daily Lidar Aerosols' sigma dataset
    sigma_ds = xr.concat([sigma_uv, sigma_g, sigma_ir], dim='Wavelength')
    sigma_ds.attrs = {'info': "Daily aerosols' generated extinction coefficient",
                      'long_name': r'$\sigma$', 'units': r'$1/km$', 'name': 'sigma',
                      'source_file': os.path.basename(__file__),
                      'location': station.location, }
    sigma_ds['date'] = day_date

    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        for wavelength, ax in zip(sigma_ds.Wavelength.values, axes.ravel()):
            sigma_ds.sel(Wavelength=wavelength).plot(ax=ax, cmap='turbo')
        plt.suptitle(sigma_ds.info)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()

        times = [sigma_uv.Time[ind].values for ind in t_index]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 6), sharey=True)
        for t, ax in zip(times, axes.ravel()):
            sigma_ds.sel(Time=t).plot.line(ax=ax, y='Height', hue='Wavelength')
            ax.set_title(pd.to_datetime(str(t)).strftime('%H:%M:%S'))
        plt.suptitle(fr"$\sigma$ at different times - {pd.to_datetime(str(t)).strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.show()

        fix, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        sigma_max.plot(hue='Wavelength', ax=ax)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        ax.set_title(r'Daily $\sigma_{\rm max}$')
        plt.tight_layout()
        plt.show()

    return sigma_ds, tau_ds, ang_ds


def generate_beta_ds(station, day_date, day_params_ds, sigma_ds):
    # Creating Daily Lidar Aerosols' beta dataset
    LR_ds = get_LR(day_params_ds, time_index=sigma_ds.Time)
    beta_ds = xr.apply_ufunc(lambda LR, sigma: sigma / LR, LR_ds.LR, sigma_ds)
    beta_ds.attrs = {'info': "Daily aerosols' generated backscatter coefficient",
                     'long_name': r'$\beta$', 'units': r'$1/km \cdot sr$', 'name': 'beta',
                     'source_file': os.path.basename(__file__),
                     'location': station.location, }
    beta_ds = beta_ds.transpose('Wavelength', 'Height', 'Time')

    sigma_ds['date'] = day_date
    if PLOT_RESULTS:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 8))
        for wavelength, ax in zip(sigma_ds.Wavelength.values, axes.ravel()):
            beta_ds.sel(Wavelength=wavelength).plot(ax=ax, cmap='turbo')
        plt.suptitle(beta_ds.info)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        plt.tight_layout()
        plt.show()
    return beta_ds, LR_ds


def generate_aerosol(station, day_date, day_params_ds, density_ds):
    # TODO: add log.debug "start generating aerosol optical density for day_date..."
    sigma_ds, tau_ds, ang_ds = generate_sigma_ds(station, day_date, day_params_ds, density_ds)
    beta_ds, LR_ds = generate_beta_ds(station, day_date, day_params_ds, sigma_ds)

    # Wrap Daily Lidar Aerosols' dataset
    aer_ds = wrap_aerosol_dataset(station, day_date, day_params_ds, sigma_ds, beta_ds, ang_ds, LR_ds)

    # TODO: add log.debug "finished generating aerosol optical density for day_date..."
    return aer_ds


def wrap_aerosol_dataset(station, day_date, day_params_ds, sigma_ds, beta_ds, ang_ds, LR_ds):
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
    aer_ds = xr.Dataset()
    aer_ds = aer_ds.assign(sigma=sigma_ds, beta=beta_ds)

    # add generations parameters
    aer_ds = aer_ds.assign(LR=xr.Variable(dims='Time', data=LR_ds.LR),
                           ang_532_1064=xr.Variable(dims='Time', data=ang_ds.ang_532_1064),
                           ang_355_532=xr.Variable(dims='Time', data=ang_ds.ang_355_532),
                           max_sigma_g=xr.Variable(dims=(),
                                                   data=np.float(
                                                       day_params_ds.sel(Time=day_date).beta532.values) * LR_tropos,
                                                   attrs={'long_name': r'$\sigma_{532}^{max}$', 'units': r'$1/km$',
                                                          'info': r'A generation parameter. A typical maximum '
                                                                  r'extinction value, calculated as: '
                                                                  r'$\beta_{532}^{max}\cdot LR$, $LR=55sr$'}),
                           r_max=xr.Variable(dims=(), data=np.float(day_params_ds.rm.sel(Time=day_date).values),
                                             attrs={'long_name': r'$r_{max}$', 'units': r'$km$',
                                                    'info': r'A generation parameter. The top height of aerosol layer.'}),
                           params_source=xr.Variable(dims=(),
                                                     data=gen_utils.get_month_gen_params_path(station, day_date),
                                                     attrs={
                                                         'info': 'netcdf file name, containing generated density parameters.'}))

    aer_ds.attrs = {'info': 'Daily generated aerosol profiles',
                    'source_file': os.path.basename(__file__),
                    'location': station.location, }
    aer_ds.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    aer_ds.Wavelength.attrs = {'units': r'$\lambda$', 'units': r'$nm$'}
    aer_ds = aer_ds.transpose('Wavelength', 'Height', 'Time')
    aer_ds['date'] = day_date
    return aer_ds


# %% General Helper functions

def explore_gen_day(station, day_date, aer_ds, density_ds):
    # Show relative ratios between aerosols and molecular backscatter

    mol_month_folder = prep.get_month_folder_name(station.molecular_dataset, day_date)
    nc_mol = fr"{day_date.strftime('%Y_%m_%d')}_{station.location}_molecular.nc"
    mol_ds = prep.load_dataset(os.path.join(mol_month_folder, nc_mol))
    ratio_beta = aer_ds.beta / (mol_ds.beta + aer_ds.beta)
    ratio_beta.where(ratio_beta < 0.1).plot(x='Time', y='Height', row='Wavelength',
                                            cmap='turbo_r', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()

    ratio_sigma = aer_ds.sigma / (mol_ds.sigma + aer_ds.sigma)
    ratio_sigma.where(ratio_sigma < 0.1).plot(x='Time', y='Height', row='Wavelength',
                                              cmap='turbo_r', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()

    density_ds.density.plot(x='Time', y='Height', row='Component',
                            cmap='turbo', figsize=(10, 10), sharex=True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(TIMEFORMAT)
    ax.xaxis.set_tick_params(rotation=0)
    plt.show()



