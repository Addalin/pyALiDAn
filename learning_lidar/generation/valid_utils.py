from __future__ import division

import os
import pickle
import platform
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import xarray as xr
from matplotlib import colors
from matplotlib import dates as mdates
from numpy import random
from pytictoc import TicToc
from scipy.spatial.distance import pdist, cdist
from scipy.stats import genextreme
from scipy.stats import kstwobign, pearsonr
from sklearn.mixture import BayesianGaussianMixture as BayesGMM

import learning_lidar.preprocessing.preprocessing_utils as prep_utils
from learning_lidar.utils import xr_utils, vis_utils, proc_utils, global_settings as gs

warnings.filterwarnings("ignore", '.*Index.ravel returning ndarray is deprecated.*', )

""" Plotting functions & settings"""

vis_utils.set_visualization_settings()
TIMEFORMAT = mdates.DateFormatter(r'%H')


def plot_daily_profile_for_publish(profile_ds, height_slice=None, figsize=(15, 5), fname: str = None,
                                   save_fig=False, threshold=None, cbar_text=None, show_title=True,
                                   folder_name: os.path = None, format_fig: str = 'png'):
    wavelengths = profile_ds.Wavelength.values
    if height_slice is None:
        height_slice = slice(profile_ds.Height[0].values, profile_ds.Height[-1].values)
    str_date = profile_ds.Time[0].dt.strftime("%Y-%m-%d").values.tolist()
    ncols = wavelengths.size
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
    if ncols > 1:
        for col_index, (wavelength, ax) in enumerate(zip(wavelengths, axes.ravel())):
            ds = profile_ds.sel(Height=height_slice, Wavelength=wavelength)
            # try:
            #     ds.attrs['units'] = ds.attrs['units'].replace("km", r"\textbf{km}")
            # except Exception as e:
            #     print(e)
            ds.Height.attrs['units'] = ds.Height.attrs['units'].replace("km", r"\textbf{km}")

            if threshold is None:
                threshold = ds.max()
                if not (isinstance(threshold, int) or isinstance(threshold, float)):
                    threshold = threshold.values
                    # print(threshold)
            ds = xr.DataArray.clip(ds, max=threshold, keep_attrs=True)
            # ds = ds.where(ds<threshold)
            if col_index == 0:
                cbar_ax = fig.add_axes([.91, .15, .03, .7])

            ds.plot(cmap='turbo', ax=ax, vmin=0, vmax=threshold, cbar_ax=cbar_ax, zorder=-20)
            ax.set_rasterization_zorder(-10)
            ax.xaxis.set_major_formatter(TIMEFORMAT)
            ax.xaxis.set_tick_params(rotation=0)
            if not show_title:
                ax.set_title("")
            ax.set_ylabel(ax.get_ylabel().replace("Height", r'\textbf{Height}'))
            ax.set_xlabel(ax.get_xlabel().replace("Time", r'\textbf{Time}'))
            for tick in ax.xaxis.get_majorticklabels():
                tick.set_horizontalalignment("left")
            if col_index != 0:
                ax.yaxis.label.set_visible(False)
    else:
        ax = axes
        ds = profile_ds.sel(Height=height_slice)
        # try:
        #     ds.attrs['units'] = ds.attrs['units'].replace("km", r"\textbf{km}")
        # except Exception as e:
        #     print(e)
        ds.Height.attrs['units'] = ds.Height.attrs['units'].replace("km", r"\textbf{km}")
        cbar_ax = fig.add_axes([.87, .15, .04, .7])
        if threshold is None:
            threshold = ds.max()
            if not (isinstance(threshold, int) or isinstance(threshold, float)):
                threshold = threshold.values
                # print(threshold)
                ds = xr.DataArray.clip(ds, max=threshold, keep_attrs=True)

        ds.plot(cmap='turbo', ax=ax, vmin=0, vmax=threshold, cbar_ax=cbar_ax, zorder=-20)
        ax.set_rasterization_zorder(-10)
        ax.xaxis.set_major_formatter(TIMEFORMAT)
        ax.xaxis.set_tick_params(rotation=0)
        if not show_title:
            ax.set_title("")
        ax.set_ylabel(ax.get_ylabel().replace("Height", r'\textbf{Height}'))
        ax.set_xlabel(ax.get_xlabel().replace("Time", r'\textbf{Time}'))
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")

    if cbar_text:
        cbar_ax.set_ylabel(cbar_text)

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=True)
    if ncols > 1:
        fig.tight_layout(rect=[0, 0, .9, 1])
    else:
        fig.tight_layout(rect=[0, 0, .86, 1])
    if save_fig:
        if folder_name is None:
            folder_name = os.path.join(os.path.abspath(os.path.curdir), 'figures')
        if fname is None:
            fname = f"{profile_ds.info} - {str_date}" if ncols > 1 \
                else f"{profile_ds.info} - {str_date} - {wavelengths.item()}"
        fig_path = vis_utils.save_fig(fig, fig_name=fname,
                                      folder_name=folder_name,
                                      format_fig=format_fig)
    else:
        fig_path = None
    plt.show()
    return fig_path, fig, axes


def plot_regression(source_vals, pred_vals, title_source: str = 'source', title_pred: str = 'predicted',
                    figsize=(8, 5), fname: str = 'Linear regression scatter plot', folder_name: os.path = None,
                    min_val: float = None, max_val: float = None, hist_bins=200,
                    save_fig: bool = False, format_fig: str = 'png', context: str = 'poster'):
    sns.set_style("whitegrid", {"grid.color": ".6"})
    sns.set_context(context)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    slope, intercept, r_value, p_value, std_err = ss.linregress(source_vals, pred_vals)

    nrmse = np.linalg.norm((source_vals - pred_vals)) / np.linalg.norm((source_vals))
    plot_text = fr"$\rm R^2$" + f":{r_value ** 2:.3f}\n" \
                                f"Slope: {slope:.3f}\n" \
                                f"Offset: {intercept:.1e} \n" \
                                f"NRMSE: {nrmse:.3f}\n" \
                                f"Observations: {len(source_vals)}"
    # f"std err: {std_err:.2e}\n" \
    x_vals = np.unique(source_vals)
    line = slope * x_vals + intercept

    ax.plot(x_vals, line, color='magenta', linestyle='dashed', label=plot_text)

    counts, xedges, yedges, im = ax.hist2d(source_vals, pred_vals, bins=hist_bins, cmap='turbo', cmin=1,
                                           norm=colors.LogNorm())
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('counts')
    plt.legend()
    # ax.axis('equal')
    min_val = min(min(source_vals), min(pred_vals)) if min_val is None else min_val
    max_val = max(max(source_vals), max(pred_vals)) if max_val is None else max_val

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel(title_source)
    ax.set_ylabel(title_pred)
    ax.set_axisbelow(True)
    ax.grid(color='gray', linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    if save_fig:
        if folder_name is None:
            folder_name = os.path.join(os.path.abspath(os.path.curdir), 'figures')
        if fname is None:
            fname = title_pred
        fig_path = vis_utils.save_fig(fig, fig_name=fname,
                                      folder_name=folder_name,
                                      format_fig=format_fig)
    else:
        fig_path = None

    plt.show()
    return fig_path, fig, ax


""" GMM helper functions"""


def create_or_copy_model(model_path: Path = None, gmm_params: dict = None) -> BayesGMM:
    if os.path.isfile(model_path):
        print(f"Loading trained gmm model from {model_path}")
        with open(model_path, 'rb') as f:
            gmm_model = pickle.load(f)
            gmm_model.warm_start = True  # 'make sure use previous fit'
    else:
        print(f"{model_path} does not exist. Initializing new model")
        gmm_model = BayesGMM()

    if gmm_params:
        print(f"Updating parameters: {gmm_params}")
        gmm_model = update_attributes(gmm_model, gmm_params)

    print(gmm_model)
    return gmm_model


def update_attributes(model: BayesGMM, attributes: dict) -> BayesGMM:
    """
    Update gmm attributes
    """

    for attr, value in attributes.items():
        if hasattr(model, attr):
            setattr(model, attr, value)
        else:
            print(f"GMM doesn't have attribute: {attr}")
    if hasattr(model, 'n_iter_'):
        assert not (model.n_iter_ > 0 and model.n_init > 1), 'if already trained, n init does not have any effect!'
    return model


def fit_gmm(model: BayesGMM, data: np.ndarray) -> float:
    print('---- Start GMM fit -----')
    tic_fit = TicToc()
    tic_fit.tic()
    model.fit(data)
    tic_fit.toc()
    return tic_fit.tocvalue()  # Run time


def save_model(model: BayesGMM, save_path: Union[str, Path]):
    print(f"Saved model to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def get_model_name(models_folder: os.path,
                   model_base_name: str = 'sklearn_GMM_Model',
                   model_name_end=None, ) -> Path:
    """
    If model_name_end is None - defaults to current time (MMHH_DDMM)
    """
    if not model_name_end:
        model_name_end = datetime.now().strftime('%H%M_%d%m')
    model_fname = os.path.join(models_folder, f"{model_base_name}_{model_name_end}.pkl")
    print(f"Model name (get or create): {model_fname}")
    return model_fname


def save_params_to_csv(save_params: dict, csv_models: os.path = 'models.csv'):
    table = pd.read_csv(csv_models)
    # get keys of the table
    keys = table.keys()
    model_unique_name = save_params['start_time']
    if model_unique_name not in table.start_time.values:
        # set a row according to the keys
        row = pd.DataFrame.from_dict(data=save_params, orient='index').T.reindex(columns=keys)
        # append a row to the csv
        row.to_csv(csv_models, mode='a', index=False, header=False)
        print(f"Appending new row of model name {model_unique_name}")
        saved_model = True
    else:
        # skipp appending a row to the csv
        print(f"The model with name {model_unique_name} already exists in the table.")
        saved_model = False
    return saved_model


def run(model_path: Path, gmm_params: dict, db_name: str, samples: np.ndarray):
    save_params = gmm_params.copy()
    save_params.update({'model_path': model_path,
                        'start_time': datetime.now().strftime('%H%M_%d%m'),
                        'db_name': db_name})
    gmm = create_or_copy_model(gmm_params=gmm_params, model_path=model_path)
    if hasattr(gmm, 'n_iter_'):
        prev_n_iter = gmm.n_iter_
    else:
        prev_n_iter = 0

    run_time = fit_gmm(gmm, samples)
    save_path = get_model_name(model_base_name='sklearn_GMM_Model', model_name_end=None)
    save_params.update({'converged': gmm.converged_,
                        'n_iter': gmm.n_iter_,
                        'warm_start': gmm.warm_start,
                        'run_time': round(run_time),
                        'machine': platform.node(),
                        'prev_n_iter': prev_n_iter,
                        'save_path': save_path})
    save_model(gmm, save_path)
    save_params_to_csv(save_params)


""" Pre-processing helper functions """


def daily_backscatter_from_profiles(day_date: datetime, station: gs.Station, wavelength: int = 532,
                                    df_calib: pd.DataFrame = None):
    # set daily_datset_df.
    # If df_calib is None, then generate a dataframe from TROPOS data at level1a
    if df_calib is None:
        profiles_paths = prep_utils.get_TROPOS_dataset_paths(station, day_date, file_type='profiles', level='level1a')
        profiles_paths.sort()
        format_filename = r"(.*)_(.*)_TROPOS_(.*)_(.*)_(.*)_profiles.nc"
        format_times = ["%Y_%m_%d", "%a", "%H_%M_%S", "%H%M", "%H%M"]
        time_stamps = [prep_utils.extract_date_time(path, format_filename, format_times) for path in profiles_paths]
        timestamps_df = pd.DataFrame(time_stamps, profiles_paths,
                                     columns=['dt_day', 'day', 'raw_time', 'start_time', 'end_time'])
        daily_datset_df = timestamps_df.apply(lambda row: [row['dt_day'].date(),
                                                           datetime.combine(date=row['dt_day'].date(),
                                                                            time=row['raw_time'].time()),
                                                           datetime.combine(date=row['dt_day'].date(),
                                                                            time=row['start_time'].time()),
                                                           datetime.combine(date=row['dt_day'].date(),
                                                                            time=row['end_time'].time())],
                                              axis=1, result_type='expand').set_axis(
            ['date', 'raw_time', 'start_time_period', 'end_time_period'], axis=1). \
            reset_index().rename(columns={'index': 'profile_path'}).sort_values(by='start_time_period', ascending=True,
                                                                                ignore_index=True)
    else:
        daily_datset_df = df_calib.loc[
            (df_calib['wavelength'] == wavelength) & (pd.to_datetime(df_calib['date']) == day_date)]. \
            sort_values(by='start_time_period', ascending=True, ignore_index=True)

    def _calc_mid_time(row):
        # Set time of profile to the center of the time slice
        dt_start = datetime.strptime(str(row['start_time_period']), '%Y-%m-%d %H:%M:%S')
        dt_end = datetime.strptime(str(row['end_time_period']), '%Y-%m-%d %H:%M:%S')
        dt_mid = dt_start + 0.5 * (dt_end - dt_start)
        round_seconds = timedelta(seconds=dt_mid.second % 30)  # round according to time resolution of TROPOS
        dt_mid += round_seconds
        return dt_mid

    daily_datset_df['mid_time_period'] = daily_datset_df.apply(lambda row: _calc_mid_time(row), axis=1,
                                                               result_type='expand')
    # display(daily_datset_df)

    ## % initialize dataset of daily_beta_chan (xr.DataSet)
    heightIndx = station.get_height_bins_values()
    timeIndx = station.calc_daily_time_index(day_date)
    height_units = 'km'

    daily_beta_chan = xr.Dataset(data_vars={'beta': (('Height', 'Time', 'Wavelength'),
                                                     np.empty((heightIndx.shape[0], timeIndx.shape[0], 1)))},
                                 coords={'Height': heightIndx, 'Time': timeIndx, 'Wavelength': [wavelength]})
    daily_beta_chan.beta.attrs = {'long_name': r'$\beta $',
                                  'units': r'$\rm \frac{1}{km \cdot sr}$',
                                  'info': "Daily estimated aerosol backscatter by PollyNet Processing Chain, Tropos",
                                  'source_data': 'level1a',
                                  }
    daily_beta_chan.attrs = {'location': station.location}

    ## set attributes of coordinates
    daily_beta_chan.Height.attrs = {'units': fr'$\rm {height_units}$',
                                    'info': 'Measurements heights above sea level'}
    daily_beta_chan.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$\rm nm$'}
    daily_beta_chan = daily_beta_chan.transpose('Wavelength', 'Height', 'Time')
    daily_beta_chan['date'] = day_date

    ## Loading beta profiles and save into  daily_beta_chan
    Pollynet_key = f'aerBsc_klett_{wavelength}'
    valids = []
    print(f"Loading data from {day_date.strftime('%Y-%m-%d')} at {wavelength}...")
    for profile_name, dt_mid in zip(daily_datset_df.profile_path, daily_datset_df.mid_time_period):
        # Load profile
        pollynet_da = xr_utils.load_dataset(profile_name)
        profile = pollynet_da[Pollynet_key].values
        if np.isnan(profile).all():
            valids.append(False)
            continue

        # print([profile is None])
        valids.append(True)
        daily_beta_chan.beta.loc[dict(Time=dt_mid)] = profile

    # Filter valid times
    daily_datset_df['valid'] = valids

    print(f"Done loading data.")

    # Convert from 1/(mr sr) to 1/(km sr)
    daily_beta_chan = prep_utils.convert_profiles_units(daily_beta_chan)
    return daily_beta_chan, daily_datset_df


def sample_grid_weights(XY_grid: np.ndarray,
                        weights: xr.DataArray,
                        n_iter=10, total_samples=10000):
    # estimate xy samples

    X = XY_grid[0]
    Y = XY_grid[1]
    weight_vector = weights.values.reshape(X.size)
    XY = np.vstack([X.reshape(X.size), Y.reshape(X.size)])
    sampels_per_iter = round(total_samples / n_iter)
    XY_s = []
    for n_samples in range(n_iter):
        # Sample indexes of (x,y) locations according to weight vector
        inds = random.choices(population=np.arange(weight_vector.shape[0]),  # list to pick from
                              weights=weight_vector,  # weights of the population, in order
                              k=sampels_per_iter,  # amount of samples to draw

                              )
        inds.sort()
        XY_s.append(XY[:, inds])

    XY_samples = np.concatenate(XY_s, axis=1)

    # Show scatter of samples
    plt.scatter(XY_samples[0], XY_samples[1], s=.1)
    plt.show()
    samples_df = pd.DataFrame(XY_samples.T, columns=['x', 'y'])
    return samples_df


def calc_2D_hist(XY_grid: np.array, samples_df: pd.DataFrame, weights: xr.DataArray, sample_source_name: str,
                 norm_hist: bool = False):
    X = XY_grid[0]
    Y = XY_grid[1]
    d_x = X[0, 1] - X[0, 0]
    d_y = Y[1, 0] - Y[0, 0]
    edges_x = X[0, :] + .5 * d_x
    edges_y = Y[:, 0] + .5 * d_y
    edge0_x = np.array(max(0, X[0, 0] - .5 * d_x)).reshape(1)
    edge0_y = np.array(max(0, Y[0, 0] - .5 * d_y)).reshape(1)
    edges_x = np.append(edge0_x, edges_x, axis=0)
    edges_y = np.append(edge0_y, edges_y, axis=0)
    range_x = [edges_x[0], edges_x[-1]]
    range_y = [edges_y[0], edges_y[-1]]

    H, yedges, xedges = np.histogram2d(samples_df.x, samples_df.y, bins=(edges_x, edges_y),
                                       range=[range_x, range_y], density=norm_hist)
    # print(H.sum(),[range_x, range_y])
    kernel_weights_da = xr.zeros_like(weights)

    H = H.T / (H.sum()) if norm_hist else H.T
    H = H.reshape(kernel_weights_da.data.shape)
    # print(kernel_weights_da.data.shape)
    # print(H.shape)
    kernel_weights_da.data = H
    kernel_weights_da.attrs = {'name': 'Norm Histogram', 'long_name': r'$\rho_s$',
                               'info': rf"2D normalized histogram of {sample_source_name}, sample number {len(samples_df)}"} if norm_hist \
        else {'name': 'Histogram', 'units': 'counts', 'long_name': r'$N_s$',
              'info': rf"2D histogram of {sample_source_name}, sample number {len(samples_df)}"}
    return kernel_weights_da, (H, yedges, xedges)


def calc_beta_from_gmm(gmm: BayesGMM, gmm_name: str, XY_grid: np.ndarray,
                       orig_beta: xr.DataArray,
                       timeIndx: list[datetime],
                       height_slice: slice, wavelength: int,
                       scale_type: str = 'min_max'):
    print('calculate log-likelihood of gmm for the input grid')
    tic_score = TicToc()
    tic_score.tic()
    X = XY_grid[0]
    Y = XY_grid[1]
    XY = np.vstack([X.reshape(X.size), Y.reshape(X.size)])
    z = gmm.score_samples(XY.T)  # Note that this method returns log-likelihood
    z = np.exp(z)  # e^x to get likelihood values
    Z = z.reshape(X.shape)  # reshaping to the grid shape of XY_grid, i.e. (XY_grid[1],XY_grid[2])
    tic_score.toc()

    if scale_type == 'min_max':
        beta_norm_factor = orig_beta.sel(Height=height_slice).max().item()
        beta_gmm_vals = beta_norm_factor * proc_utils.normalize(Z.copy())
    else:
        beta_gmm_vals = (Z.copy()) / Z.sum()
    beta_gmm_da = xr.zeros_like(orig_beta)
    beta_gmm_da.loc[dict(Wavelength=wavelength, Height=height_slice, Time=timeIndx)] = beta_gmm_vals
    beta_gmm_da.attrs = {'long_name': r'$\beta _{\rm GMM}$',
                         'units': r'$\rm \frac{1}{km \cdot sr}$',
                         'info': f"2D GMM representation of aerosol backscatter using {type(gmm)}, model name {gmm_name}."
                         }
    return beta_gmm_da


"""Kolmogorov-Smirnov test (2D)"""

__all__ = ['ks2d2s', 'estat', 'estat2d']


def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples.
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.

    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.

    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)

    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8

    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1 ** 2 + r2 ** 2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            # ix1 = random.choice(n, n1, replace=True)
            # ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist

    '''
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: random.randint(x, size=x)
    else:
        rand = random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n ** 2 - dy.sum() / m ** 2
    # z = ((n*m)/(n+m)) * z # ref. SR
    return z
