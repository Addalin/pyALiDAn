import glob
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune

from learning_lidar.utils import global_settings as gs, vis_utils

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"usepackage{amsmath}"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
mpl.rc('text', usetex=True)
mpl.ticker.ScalarFormatter(useMathText=True)


def extract_powers(row, in_channels):
    """
    # TODO: add usage
    :param row:
    :param in_channels:
    :return:
    """
    powers = eval(row['powers']) if type(row['powers']) == str else None
    pow_y = np.array(powers[1])[0] if type(powers) == tuple else None
    pow_x = np.array(powers[0]) if type(powers) == tuple else None
    pow_xi = np.zeros(in_channels)
    if type(pow_x) == np.ndarray:
        for chan in range(in_channels):
            pow_xi[chan] = pow_x[chan] if (len(pow_x) >= chan + 1) else None
    else:
        for chan in range(in_channels):
            pow_xi[chan] = None
    return [pow_y, *pow_xi]


def plot_pivot_table(pivot_table: pd.pivot_table, title: str, figsize=(7, 5), vis_title: bool = True,
                     y_label=r'Relative error $[\%]$', ylim=None, sci_view=False,
                     legend_col: int = 1, legend_loc: str = 'best', fig_txt: dict = None,
                     save_fig: bool = False, fig_path: os.path = os.path.join(gs.PKG_ROOT_DIR, 'figures'),
                     format_fig: str = 'png'):
    """
    # TODO: add usage
    :param vis_title:
    :param fig_txt:
    :param pivot_table:
    :param title:
    :param figsize:
    :param y_label:
    :param ylim:
    :param sci_view:
    :param legend_col:
    :param legend_loc:
    :param save_fig:
    :param fig_path:
    :param format_fig:
    :return:
    """
    if not pivot_table.empty:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        pivot_table.plot(kind='bar', ax=ax, title=title, rot=0). \
            legend(ncol=legend_col, title=pivot_table.columns.name, loc=legend_loc, framealpha=0.9)

        ax.title.set_visible(vis_title)
        if ylim:
            ax.set_ylim(ylim)
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible=True, which='minor', linewidth=0.6)  # , color='w'
        ax.grid(visible=True, which='major', linewidth=1.2)  # , color='w'
        ax.xaxis.grid(False)
        ax.set_ylabel(y_label)
        if sci_view:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if fig_txt:
            fig.text(fig_txt['locx'], fig_txt['locy'], fig_txt['str'], fontsize=14)
        plt.tight_layout()
        plt.show()

    else:
        print("No results to display!")
    plt.tight_layout()
    if save_fig:
        fpath = vis_utils.save_fig(fig, fig_name=title, folder_name=fig_path, format_fig=format_fig)
    else:
        fpath = ''
    return fig, ax, fpath


def generate_results_table(results_folder: str = os.path.join(gs.PKG_ROOT_DIR, 'results'),
                           experiments_table_fname: str = 'runs_board.xlsx',
                           dst_fname='total_results.csv'):
    """
    # TODO add usage
    :param results_folder:
    :param experiments_table_fname:
    :param dst_fname:
    :return:
    """
    table_fname = os.path.join(results_folder, experiments_table_fname)
    runs_df = pd.read_excel(table_fname)
    runs_df = runs_df[runs_df.include == True][runs_df.state != 'PENDING']
    print(runs_df)

    for idx, row in runs_df.iterrows():
        try:
            state_fname = sorted(glob.glob(os.path.join(row.experiment_folder, r'experiment_state*.json')))[-1]
            analysis = tune.ExperimentAnalysis(state_fname)
            ignore_MARELoss = "MARELoss" in [row.field_to_ignore]
            analysis.default_metric = "MARELoss" if ignore_MARELoss else "MARELoss"
            analysis.default_mode = "min"
            results_df = analysis.dataframe(metric="MARELoss", mode="min", )

            # update fields:
            if ignore_MARELoss:
                results_df["MARELoss"] = None

            # rename column names:
            cols = results_df.columns.values.tolist()
            new_cols = [col.replace('config/', "") for col in cols]
            dict_cols = {}
            for col, new_col in zip(cols, new_cols):
                dict_cols.update({col: new_col})
            results_df = results_df.rename(columns=dict_cols)

            # update power values:
            # TODO: incase Adi Vainiger: use_bg : TRUE, range_corr and pow_x3 is not given --> use the default values to 0.5
            len_pow = len(results_df[results_df.use_power == True])
            len_no_pow = len(results_df[results_df.use_power == False])
            len_pows = len(results_df[results_df.use_power != False])
            if len_no_pow > 0:
                results_df.loc[results_df[results_df.use_power == False].index, 'powers'] = ''
            if len_pows != len_pow:
                results_df.loc[results_df[results_df.use_power != False].index, 'powers'] = results_df.use_power
                results_df.loc[results_df[results_df.use_power != False].index, 'use_power'] = True
            else:
                results_df.loc[results_df[results_df.use_power == True].index, 'powers'] = '([0.5,0.5],[0.5])'

            # Update source type of database, i.e., extended/initial with/without overlap
            results_df['overlap'] = row.overlap
            results_df['db'] = row.database
            # note = row['note']
            # row.note.str.__contains__('overlap')
            # results_df['note']= note if type(note)==str else 'ok'

            # Drop irrelevant columns:
            drop_cols = ['time_this_iter_s', 'should_checkpoint', 'done',
                         'timesteps_total', 'episodes_total',
                         'experiment_id', 'timestamp', 'pid', 'hostname',
                         'node_ip', 'time_since_restore', 'timesteps_since_restore',
                         'iterations_since_restore']
            results_df.drop(columns=drop_cols, inplace=True)

            # reorganize columns:
            if 'opt_powers' not in results_df.keys():
                results_df['opt_powers'] = False

            new_order = ['trial_id', 'date', 'time_total_s', 'training_iteration',
                         'loss', 'MARELoss',
                         'bsize', 'dfilter', 'dnorm', 'fc_size', 'hsizes', 'lr',
                         'ltype', 'source', 'use_bg',
                         'use_power', 'opt_powers', 'powers',
                         'db', 'overlap', 'logdir']  # 'note'
            results_df = results_df.reindex(columns=new_order)

            # Remove irrelevant trials (e.g. when dnorm had wrong calculation)
            if row.trial_to_ignore is not np.nan:
                key, cond = eval(row.trial_to_ignore)
                results_df.drop(index=results_df[results_df[key] == cond].index, inplace=True)

            # Save csv
            results_csv = os.path.join(row.experiment_folder, f'experiment_results.csv')
            results_df.to_csv(results_csv, index=False)

            # Update csv path in main runs_board
            runs_df.loc[idx, 'results_csv'] = results_csv
            print(results_csv, idx)

        except:
            continue

    # Concatenate all csv files with include = 1 (TRUE)

    paths = [row['results_csv'] for idx, row in runs_df.iterrows() if not (pd.isnull(row['results_csv'])) == True]
    results_dfs = [pd.read_csv(path) for path in paths]
    total_results = pd.concat(results_dfs, ignore_index=True)
    total_results['fc_size'] = total_results.fc_size.apply(lambda x: eval(str(x))[0])

    # update powers values
    in_channels = 3
    res = total_results.apply(extract_powers, args=(in_channels,), axis=1, result_type='expand')
    cols_powx = [f"pow_x{ind + 1}" for ind in range(in_channels)]
    res.rename(columns={0: 'pow_y', 1: cols_powx[0], 2: cols_powx[1], 3: cols_powx[2]}, inplace=True)
    total_results[res.columns.values] = res
    total_results['powers'] = total_results.powers.apply(lambda x: eval(x) if type(x) == str else None)
    hsizes = total_results.hsizes.apply(lambda x: eval(x))
    total_results['u_hsize'] = hsizes.apply(lambda x: all(
        [(hi == x[0]) for hi in x]))  # The test of changing the with at the last level , didn't show improvements

    # Specifying column of wavelength usage
    wavelengths = []
    filtered = total_results.dfilter.apply(lambda x: type(x) == str)
    # inds = total_results.dfilter[filtered].index
    for ind, f in enumerate(filtered):
        if f:
            [filter_by, filter_values] = total_results.dfilter.iloc[ind].split(' ')
            filter_values = eval(filter_values)
            if filter_by == 'wavelength':
                wavelength = tuple(filter_values) if len(filter_values) > 1 else filter_values[0]
            else:
                wavelength = 'all'
        else:
            wavelength = 'all'
        wavelengths.append(wavelength)

    total_results['wavelength'] = wavelengths

    # Specify config name
    configs = []
    for idx, row in total_results.iterrows():
        hsize = eval(row.hsizes)[0]
        fcsize = row.fc_size
        u_hsize = row.u_hsize
        if (hsize == 4) and (fcsize == 16) and u_hsize:
            configs.append('A')
        elif (hsize == 4) and (fcsize == 32) and u_hsize:
            configs.append('B')
        elif (hsize == 5) and (fcsize == 16) and u_hsize:
            configs.append('C')
        elif (hsize == 5) and (fcsize == 32) and u_hsize:
            configs.append('D')
        elif (hsize == 6) and (fcsize == 16) and u_hsize:
            configs.append('E')
        elif (hsize == 6) and (fcsize == 32) and u_hsize:
            configs.append('F')
        else:
            configs.append('Other')

    total_results['config'] = configs

    # save
    # TODO: save runs_df with results_csv paths
    current_res_fname = os.path.join(results_folder, dst_fname)
    total_results.to_csv(current_res_fname)
    print('Saving total results to: ', current_res_fname)

    return total_results


if __name__ == '__main__':
    results_folder = os.path.join(gs.PKG_ROOT_DIR, 'results')
    experiments_table_fname = 'runs_board.xlsx'
    dst_fname = 'total_results.csv'
    dst_fname = 'remote_' + dst_fname if (sys.platform in ['linux', 'ubuntu']) else dst_fname
    generate_results_table(results_folder, experiments_table_fname, dst_fname=dst_fname)
