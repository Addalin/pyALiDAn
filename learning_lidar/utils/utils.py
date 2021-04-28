import csv
import logging
# %% testing multiproccesing from: https://gist.github.com/morkrispil/3944242494e08de4643fd42a76cb37ee
# import multiprocessing as mp
import multiprocess as mp
from functools import partial
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

def create_and_configer_logger(log_name='log_file.log', level=logging.DEBUG):
    """
    Sets up a logger that works across files.
    The logger prints to console, and to log_name log file. 
    
    Example usage:
        In main function:
            logger = create_and_configer_logger(log_name='myLog.log')

        Then in all other files:
            logger = logging.getLogger(__name__)
            
        To add records to log:
            logger.debug(f"New Log Message. Value of x is {x}")
    
    Args:
        log_name: str, log file name
        
    Returns: logger
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=level,
        format='\n' + '[%(asctime)s - %(levelname)s] {%(pathname)s:%(lineno)d} -' + '\n' + ' %(message)s' + '\n',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger


def _df_split(tup_arg, **kwargs):
    split_ind, df_split, df_f_name = tup_arg
    return (split_ind, getattr(df_split, df_f_name)(**kwargs))


def df_multi_core(df, df_f_name, subset=None, njobs=-1, **kwargs):
    if njobs == -1:
        njobs = mp.cpu_count()
    pool = mp.Pool(processes=njobs)

    try:
        df_sub = df[subset] if subset else df
        splits = np.array_split(df_sub, njobs)
    except ValueError:
        splits = np.array_split(df, njobs)

    pool_data = [(split_ind, df_split, df_f_name) for split_ind, df_split in enumerate(splits)]
    results = pool.map(partial(_df_split, **kwargs), pool_data)
    pool.close()
    pool.join()
    results = sorted(results, key=lambda x: x[0])
    results = pd.concat([split[1] for split in results])
    return results


def write_row_to_csv(csv_path, msg):
    """
    Writes msg to a new row in csv_path which is a csv file.
    Logs message in logger.
    
    Args:
        csv_path: str, path to the csv file. Creates it if file doesn't exist.
        msg: list, message to write to file. Each entry in list represents a column.
    """
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(msg)
    logger = logging.getLogger()
    logger.debug(f"Wrote to tracker{csv_path} message - {msg}")


def get_time_slice_dataset(dataset, start_time, end_time):
    sub_ds = dataset.sel(Time=slice(start_time, end_time))
    return sub_ds


def humanbytes(B):
    'Return the given bytes as a human friendly KB, MB, GB, or TB string'
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)


def visCurve(lData, rData, stitle=""):
    '''Visualize 2 curves '''

    fnt_size = 18
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 6))
    ax = axes.ravel()

    for (x_j, y_j) in zip(lData['x'], lData['y']):
        ax[0].plot(x_j, y_j)
    if lData.__contains__('legend'):
        ax[0].legend(lData['legend'], fontsize=fnt_size - 6)
    ax[0].set_xlabel(lData['lableX'], fontsize=fnt_size, fontweight='bold')
    ax[0].set_ylabel(lData['lableY'], fontsize=fnt_size, fontweight='bold')
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[0].set_title(lData['stitle'], fontsize=fnt_size, fontweight='bold')

    for (x_j, y_j) in zip(rData['x'], rData['y']):
        ax[1].plot(x_j, y_j)
    if rData.__contains__('legend'):
        ax[1].legend(rData['legend'], fontsize=fnt_size - 6)
    ax[1].set_xlabel(rData['lableX'], fontsize=fnt_size, fontweight='bold')
    ax[1].set_ylabel(rData['lableY'], fontsize=fnt_size, fontweight='bold')
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1].set_title(rData['stitle'], fontsize=fnt_size, fontweight='bold')

    fig.suptitle(stitle, fontsize=fnt_size + 4, va='top', fontweight='bold')
    fig.set_constrained_layout = True

    return [fig, axes]