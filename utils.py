import csv
import logging


def create_and_configer_logger(log_name='log_file.log', level = logging.DEBUG):
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
        format='\n'+'[%(asctime)s - %(levelname)s] {%(pathname)s:%(lineno)d} -'+'\n'+' %(message)s'+'\n',
        datefmt = '%Y-%m-%d %H:%M:%S'
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


#%% testing multiproccesing from: https://gist.github.com/morkrispil/3944242494e08de4643fd42a76cb37ee
#import multiprocessing as mp
import multiprocess as mp
from functools import partial
import pandas as pd
import numpy as np
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
	results = sorted(results, key=lambda x:x[0])
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



