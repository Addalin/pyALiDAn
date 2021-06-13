import os
from datetime import datetime, timedelta
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat

# Local modules
import learning_lidar.utils.global_settings as gs
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.generation.daily_signals_generations_utils import calc_total_optical_density, \
    calc_lidar_signal, calc_daily_measurement
import learning_lidar.generation.daily_signals_generations_utils as gen_sig_utils
from learning_lidar.utils.utils import create_and_configer_logger
import pandas as pd

# TODO:  add 2 flags - Debug and save figure.

def generate_daily_lidar_measurement(station, day_date, SAVE_DS=True):
    ds_total = calc_total_optical_density(station=station, day_date=day_date)
    signal_ds = calc_lidar_signal(station, day_date, ds_total)
    measure_ds = calc_daily_measurement(station, day_date, signal_ds)

    if SAVE_DS:
        # TODO: check that the range_corr_p is added to measure_ds, and that the LCNET is uploading the new paths
        #  (especially if range_corr_p )  . and if so, save only 2 single files of measure_ds, and signal_ds to save
        #  time and space
        # NOTE: saving to separated datasets (for the use of the learning phase), is done in dataseting.prepare_generated_samples()
        gen_utils.save_generated_dataset(station, measure_ds,
                                         data_source='lidar',
                                         save_mode='single')

        gen_utils.save_generated_dataset(station, signal_ds,
                                         data_source='signal',
                                         save_mode='single')

    return measure_ds, signal_ds


def daily_signals_generation_main(station_name='haifa', start_date=datetime(2017, 9, 1), end_date=datetime(2017, 9, 2)):
    gs.set_visualization_settings()
    gen_sig_utils.PLOT_RESULTS = True  # Toggle True for debug. False for run.
    # TODO: Toggle PLOT_RESULTS to True - doesn't seem to work
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
    station = gs.Station(station_name=station_name)

    logger.info(f"\nStation name:{station.location}\nStart generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
    days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    num_days = len(days_list)
    num_processes = 1 if gen_sig_utils.PLOT_RESULTS else min((cpu_count() - 1, num_days))
    with Pool(num_processes) as p:
        p.starmap(generate_daily_lidar_measurement, zip(repeat(station), days_list, repeat(False)))

    logger.info(f"\nDone generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")


if __name__ == '__main__':
    station_name = 'haifa'
    start_date = datetime(2017, 4, 1)
    end_date = datetime(2017, 4, 2)
    daily_signals_generation_main(station_name, start_date, end_date)
