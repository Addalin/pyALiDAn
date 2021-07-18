import os
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

import learning_lidar.utils.global_settings as gs
import learning_lidar.generation.generation_utils as gen_utils
import learning_lidar.utils.vis_utils as vis_utils
import learning_lidar.generation.daily_signals_generations_utils as gen_sig_utils
from learning_lidar.utils.utils import create_and_configer_logger


# TODO:  add 2 flags - Debug and save figure.

def generate_daily_lidar_measurement(station, day_date, SAVE_DS=True):
    ds_total = gen_sig_utils.calc_total_optical_density(station=station, day_date=day_date)
    signal_ds = gen_sig_utils.calc_lidar_signal(station, day_date, ds_total)
    measure_ds = gen_sig_utils.calc_daily_measurement(station, day_date, signal_ds)

    if SAVE_DS:
        # TODO: check that the range_corr_p is added to measure_ds, and that the LCNET is uploading the new paths
        #  (especially if range_corr_p )  . and if so, save only 2 single files of measure_ds, and signal_ds to save
        #  time and space
        # NOTE: saving to separated datasets (for the use of the learning phase),
        # is done in dataseting.prepare_generated_samples()
        gen_utils.save_generated_dataset(station, measure_ds,
                                         data_source='lidar',
                                         save_mode='single')

        gen_utils.save_generated_dataset(station, signal_ds,
                                         data_source='signal',
                                         save_mode='single')

    return measure_ds, signal_ds


def generate_daily_lidar_measurement2(station, day_date, SAVE_DS=True):
    nc_path = os.path.join(station.gen_lidar_dataset, str(day_date.year), f"{day_date.strftime('%m')}", f"{day_date.strftime('%Y_%m_%d')}_Haifa_generated_lidar.nc")
    overlap_params = pd.read_csv("../../data/overlap_params.csv", index_col=0)
    overlap_params_index = {4: 0, 5: 1, 9: 2, 10: 3}[day_date.month]
    measure_ds = gen_sig_utils.calc_daily_measurement_withoverlap(station, day_date,
                                                                  overlap_params=overlap_params.loc[
                                                                                 overlap_params_index, :].values,
                                                                  signal_ds=None, nc_path=nc_path)

    if SAVE_DS:
        # TODO: check that the range_corr_p is added to measure_ds, and that the LCNET is uploading the new paths
        #  (especially if range_corr_p )  . and if so, save only 2 single files of measure_ds, and signal_ds to save
        #  time and space
        # NOTE: saving to separated datasets (for the use of the learning phase),
        # is done in dataseting.prepare_generated_samples()
        gen_utils.save_generated_dataset(station, measure_ds,
                                         data_source='lidar',
                                         save_mode='single')

    return measure_ds


def daily_signals_generation_main(station_name='haifa', start_date=datetime(2017, 9, 1), end_date=datetime(2017, 9, 2)):
    vis_utils.set_visualization_settings()
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
    save_ds = True
    generate_daily_lidar_measurement2(station, days_list[0], save_ds)
    # TODO merge generate_daily_lidar_measurement with generate_daily_lidar_measurement2
    with Pool(num_processes) as p:
        p.starmap(generate_daily_lidar_measurement2, zip(repeat(station), days_list, repeat(save_ds)))

    logger.info(f"\nDone generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")


if __name__ == '__main__':
    station_name = 'haifa_shubi'
    start_date = datetime(2017, 9, 1)
    end_date = datetime(2017, 9, 2)
    daily_signals_generation_main(station_name, start_date, end_date)
