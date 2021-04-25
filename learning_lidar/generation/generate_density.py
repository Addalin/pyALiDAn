import logging
from datetime import datetime
from itertools import repeat

import matplotlib.pyplot as plt
import seaborn as sns
import learning_lidar.utils.global_settings as gs
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.generation.generate_density_utils import generate_density, generate_aerosol,explore_gen_day
from multiprocessing import Pool, cpu_count

from learning_lidar.utils.utils import create_and_configer_logger


def generate_daily_aerosol_density(station, day_date, SAVE_DS=True):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param SAVE_DS:
    :return:
    """
    logger.debug(f"Start generate_daily_aerosol_density for {station.name} on {day_date}")
    ds_day_params = gen_utils.get_daily_gen_param_ds(station=station, day_date=day_date)

    # Generate Daily Aerosols' Density
    density_ds = generate_density(station=station, day_date=day_date, day_params_ds=ds_day_params)

    # Generate Daily Aerosols' Optical Density
    aer_ds = generate_aerosol(station=station, day_date=day_date, day_params_ds=ds_day_params,
                              density_ds=density_ds)

    # Save the aerosols dataset
    if SAVE_DS:
        gen_utils.save_generated_dataset(station, aer_ds, data_source='aerosol', save_mode='single')
        gen_utils.save_generated_dataset(station, density_ds, data_source='density', save_mode='single')

    return aer_ds, density_ds


if __name__ == '__main__':
    gen_utils.set_visualization_settings()
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = create_and_configer_logger('generate_density.log', level=logging.DEBUG)
    station = gs.Station(station_name='haifa')

    days_list = [datetime(2017, 9, 3, 0, 0)]
    num_days = len(days_list)
    num_processes = min((cpu_count() - 1, num_days))
    # todo make this works correctly
    with Pool(num_processes) as p:
        p.starmap(generate_daily_aerosol_density, zip(repeat(station), days_list))

    for cur_day in days_list:
        # TODO: Parallel days creation + tqdm (if possible - to asses the progress)
        aer_ds, density_ds = generate_daily_aerosol_density(station, day_date=cur_day)

        EXPLORE_GEN_DAY = False
        if EXPLORE_GEN_DAY:
            explore_gen_day(station, cur_day, aer_ds, density_ds)
