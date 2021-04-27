import logging
from datetime import datetime
from itertools import repeat

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import learning_lidar.utils.global_settings as gs
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.generation.generate_density_utils import generate_density, generate_aerosol, explore_gen_day
from multiprocessing import Pool, cpu_count
import learning_lidar.generation.generate_density_utils as gen_den_utils
from learning_lidar.utils.utils import create_and_configer_logger


def generate_daily_aerosol_density(station, day_date, SAVE_DS=True):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param SAVE_DS:
    :return:
    """
    logger = logging.getLogger()
    logger.debug(f"Start generate_daily_aerosol_density for {station.name} on {day_date}")
    ds_day_params = gen_utils.get_daily_gen_param_ds(station=station, day_date=day_date,type='density_params')

    # Generate Daily Aerosols' Density
    density_ds = generate_density(station=station, day_date=day_date, day_params_ds=ds_day_params)

    # Generate Daily Aerosols' Optical Density
    aer_ds = generate_aerosol(station=station, day_date=day_date, day_params_ds=ds_day_params,
                              density_ds=density_ds)

    # TODO: add שמ option of 'size_optim' to optimize size from float64 to float32.
    #  example:  rho32= density_ds.rho.astype('float32',casting='same_kind')

    # Save the aerosols dataset
    if SAVE_DS:
        gen_utils.save_generated_dataset(station, aer_ds, data_source='aerosol', save_mode='single')
        gen_utils.save_generated_dataset(station, density_ds, data_source='density', save_mode='single')

    return aer_ds, density_ds


if __name__ == '__main__':
    gen_den_utils.PLOT_RESULTS = False
    gen_utils.set_visualization_settings()
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = create_and_configer_logger('generate_density.log', level=logging.DEBUG)
    station = gs.Station(station_name='haifa')

    start_date = datetime(2017,10,1)
    end_date = datetime(2017,10,31)
    days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    num_days = len(days_list)
    num_processes = min((cpu_count() - 1, num_days))
    with Pool(num_processes) as p:
        p.starmap(generate_daily_aerosol_density, zip(repeat(station), days_list))

    RUN_NEXT = False
    if RUN_NEXT:
        for cur_day in days_list: # TODO: isn't it reaping the above?
            aer_ds, density_ds = generate_daily_aerosol_density(station, day_date=cur_day)

            EXPLORE_GEN_DAY = False
            if EXPLORE_GEN_DAY:
                explore_gen_day(station, cur_day, aer_ds, density_ds)
