import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import learning_lidar.global_settings as gs
from learning_lidar.generation.generate_density_utils import explore_gen_day, PLOT_RESULTS, \
    generate_aerosol, get_daily_gen_param_ds, generate_density
from learning_lidar.generation.generation_utils import save_generated_dataset
from learning_lidar.utils.utils import create_and_configer_logger

# TODO: move plot settings to a general utils, this is being used throughout the module
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
colors = ["darkblue", "darkgreen", "darkred"]
sns.set_palette(sns.color_palette(colors))
customPalette = sns.set_palette(sns.color_palette(colors))


def generate_aerosol_density(station, day_date, SAVE_DS=True):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param SAVE_DS:
    :return:
    """
    ds_day_params = get_daily_gen_param_ds(station=station, day_date=day_date)

    # Generate Daily Aerosols' Density
    density_ds = generate_density(station=station, day_date=day_date, day_params_ds=ds_day_params)

    # Generate Daily Aerosols' Optical Density
    aer_ds = generate_aerosol(station=station, day_date=day_date, day_params_ds=ds_day_params,
                              density_ds=density_ds)

    # Save the aerosols dataset
    if SAVE_DS:
        save_generated_dataset(station, aer_ds, data_source='aerosol', save_mode='single')
        save_generated_dataset(station, density_ds, data_source='density', save_mode='single')

    return aer_ds, density_ds


if __name__ == '__main__':
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = create_and_configer_logger('generate_density.log', level=logging.DEBUG)
    station = gs.Station(station_name='haifa')

    days_list = [datetime(2017, 9, 3, 0, 0)]
    for cur_day in days_list:
        # TODO: Parallel days creation + tqdm (if possible - to asses the progress)
        aer_ds, density_ds = generate_aerosol_density(station, day_date=cur_day)

        EXPLORE_GEN_DAY = False
        if EXPLORE_GEN_DAY:
            explore_gen_day(station, cur_day, aer_ds, density_ds)
