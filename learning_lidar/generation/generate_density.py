import logging
import os
from itertools import repeat
from multiprocessing import Pool, cpu_count

import pandas as pd

import learning_lidar.generation.generate_density_utils as gen_den_utils
import learning_lidar.generation.generation_utils as gen_utils
from learning_lidar.utils import utils, vis_utils, global_settings as gs


# TODO:  add 2 flags - Debug and save figure.
def generate_daily_aerosol_density(station, day_date, save_ds):
    """
    TODO: add usage
    :param station:
    :param day_date:
    :param save_ds:
    :return:
    """
    logger = logging.getLogger()
    logger.debug(f"Start generate_daily_aerosol_density for {station.name} on {day_date}")
    ds_day_params = gen_utils.get_daily_gen_param_ds(station=station, day_date=day_date, type_='density_params')

    # Generate Daily Aerosols' Density
    density_ds = gen_den_utils.generate_density(station=station, day_date=day_date, day_params_ds=ds_day_params)

    # Generate Daily Aerosols' Optical Density
    aer_ds = gen_den_utils.generate_aerosol(station=station, day_date=day_date, day_params_ds=ds_day_params,
                                            density_ds=density_ds)

    # TODO: add option of 'size_optim' to optimize size from float64 to float32.
    #  example:  rho32= density_ds.rho.astype('float32',casting='same_kind')

    # Save the aerosols dataset
    if save_ds:
        gen_utils.save_generated_dataset(station, aer_ds, data_source='aerosol', save_mode='single')
        gen_utils.save_generated_dataset(station, density_ds, data_source='density', save_mode='single')

    return aer_ds, density_ds


def generate_density_main(params):
    vis_utils.set_visualization_settings()
    gen_utils.PLOT_RESULTS = params.plot_results
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = utils.create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
    logger.info(params)
    station = gs.Station(station_name=params.station_name)
    start_date, end_date = params.start_date, params.end_date
    days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    logger.info(f"\nStation name:{station.location}\nStart generating aerosols densities "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")
    num_days = len(days_list)
    num_processes = 1 if gen_den_utils.PLOT_RESULTS else min((cpu_count() - 1, num_days))

    with Pool(num_processes) as p:
        p.starmap(generate_daily_aerosol_density, zip(repeat(station), days_list, repeat(params.save_ds)))

    logger.info(f"\nDone generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")


if __name__ == '__main__':
    parser = utils.get_base_arguments()

    parser.add_argument('--plot_results', action='store_true',
                        help='Whether to plot graphs')

    parser.add_argument('--save_ds', action='store_true',
                        help='Whether to save the datasets')

    args = parser.parse_args()
    generate_density_main(args)
