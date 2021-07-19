import os
import logging
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

import learning_lidar.utils.global_settings as gs
import learning_lidar.generation.generation_utils as gen_utils
import learning_lidar.utils.vis_utils as vis_utils
import learning_lidar.utils.xr_utils as xr_utils
import learning_lidar.generation.daily_signals_generations_utils as gen_sig_utils
from learning_lidar.utils.utils import create_and_configer_logger, get_base_arguments


# TODO:  add 2 flags - Debug and save figure.

def generate_daily_lidar_measurement(station, day_date, save_ds=True, update_overlap_only=False):

    overlap_params = pd.read_csv("../../data/overlap_params.csv", index_col=0)
    try:
        # TODO Generalize. Currently adapted to 4 months only - (4,5,9,10)
        overlap_params_index = {4: 0, 5: 1, 9: 2, 10: 3}[day_date.month]
        overlap_param = overlap_params.loc[overlap_params_index, :].values
    except KeyError:
        raise KeyError(f"This month is not currently supported in overlap function.")

    if update_overlap_only:
        month_folder = xr_utils.get_month_folder_name(station.gen_lidar_dataset,  day_date)
        nc_path = os.path.join(month_folder, f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_lidar.nc")

        measure_ds = gen_sig_utils.calc_daily_measurement(station, day_date, overlap_params=overlap_param,
                                                          signal_ds=None, measure_ds_path=nc_path)

        if save_ds:
            # NOTE: saving to separated datasets (for the use of the learning phase),
            # is done in dataseting.prepare_generated_samples()
            gen_utils.save_generated_dataset(station, measure_ds, data_source='lidar', save_mode='single')

        return measure_ds

    else:
        ds_total = gen_sig_utils.calc_total_optical_density(station=station, day_date=day_date)
        signal_ds = gen_sig_utils.calc_lidar_signal(station, day_date, ds_total)
        measure_ds = gen_sig_utils.calc_daily_measurement(station=station, day_date=day_date, signal_ds=signal_ds,
                                                          overlap_params=overlap_param)

        if save_ds:
            # TODO: check that the range_corr_p is added to measure_ds, and that the LCNET is uploading the new paths
            #  (especially if range_corr_p )  . and if so, save only 2 single files of measure_ds, and signal_ds to save
            #  time and space
            # NOTE: saving to separated datasets (for the use of the learning phase),
            # is done in dataseting.prepare_generated_samples()
            gen_utils.save_generated_dataset(station, measure_ds, data_source='lidar', save_mode='single')
            gen_utils.save_generated_dataset(station, signal_ds, data_source='signal', save_mode='single')

        return measure_ds, signal_ds


def daily_signals_generation_main(params):

    vis_utils.set_visualization_settings()
    gen_sig_utils.PLOT_RESULTS = params.plot_results
    # TODO: Toggle PLOT_RESULTS to True - doesn't seem to work. Omer - works for me. Adi, please check again..
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
    logger.info(params)

    station = gs.Station(station_name=params.station_name)
    start_date = params.start_date
    end_date = params.end_date
    logger.info(f"\nStation name:{station.location}\nStart generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
    num_days = len(days_list)
    num_processes = 1 if gen_sig_utils.PLOT_RESULTS else min((cpu_count() - 1, num_days))

    with Pool(num_processes) as p:
        p.starmap(generate_daily_lidar_measurement, zip(repeat(station), days_list,
                                                        repeat(params.save_ds), repeat(params.update_overlap_only)))

    logger.info(f"\nDone generating lidar signals & measurements "
                f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")


if __name__ == '__main__':
    parser = get_base_arguments()
    parser.add_argument('--save_ds', action='store_true',
                        help='Whether to save the datasets')
    parser.add_argument('--update_overlap_only', action='store_true',
                        help='Whether to update the overlap only or create from scratch')
    parser.add_argument('--plot_results', action='store_true',
                        help='Whether to plot graphs')
    args = parser.parse_args()
    daily_signals_generation_main(args)
