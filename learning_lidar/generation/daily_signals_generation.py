import datetime
import logging
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
import xarray as xr

from learning_lidar.generation import generation_utils as gen_utils, daily_signals_generations_utils as gen_sig_utils
from learning_lidar.utils import utils, vis_utils, global_settings as gs


# TODO:  add 2 flags - Debug

class DailySignalGenerator:
    def __init__(self, station_name: gs.Station, save_ds: bool, plot_results: bool, logger: logging.Logger = None):
        station_name = station_name
        self.station = gs.Station(station_name=station_name)
        self.save_ds = save_ds
        self.plot_results = plot_results

        logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
        logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
        if logger:
            self.logger = logger
        else:
            self.logger = utils.create_and_configer_logger(os.path.join(gs.PKG_ROOT_DIR, "generation", "logs",
                                                                        f"{os.path.basename(__file__)}.log"),
                                                           level=logging.INFO)

        vis_utils.set_visualization_settings()

    def generate_daily_lidar_measurement(self, day_date: datetime.date) -> (xr.Dataset, xr.Dataset):
        total_ds = gen_sig_utils.calc_total_optical_density(station=self.station, day_date=day_date,
                                                            PLOT_RESULTS=self.plot_results)
        signal_ds = gen_sig_utils.calc_lidar_signal(self.station, day_date, total_ds,
                                                    PLOT_RESULTS=self.plot_results)
        measure_ds = gen_sig_utils.calc_daily_measurement(station=self.station, day_date=day_date, signal_ds=signal_ds,
                                                          PLOT_RESULTS=False, update_overlap_only=False)

        if self.save_ds:
            # TODO: check that the LCNET is uploading the new paths
            #  (especially if range_corr_p )  . and if so, save only 2 single files of measure_ds, and signal_ds to save
            #  time and space
            # NOTE: saving to separated datasets (for the use of the learning phase),
            # is done in dataseting.prepare_generated_samples() create_generated_dataset
            gen_utils.save_generated_dataset(self.station, measure_ds, data_source='lidar', save_mode='single')
            gen_utils.save_generated_dataset(self.station, signal_ds, data_source='signal', save_mode='single')

        return measure_ds, signal_ds

    def update_daily_lidar_measurement(self, day_date: datetime.date) -> xr.Dataset:
        """
        Updates a measure ds with overlap.

        :param day_date: datetime.date object of the required date
        :return: xr.Dataset with the generated measure ds
        """

        measure_ds = gen_sig_utils.calc_daily_measurement(self.station, day_date, signal_ds=None,
                                                          update_overlap_only=True, PLOT_RESULTS=False)

        if self.save_ds:
            # NOTE: saving to separated datasets (for the use of the learning phase),
            # is done in dataseting.prepare_generated_samples()
            gen_utils.save_generated_dataset(self.station, measure_ds, data_source='lidar', save_mode='single')

        return measure_ds

    def daily_signals_generation(self, start_date, end_date, update_overlap_only):
        self.logger.info(f"\nStation name:{self.station.location}\nStart generating lidar signals & measurements "
                         f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        days_list = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
        num_days = len(days_list)
        num_processes = 1 if self.plot_results else min((cpu_count() - 1, num_days))

        func = self.generate_daily_lidar_measurement if not update_overlap_only else self.update_daily_lidar_measurement
        with Pool(num_processes) as p:
            p.starmap(func, zip(days_list))

        self.logger.info(f"\nDone generating lidar signals & measurements "
                         f"for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")


if __name__ == '__main__':
    parser = utils.get_base_arguments()

    parser.add_argument('--update_overlap_only', action='store_true',
                        help='Whether to update the overlap only or create from scratch')

    args = parser.parse_args()

    logger = utils.create_and_configer_logger(os.path.join(gs.PKG_ROOT_DIR, "generation", "logs",
                                                           f"{os.path.basename(__file__)}.log"),
                                              level=args.log)
    logger.info(args)

    daily_signals_generator = DailySignalGenerator(station_name=args.station_name,
                                                   save_ds=args.save_ds, logger=logger, plot_results=args.plot_results)

    daily_signals_generator.daily_signals_generation(start_date=args.start_date, end_date=args.end_date,
                                                     update_overlap_only=args.update_overlap_only)
