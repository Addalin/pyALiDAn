import os

import pandas as pd

from KDE_estimation_sample import kde_estimation_main
from generate_LC_pattern import generate_LC_pattern_main
from generate_density import generate_density_main
from daily_signals_generation import daily_signals_generation_main
from learning_lidar.utils import utils
from read_AERONET_data import read_aeronet_data_main

if __name__ == '__main__':
    parser = utils.get_base_arguments()

    parser.add_argument('--plot_results', action='store_true',
                        help='Whether to plot graphs')

    parser.add_argument('--save_ds', action='store_true',
                        help='Whether to save the datasets')

    # For KDE Estimation
    parser.add_argument('--extended_smoothing_bezier', action='store_true',
                        help='Whether to do extended smoothing bezier')

    # For daily signals generation
    parser.add_argument('--update_overlap_only', action='store_true',
                        help='Whether to update the overlap only or create from scratch')

    args = parser.parse_args()

    # ####### Ingredients generation #########
    # 1. Daily mean background signal
    # TODO Background generation (use generate_bg_signals.py)

    # 2. Daily Angstrom Exponent and Optical Depth
    for month_date in pd.date_range(start=args.start_date, end=args.end_date, freq='MS'):
        read_aeronet_data_main(args.station_name, month_date.month, month_date.year)

    # 3. Initial parameters for density generation
    HOME_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    DATA_DIR = os.path.join(HOME_DIR, 'data')

    # start_date and end_date should correspond to the extended csv!
    # months to run KDE on, one month at a time.
    for month_date in pd.date_range(args.start_date, args.end_date, freq='MS'):
        kde_estimation_main(args, month_date.month, month_date.year, DATA_DIR)

    # 4. Lidar Constant for a period
    generate_LC_pattern_main(args)

    # 5. Density Generation
    generate_density_main(args)

    # ####### Lidar Signal generation #######
    daily_signals_generation_main(args)
