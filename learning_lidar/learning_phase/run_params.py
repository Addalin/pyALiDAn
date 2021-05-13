import os
from datetime import datetime

import torch
from ray import tune

NUM_AVAILABLE_GPU = torch.cuda.device_count()


def get_paths(station_name, start_date, end_date):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    csv_base_name = f"gen_{station_name}_{start_date.strftime('%Y-%m-%d')}_" \
                    f"{end_date.strftime('%Y-%m-%d')}"
    train_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_train.csv')
    test_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_test.csv')
    stats_csv_path = os.path.join(base_path, 'data', "stats_" + csv_base_name + '.csv')
    results_path = os.path.join(base_path, 'results')  # TODO: save in D or E

    return train_csv_path, test_csv_path, stats_csv_path, results_path


train_csv_path, test_csv_path, stats_csv_path, RESULTS_PATH = get_paths(station_name='haifa',
                                                                        start_date=datetime(2017, 9, 1),
                                                                        end_date=datetime(2017, 10, 31))

# Constants - should correspond to data, dataloader and model
CONSTS = {
    'max_epochs': 4,
    'num_workers': 6,
    'train_csv_path': train_csv_path,
    'test_csv_path': test_csv_path,
    'stats_csv_path': stats_csv_path,
    'powers': {'range_corr': 0.5, 'range_corr_p': 0.5, 'attbsc': 0.5, 'p_bg': 0.5,
               'LC': 0.5, 'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1},
    'num_gpus': NUM_AVAILABLE_GPU,
    "top_height": 15.3,  # NOTE: CHANGING IT WILL AFFECT BOTH THE INPUT DIMENSIONS TO THE NET, AND THE STATS !!!
    "Y_features": ['LC'],
}

# Note, replace tune.choice with grid_search if want all possible combinations
RAY_HYPER_PARAMS = {
    "hsizes": tune.grid_search(['[3, 3, 3, 3]', '[4, 4, 4, 4]', '[5, 5, 5, 5]']),  # '[1,1,1,1]','[2,2,2,2]',
    "fc_size": tune.grid_search(['[16]', '[32]']),  # '[4]',
    "lr": tune.choice([1 * 1e-3]),  # [1e-3, 0.5 * 1e-3, 1e-4]),
    "bsize": tune.choice([32]),  # 48 , 64]),  # [16, 8]),
    "ltype": tune.choice(['MAELoss']),  # , 'MSELoss']),  # ['MARELoss']
    # [['LC'], ['r0', 'r1', 'LC'], ['r0', 'r1'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']]
    "use_power": tune.grid_search([False, "([0.5, 0.5], [0.5])", "([0.5, 0.5], [1])",
                                   "([0.5 ,0.5], [0.5])", "([0.5, 0.5], [1])",
                                   "([0.5, 0.25], [0.5])", "([0.5, 0.25], [1])",
                                   "([0.5, -0.25], [0.5])", "([0.5, -0.25], [1])"]),
    "use_bg": tune.grid_search([False]),  # , True]),
    # True - bg is relevant for 'lidar' case # TODO if lidar - bg T\F, if signal - bg F
    "source": tune.grid_search(['lidar', 'signal', 'signal_p']),
    'dfilter': tune.grid_search([None]),  # , ('wavelength', [355])]), # data_filter
    'dnorm': tune.grid_search([True, False]),  # data_norm, , False
}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 32,
    "ltype": 'MAELoss',  # loss_type
    "use_power": True,
    "use_bg": False,
    "source": 'signal',
    "hsizes": [2, 2, 2, 2],  # hidden_sizes
    "fc_size": [4],
    'dfilter': None,  # data_filter
    'dnorm': True,  # data_norm
}
USE_RAY = True
DEBUG_RAY = False
