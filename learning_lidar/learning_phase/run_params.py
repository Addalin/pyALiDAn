import os
from datetime import datetime

import torch
from ray import tune

NUM_AVAILABLE_GPU = torch.cuda.device_count()


def get_paths(station_name, start_date, end_date):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    csv_base_name = os.path.join(base_path, 'data', f"dataset_gen_{station_name}_"
                                                    f"{start_date.strftime('%Y-%m-%d')}_"
                                                    f"{end_date.strftime('%Y-%m-%d')}")
    train_csv_path = csv_base_name + '_train.csv'
    test_csv_path = csv_base_name + '_test.csv'

    results_path = os.path.join(base_path, 'results')

    return train_csv_path, test_csv_path, results_path


train_csv_path, test_csv_path, RESULTS_PATH = get_paths(station_name='haifa',
                                                        start_date=datetime(2017, 9, 1),
                                                        end_date=datetime(2017, 10, 31))

# Constants - should correspond to data, dataloader and model
CONSTS = {
    "fc_size": [512],
    'max_epochs': 2,
    'num_workers': 6,
    'train_csv_path': train_csv_path,
    'test_csv_path': test_csv_path,
    'powers': {'range_corr': 0.5, 'attbsc': 0.5, 'p_bg': 0.5,
               'LC': 0.5, 'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1},
    'num_gpus': NUM_AVAILABLE_GPU,
}

# Note, replace tune.choice with grid_search if want all possible combinations
RAY_HYPER_PARAMS = {
    "hidden_sizes": tune.choice([[16, 32, 8]]),  # TODO: add options of [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
    "lr": tune.grid_search([1e-3, 0.5 * 1e-3, 1e-4]),
    "bsize": tune.choice([32]),  # [16, 8]),
    "loss_type": tune.choice(['MSELoss', 'MAELoss']),  # ['MARELoss']
    "Y_features": tune.choice([['LC']]),
    # [['LC'], ['r0', 'r1', 'LC'], ['r0', 'r1'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']]
    "use_power": tune.grid_search([True, False]),
    "use_bg": tune.grid_search([False]),
    # True - bg is relevant for 'lidar' case # TODO if lidar - bg T\F, if signal - bg F
    "source": tune.grid_search(['signal', 'lidar']),
    'data_filter': tune.grid_search([('wavelength', [355]), None])
}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 8,
    "loss_type": 'MSELoss',
    "Y_features": ['LC'],
    "use_power": True,
    "use_bg": False,
    "source": 'signal',
    "hidden_sizes": [16, 32, 8],
    'data_filter': None,
}
