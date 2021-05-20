import os
from datetime import datetime

import torch
from ray import tune
import json
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


experiment_dir = 'main_2021-05-19_23-17-25'
trial_dir = r'main_39b80_00000_0_bsize=32,dfilter=None,dnorm=True,fc_size=[16],hsizes=[3, 3, 3, 3],lr=0.001,ltype=MAELoss,source=lidar,use_bg=Tr_2021-05-19_23-17-25'
check_point_path = 'checkpoint_epoch=0-step=175'

model_path = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, check_point_path)
params_path = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, 'params.json')

with open(params_path) as params_file:
    params = json.load(params_file)

# Constants - should correspond to data, dataloader and model
CONSTS = {
    'max_epochs': 4,
    'num_workers': 6,
    'train_csv_path': train_csv_path,
    'test_csv_path': test_csv_path,
    'stats_csv_path': stats_csv_path,
    'powers': {'range_corr': 0.5, 'range_corr_p': 0.5, 'attbsc': 0.5,
               'p_bg': 0.5, 'p_bg_r2': 0.5,
               'LC': 0.5, 'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1},
    'num_gpus': NUM_AVAILABLE_GPU,
    "top_height": 15.3,  # NOTE: CHANGING IT WILL AFFECT BOTH THE INPUT DIMENSIONS TO THE NET, AND THE STATS !!!
    "Y_features": ['LC'],
    'resume': False,  # Can be "LOCAL" to continue experiment when it was disrupted,
    # or "ERRORED_ONLY" to reset and rerun ERRORED trials. Otherwise False to start a new experiment.
    'name': None  # 'main_2021-05-19_22-35-50'  # If 'resume' is not False, must enter experiment path.
    # e.g. - "main_2021-05-19_21-50-40". Path is relative to RESULTS_PATH. Otherwise can keep it None.
}

# Note, replace tune.choice with grid_search if want all possible combinations
RAY_HYPER_PARAMS = {
    "hsizes": tune.grid_search(['[3, 3, 3, 3]', '[4, 4, 4, 4]', '[5, 5, 5, 5]']),  # '[1,1,1,1]','[2,2,2,2]',
    "fc_size": tune.grid_search(['[16]', '[32]']),  # '[4]',
    "lr": tune.choice([1 * 1e-3]),  # [1e-3, 0.5 * 1e-3, 1e-4]),
    "bsize": tune.choice([32]),  # 48 , 64]),  # [16, 8]),
    "ltype": tune.choice(['MAELoss']),  # , 'MSELoss']),  # ['MARELoss']
    # [['LC'], ['r0', 'r1', 'LC'], ['r0', 'r1'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']]
    "use_power": tune.grid_search([  # False,
        "([0.5, 0.5, 0.5], [0.5])",
        "([0.5,-0.2, 0.5], [0.5])",
        "([0.5,0.25, 0.5], [0.5])",
        "([0.5,-0.25, 0.5],[0.5])"]),
    # "([0.5, 0.25], [0.5])", "([0.5, 0.25], [1])",
    # "([0.5, -0.25], [0.5])", "([0.5, -0.25], [1])"]),
    "use_bg": tune.grid_search(['range_corr']),  #  False, True,'range_corr'
    # True - bg is relevant for 'lidar' case # TODO if lidar - bg T\F, if signal - bg F
    "source": tune.grid_search(['lidar']),  # , 'signal', 'signal_p'
    'dfilter': tune.grid_search([None]),  # , ('wavelength', [355])]), # data_filter
    'dnorm': tune.grid_search([False]),  # data_norm, , False, True
}

RESTORE_TRIAL_PARAMS = {
    "hsizes": tune.grid_search(['[3, 3, 3, 3]']),
    "fc_size": tune.grid_search(['[16]']),
    "lr": tune.choice([1 * 1e-3]),
    "bsize": tune.choice([32]),
    "ltype": tune.choice(['MAELoss']),
    "use_power": tune.grid_search(["([0.5,0.25, 1], [0.5])"]),
    "use_bg": tune.choice([False]),
    # True - bg is relevant for 'lidar' case # TODO if lidar - bg T\F, if signal - bg F
    "source": tune.grid_search(['lidar']),
    'dfilter': tune.grid_search([None]),
    'dnorm': tune.grid_search([False]),

}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 32,
    "ltype": 'MAELoss',  # loss_type
    "use_power": ([0.5, 0.25, 0.5], [0.5]),
    "use_bg": True,
    "source": 'signal',
    "hsizes": [3, 3, 3, 3],  # hidden_sizes
    "fc_size": [16],
    'dfilter': None,  # data_filter
    'dnorm': True,  # data_norm
}
USE_RAY = True
DEBUG_RAY = False
