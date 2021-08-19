import json
import os
from datetime import datetime

import torch
from ray import tune

NUM_AVAILABLE_GPU = torch.cuda.device_count()
START_DATE = datetime(2017, 9, 1)
END_DATE = datetime(2017, 10, 31)


def get_paths(station_name, start_date, end_date):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    csv_base_name = f"gen_{station_name}_{start_date.strftime('%Y-%m-%d')}_" \
                    f"{end_date.strftime('%Y-%m-%d')}"
    train_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_train.csv')
    test_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_test.csv')
    stats_csv_path = os.path.join(base_path, 'data', "stats_" + csv_base_name + '_train.csv')
    results_path = os.path.join(base_path, 'results')  # TODO: save in D or E

    return train_csv_path, test_csv_path, stats_csv_path, results_path


train_csv_path, test_csv_path, stats_csv_path, RESULTS_PATH = get_paths(station_name='haifa',
                                                                        start_date=START_DATE,
                                                                        end_date=END_DATE)


# TODO - update dates to change between datasets


def update_params(config, consts):
    # Define X_features
    source_x = config['source']
    source_features = (f"{source_x}_path", "range_corr") \
        if (source_x == 'lidar' or source_x == 'signal') \
        else (f"{source_x}_path", "range_corr_p")
    mol_features = ("molecular_path", "attbsc")
    if config['use_bg']:
        bg_features = ("bg_path", "p_bg_r2") if config['use_bg'] == "range_corr" else ("bg_path", "p_bg")
        X_features = (source_features, mol_features, bg_features)
    else:
        X_features = (source_features, mol_features)

    # Update powers
    powers = consts['powers']
    use_power = config['use_power']
    if use_power and type(use_power) == str:
        power_in, power_out = eval(use_power)
        for yf, pow in zip(consts['Y_features'], power_out):
            powers[yf] = pow

        for xf, pow in zip(X_features, power_in):
            _, profile = xf
            powers[profile] = pow
        config.update({'power_in': str(power_in), 'power_out': str(power_out), 'use_power': True})

    return config, X_features, powers


# ######## RESUME EXPERIMENT #########
RESUME_EXP = False  # 'ERRORED_ONLY' # Can be "LOCAL" to continue experiment when it was disrupted
# (trials that were completed seem to continue training),
# or "ERRORED_ONLY" to reset and rerun ERRORED trials (not tested). Otherwise False to start a new experiment.
# Note: if fail_fast was 'True' in the the folder of 'EXP_NAME', then tune will not be able to load trials that didn't store any folder

EXP_NAME = None  # If 'resume' is not False, must enter experiment path.
# e.g. - "main_2021-05-19_21-50-40". Path is relative to RESULTS_PATH. Otherwise can keep it None.
# And it is generated automatically.


# ######## RESTORE or VALIDATE TRIAL PARAMS #########
experiment_dir = 'main_2021-05-19_23-17-25'
trial_dir = r'main_39b80_00000_0_bsize=32,dfilter=None,dnorm=True,fc_size=[16],hsizes=[3, 3, 3, 3],lr=0.001,' \
            r'ltype=MAELoss,source=lidar,use_bg=Tr_2021-05-19_23-17-25'
check_point_name = 'checkpoint_epoch=0-step=175'


def get_trial_params_and_checkpoint(experiment_dir, trial_dir, check_point_name):
    trial_params_path = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, 'params.json')
    with open(trial_params_path) as params_file:
        params = json.load(params_file)
    checkpoint = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, check_point_name)

    return checkpoint, params


# ######## VALIDATE TRIAL #########
VALIDATE_TRIAL = False  # TODO: What is the mode for validating experiment / trial?
if VALIDATE_TRIAL:
    PRETRAINED_MODEL_PATH, MODEL_PARAMS = get_trial_params_and_checkpoint(experiment_dir, trial_dir, check_point_name)

# ######## RESTORE TRIAL #########
RESTORE_TRIAL = False  # If true restores the given trial
if RESTORE_TRIAL:
    CHECKPOINT_PATH, TRIAL_PARAMS = get_trial_params_and_checkpoint(experiment_dir, trial_dir, check_point_name)
else:
    TRIAL_PARAMS = None
    CHECKPOINT_PATH = None

# Constants - should correspond to data, dataloader and model
CONSTS = {
    'max_epochs': 12,
    'max_steps': None,
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
}

# Note, replace tune.choice with grid_search if want all possible combinations
RAY_HYPER_PARAMS = {
    "hsizes": tune.grid_search(['[4, 4, 4, 4]', '[5, 5, 5, 5]']),  # '[3, 3, 3, 3]',
    "fc_size": tune.grid_search(['[16]', '[32]']),  # '[4]','[1]'
    "lr": tune.grid_search([1 * 1e-3]),
    "bsize": tune.grid_search([32]),
    "ltype": tune.choice(['MAELoss']),  # , 'MSELoss']),  # ['MARELoss']
    "use_power": tune.grid_search([False, '([0.5,1,1], [0.5])', '([0.5,1,0.5], [0.5])']),
    # "([0.5, -0.11, 0.5], [0.5])"]),
    # UV : -0.27 , G: -0.263 , IR: -0.11
    "use_bg": tune.grid_search([False, True, 'range_corr']),
    # True - bg is relevant for 'lidar' case # TODO if lidar - bg T\F, if signal - bg F
    "source": tune.grid_search(['lidar']),  # , 'lidar','signal_p'
    'dfilter': tune.grid_search([None, ('wavelength', [355]), ('wavelength', [532]), ('wavelength', [1064])]),
    'dnorm': tune.grid_search([False]),  # data_norm True - only for the best results achieved.
}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 32,
    "ltype": 'MAELoss',  # loss_type
    "use_power": '[0.5, -0.25], [0.5]',
    "use_bg": True,
    "source": 'signal_p',
    "hsizes": '[3, 3, 3, 3]',  # hidden_sizes
    "fc_size": '[16]',
    'dfilter': None,  # data_filter
    'dnorm': True,  # data_norm
}
USE_RAY = True
DEBUG_RAY = False
