import json
import os
import sys
from datetime import datetime

import torch
from ray import tune

from learning_lidar.utils import global_settings as gs

NUM_AVILABLE_CPU = os.cpu_count()
NUM_AVAILABLE_GPU = torch.cuda.device_count()
START_DATE = datetime(2017, 4, 1)
END_DATE = datetime(2017, 10, 31)
station_name = 'haifa'
station_name = station_name + '_remote' if (sys.platform in ['linux', 'ubuntu']) else station_name
station = gs.Station(station_name)


def get_paths(station: gs.Station, start_date: datetime, end_date: datetime):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    nn_source_data = station.nn_source_data
    csv_base_name = f"gen_{station.location.lower()}_{start_date.strftime('%Y-%m-%d')}_" \
                    f"{end_date.strftime('%Y-%m-%d')}"
    train_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_train.csv')
    test_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_test.csv')
    stats_csv_path = os.path.join(base_path, 'data', "stats_" + csv_base_name + '_train.csv')
    results_path = os.path.join(station.nn_output_results)  # TODO: save in D or E
    # TODO: Add exception in case paths are invalid
    return train_csv_path, test_csv_path, stats_csv_path, results_path, nn_source_data


train_csv_path, test_csv_path, stats_csv_path, RESULTS_PATH, nn_source_data = get_paths(station,
                                                                                        start_date=START_DATE,
                                                                                        end_date=END_DATE)


# TODO - update dates to change between datasets


def update_params(config, consts):
    # Define X_features
    source_x = config['source']
    lidar_features = (f"{source_x}_path", "range_corr") \
        if (source_x == 'lidar' or source_x == 'signal') \
        else (f"{source_x}_path", "range_corr_p")
    mol_features = ("molecular_path", "attbsc")
    if config['use_bg']:
        # TODO: add option to set operations on bg channel
        bg_features = ("bg_path", "p_bg_r2") if config['use_bg'] == "range_corr" else ("bg_path", "p_bg")
        X_features = (lidar_features, mol_features, bg_features)
    else:
        X_features = (lidar_features, mol_features)

    # Update powers
    powers = consts['powers']
    use_power = config['use_power']
    if use_power and type(use_power) == str:
        power_in, power_out = eval(use_power)
        for xf, pow_x in zip(X_features, power_in):
            _, profile = xf
            powers[profile] = pow_x

        for yf, pow_y in zip(consts['Y_features'], power_out):
            powers[yf] = pow_y

        config.update({'power_in': str(power_in), 'power_out': str(power_out), 'use_power': True})

    if config['dfilter']:
        dfilter = config['dfilter'].split(' ')
        dfilter[1] = eval(dfilter[1])
    else:
        dfilter = False

    return config, X_features, powers, dfilter


# ######## RESUME EXPERIMENT ######### ---> Make sure RESTORE_TRIAL = False
RESUME_EXP = False  # 'ERRORED_ONLY'  # False | True
# Can be "LOCAL" to continue experiment when it was disrupted
# (trials that were completed seem to continue training),
# or "ERRORED_ONLY" to reset and rerun ERRORED trials (not tested). Otherwise False to start a new experiment.
# Note: if fail_fast was 'True' in the the folder of 'EXP_NAME', then tune will not be able to load trials that didn't store any folder

EXP_NAME = None  # 'main_2022-02-13_19-10-30'  #
# If 'resume' is not False, must enter experiment path.
# e.g. - "main_2021-05-19_21-50-40". Path is relative to RESULTS_PATH. Otherwise can keep it None.
# And it is generated automatically.


# ######## RESTORE or VALIDATE TRIAL PARAMS #########
experiment_dir = 'main_2022-02-04_20-14-28'
trial_dir = r"C:\Users\addalin\Dropbox\Lidar\lidar_learning\results\main_2022-02-04_20-14-28\main_4b099_00000_0_bsize" \
            r"=32,dfilter=wavelength [355],dnorm=False,fc_size=[16],hsizes=[4,4,4,4]," \
            r"lr=0.002,ltype=MAELoss,opt_powers=T_2022-02-04_20-14-29"
check_point_name = 'checkpoint_epoch=999-step=999'


# TODO: Load Trainer chekpoint  https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html#restoring-training-state
# name of trainer : epoch=29-step=3539.ckpt
# found under C:...\main_2022-01-31_23-24-22\main_28520_00024_24_bsize=32,dfilter=('wavelength', [355]),dnorm=False,fc_size=[16],hsizes=[6, 6, 6, 6],lr=0.002,ltype=MAELoss,sou_2022-02-01_11-49-31\lightning_logs\version_0
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
    'max_epochs': 20,
    'max_steps': None,
    'num_workers': int(NUM_AVILABLE_CPU * 0.8),
    'train_csv_path': train_csv_path,
    'test_csv_path': test_csv_path,
    'stats_csv_path': stats_csv_path,
    'nn_source_data': nn_source_data,
    'powers': {'range_corr': 0.5, 'range_corr_p': 0.5, 'attbsc': 0.5,
               'p_bg': 0.5, 'p_bg_r2': 0.5,
               'LC': 1.0, 'LC_std': 1.0, 'r0': 1.0, 'r1': 1.0, 'dr': 1.0},
    'num_gpus': NUM_AVAILABLE_GPU,
    "top_height": 15.3,  # NOTE: CHANGING IT WILL AFFECT BOTH THE INPUT DIMENSIONS TO THE NET, AND THE STATS !!!
    "Y_features": ['LC'],
}

# Note, replace tune.choice with grid_search if want all possible combinations
RAY_HYPER_PARAMS = {
    "hsizes": tune.grid_search(['[6,6,6,6]', '[4,4,4,4]']),
    # Options: '[4,4,4,4]' | '[5,5,5,5]' | '[6, 6, 6, 6]' ...etc.
    "fc_size": tune.grid_search(['[16]']),  # Options: '[4]' | '[16]' | '[32]' ...etc.
    "lr": tune.grid_search([2 * 1e-3]),
    "bsize": tune.grid_search([64]),
    "ltype": tune.grid_search(['MAELoss']),  # Options: 'MAELoss' | 'MSELoss' | 'MARELoss'. See 'custom_losses.py'
    "use_power": tune.grid_search(['([0.5, -0.30, 0.5], [1.0])']),  # Options: False | '([0.5,1,1], [0.5])' ...etc.
    # UV : -0.27 , G: -0.263 , IR: -0.11
    "opt_powers": tune.grid_search([False]),  # Options: False | True
    "use_bg": tune.grid_search([True, 'range_corr', False]),
    # Options: False | True | 'range_corr'. Not relevant for 'signal' as source
    "source": tune.grid_search(['lidar']),  # Options: 'lidar'| 'signal_p' | 'signal'
    'dfilter': tune.grid_search(["wavelength [355]"]),  # Options: None | '(wavelength, [lambda])' - lambda=355,532,1064
    'dnorm': tune.grid_search([False]),  # Options: False | True
    'overfit': tune.grid_search([False]),  # Apply over fit mode of pytorch lightening. Note: Change bsize to 10
    'debug': tune.choice([False]),  # Apply debug mode of pytorch lightening
    'cbias': tune.grid_search([True]),  # Calc convolution biases. This may be redundant if using batch norm
    'wdecay': tune.choice([0])  # Weight decay algorithm to test l2 regularization of NN weights.
    # Apply l2 regularization on model weights. parameter weight_decay of Adam optimiser
    # afterwards
}

NON_RAY_HYPER_PARAMS = {
    "lr": 1 * 1e-3,
    "bsize": 32,
    "ltype": 'MAELoss',  # loss_type
    "use_power": '[0.5, 0.25], [1.0]',
    "use_bg": True,
    "source": 'signal_p',
    "hsizes": '[3, 3, 3, 3]',  # hidden_sizes
    "fc_size": '[16]',
    'dfilter': None,  # data_filter
    'dnorm': True,  # Data normalization
    'overfit': False,  # Apply over fit mode of pytorch lightening. Note: Change bsize to 10
    'debug': False,  # Apply debug mode of pytorch lightening
    'cbias': True,  # Calc convolution biases
    'wdecay': 0  # Weight decay algorithm to test l2 regularization of NN weights.
}
USE_RAY = True
DEBUG_RAY = False
