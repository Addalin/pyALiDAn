import json
import os
import sys
from datetime import datetime
import yaml
import torch
from ray import tune
import glob
from learning_lidar.utils import global_settings as gs


# ######## HELPER FUNCTIONS ##########
def get_paths(station: gs.Station, start_date: datetime, end_date: datetime):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    nn_source_data = station.nn_source_data
    csv_base_name = f"gen_{station.location.lower()}_{start_date.strftime('%Y-%m-%d')}_" \
                    f"{end_date.strftime('%Y-%m-%d')}"
    train_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_train.csv')
    test_csv_path = os.path.join(base_path, 'data', "dataset_" + csv_base_name + '_test.csv')
    stats_csv_path = os.path.join(base_path, 'data', "stats_" + csv_base_name + '_train.csv')
    results_path = os.path.join(station.nn_output_results)  # TODO: Add exception in case paths are invalid
    return train_csv_path, test_csv_path, stats_csv_path, results_path, nn_source_data


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
    use_power = config['use_power']
    if not use_power:
        powers = None
    else:
        # If powers are given in the config dict as string, then update the power values accordingly, else use the
        # const case
        powers = consts['powers']
        if type(use_power) == str:
            power_in, power_out = eval(use_power)
            for xf, pow_x in zip(X_features, power_in):
                _, profile = xf
                powers[profile] = pow_x

            for yf, pow_y in zip(consts['Y_features'], power_out):
                powers[yf] = pow_y

            config.update({'power_in': str(power_in), 'power_out': str(power_out), 'use_power': True})

    if config['dfilter'] in ['all', None]:
        dfilter = False
    else:
        try:
            dfilter = config['dfilter'].split(' ')
            dfilter[1] = eval(dfilter[1])
        except ValueError as e:
            print(e)

    return config, X_features, powers, dfilter


# TODO: Load Trainer chekpoint  https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html
#  restoring-training-state name of trainer : epoch=29-step=3539.ckpt found under
#  C:...\main_2022-01-31_23-24-22\main_28520_00024_24_bsize=32,dfilter=('wavelength', [355]),dnorm=False,
#  fc_size=[16],hsizes=[6, 6, 6, 6],lr=0.002,ltype=MAELoss,sou_2022-02-01_11-49-31\lightning_logs\version_0
def get_checkpoint_params_const(results_path, experiment_name, trial_id, checkpoint_id):
    # experiment and trial directories
    experiment_dir = os.path.join(results_path, experiment_name)
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Wrong experiment path: {experiment_dir}")

    trial_dir = glob.glob(os.path.join(experiment_dir, rf'main_{trial_id}*'))[0]
    if not os.path.exists(trial_dir):
        raise ValueError(f"Wrong trial path: {trial_dir}")

    # Load consts
    trial_consts_path = os.path.join(trial_dir, 'consts.yaml')
    with open(trial_consts_path, 'r') as f:
        consts = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Load parameters
    trial_params_path = os.path.join(trial_dir, 'params.json')
    with open(trial_params_path, 'r') as f:
        params = json.load(f)

    # Load checkpoint
    chekpoint_name = glob.glob1(trial_dir, f'checkpoint_epoch={checkpoint_id}*')[0]
    chekpoint_dir = os.path.join(trial_dir, chekpoint_name)
    if not (os.path.exists(chekpoint_dir)):
        raise ValueError(f"Wrong checkpoint dir path: {chekpoint_dir}")

    return chekpoint_dir, params, consts


def get_experiment_consts(results_path, experiment_name):
    # experiment and trial directories
    experiment_dir = os.path.join(results_path, experiment_name)
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Wrong experiment path: {experiment_dir}")
    const_fnames = glob.glob(experiment_dir + "/**/consts.yaml", recursive=True)
    if const_fnames:
        with open(const_fnames[0], 'r') as f:
            consts = yaml.load(f.read(), Loader=yaml.FullLoader)
    return consts


# ######## SET BASIC INFO ##########
NUM_AVILABLE_CPU = os.cpu_count()
NUM_AVAILABLE_GPU = torch.cuda.device_count()
db_type = 'extended'  # options: 'extended' or 'initial'
START_DATE = datetime(2017, 4, 1) if db_type == 'extended' else datetime(2017, 9, 1)
END_DATE = datetime(2017, 10, 31)
station_name = 'haifa'
station_name = station_name + '_remote' if (sys.platform in ['linux', 'ubuntu']) else station_name
station = gs.Station(station_name)
train_csv_path, test_csv_path, stats_csv_path, RESULTS_PATH, nn_source_data = get_paths(station,
                                                                                        start_date=START_DATE,
                                                                                        end_date=END_DATE)

# ######## RESUME EXPERIMENT ######### ---> Make sure RESTORE_TRIAL = False
RESUME_EXP = False
# options: 'ERRORED_ONLY' | False | True
# Can be "LOCAL" to continue experiment when it was disrupted (trials that were completed seem to continue training),
# or "ERRORED_ONLY" to reset and rerun ERRORED trials (not tested). Otherwise, False to start a new experiment. Note:
# if fail_fast was 'True' in the folder of 'EXP_NAME', then tune will not be able to load trials that didn't
# store any folder

EXP_NAME = 'main_2022-05-21_10-26-35'  # None
# options: Path relative to RESULTS_PATH. e.g.: "main_2021-05-19_21-50-40"
# else, can keep it None --> creating new experiment path automatically
# If 'resume' is not False, must enter experiment path.

CONSTS = get_experiment_consts(RESULTS_PATH, EXP_NAME) if RESUME_EXP else None
# Load original CONSTS of the experiment that is resumed


# ######## RESTORE or VALIDATE TRIAL PARAMS #########
experiment_name = 'main_2022-03-26_19-43-28'
trial_id = 'dd418_00191'
checkpoint_id = 0

# ######## VALIDATE TRIAL #########
# This part is relevant for model_validation.py
VALIDATE_TRIAL = False  # TODO: make sure that the validation works properly when VALIDATE_TRIAL = True
if VALIDATE_TRIAL:
    PRETRAINED_MODEL_PATH, MODEL_PARAMS, MODEL_CONSTS = get_checkpoint_params_const(RESULTS_PATH, experiment_name,
                                                                                    trial_id, checkpoint_id)
    # set EXP_NAME - to save the new trial in same experiment
    EXP_NAME = experiment_name

# ######## RESTORE TRIAL #########
RESTORE_TRIAL = False  # If true restores the given trial
if RESTORE_TRIAL:
    CHECKPOINT_PATH, TRIAL_PARAMS, TRIAL_CONSTS = get_checkpoint_params_const(RESULTS_PATH, experiment_name,
                                                                              trial_id, checkpoint_id)
    # set EXP_NAME - to save the new trial in same experiment
    EXP_NAME = experiment_name
else:
    TRIAL_PARAMS = None
    CHECKPOINT_PATH = None
    TRIAL_CONSTS = None

# Constants - should correspond to data, dataloader and model
CONSTS = CONSTS if CONSTS else {
    'max_epochs': 20,
    'max_steps': None,
    'num_workers': int(NUM_AVILABLE_CPU * 0.9),
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

# Note, replace tune.choice with grid_search if you want all possible combinations
RAY_HYPER_PARAMS = {
    "hsizes": tune.grid_search(['[6,6,6,6]']),  # '[4,4,4,4]', '[5,5,5,5]',
    # Options: '[4,4,4,4]' | '[5,5,5,5]' | '[6, 6, 6, 6]' ...etc.
    "fc_size": tune.grid_search(['[32]']),  # Options: '[4]' | '[16]' | '[32]' ...etc.'[16]',
    "lr": tune.grid_search([2 * 1e-3]),
    "bsize": tune.grid_search([32]),
    "ltype": tune.grid_search(['MAELoss']),  # Options: 'MAELoss' | 'MSELoss' | 'MARELoss'. See 'custom_losses.py'
    "use_power": tune.grid_search(['([0.5,-.27],[1])', #'([0.5,-.27,.5],[1])',
                                   '([0.5,-.1],[1])', #'([0.5,-.1,.5],[1])',
                                   '([0.5,-.2],[1])', #'([0.5,-.2,.5],[1])',
                                   '([0.5,-.28],[1])',# '([0.5,-.28,.5],[1])',
                                   '([0.5,-.3],[1])', #'([0.5,-.3,.5],[1])',
                                   '([0.5,.5],[1])', #'([0.5,.5,.5],[1])',
                                   '([0.5,-.5],[1])', #'([0.5,-.5,.5],[1])',
                                   False
                                   ]),
    # '([0.5,-.23,1],[1])','([0.5,-.23,.5],[1])', #'([0.5,-.25,1],[1])', #'([0.5,-.25,.5],[1])',
    # '([0.5,0.21,1],[1])','([0.5,0.21,.5],[1])',#'([0.5,-0.1,0.5],[1])',# '([0.5,-0.1,1],[1])',
    # '([0.5,-0.21,1],[1])','([0.5,-0.21,.5],[1])',#'([0.5,-0.3,0.5],[1])', #'([0.5,-0.3,1],[1])',
    # '([0.5,.19,1],[1])','([0.5,.19,.5],[1])',
    # '([0.5,-.19,1],[1])','([0.5,-.19,.5],[1])',
    # '([0.5,.1,1],[1])','([0.5,.1,.5],[1])']),#'([0.5,-.5,.5],[1])', #'([0.5,-.5,1],[1])']),
    # Options: False | '([0.5,1,1], [0.5])' ...etc. UV  : -0.27 , G: -0.263 , IR: -0.11
    "opt_powers": tune.choice([False]),  # Options: False | True
    "use_bg": tune.grid_search([False]),  # False | True |  'range_corr'
    # Options: False | True | 'range_corr'. Not relevant for 'signal' as source
    "source": tune.grid_search(['lidar']),  # Options: 'lidar'| 'signal_p' | 'signal'
    'dfilter': tune.grid_search([None]),  # "wavelength [355]", "wavelength [532]",
    # None,"wavelength [355]", "wavelength [532]","wavelength [1064]"]),  # Options: None | '(wavelength, [lambda])'
    # - lambda=355,532,1064
    'dnorm': tune.grid_search([False]),  # Options: False | True
    'overfit': tune.grid_search([False]),  # Apply over fit mode of pytorch lightening. Note: Change bsize to 10
    'debug': tune.choice([False]),  # Apply debug mode of pytorch lightening
    'cbias': tune.grid_search([True]),  # Calc convolution biases. This may be redundant if using batch norm
    'wdecay': tune.choice([0]),  # Weight decay algorithm to test l2 regularization of NN weights.
    # 'operations': tune.grid_search(["(None, None, ['poiss','r2'])"])
    # Apply l2 regularization on model weights. parameter weight_decay of Adam optimiser
    # afterwards
    'db_type': tune.grid_search([db_type]),
    # 'extended' or 'initial'. This is set at the beginning.(adding it for logging)
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
    'wdecay': 0,  # Weight decay algorithm to test l2 regularization of NN weights.
    # 'operations': None
    'db_type': db_type,  # 'extended' or 'initial'. This is set at the beginning.(adding it for logging)
}
USE_RAY = True
DEBUG_RAY = False
INIT_PARAMETERS = True
CONSTS.update({'max_epochs': 30})
