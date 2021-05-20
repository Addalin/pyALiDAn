import logging
import os.path
from datetime import datetime

from learning_lidar.learning_phase.run_params import DEBUG_RAY, CONSTS, RESULTS_PATH

import json
from pytorch_lightning import Trainer, seed_everything

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN
from learning_lidar.utils.utils import create_and_configer_logger

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer

def main(config, checkpoint_dir=None, consts=None):
    # Define X_features
    source_x = config['source']
    source_features = (f"{source_x}_path", "range_corr") \
        if (source_x == 'lidar' or source_x == 'signal') \
        else (f"{source_x}_path", "range_corr_p")
    mol_features = ("molecular_path", "attbsc")
    bg_features = ("bg_path", "p_bg")
    X_features = (source_features, mol_features, bg_features) if config["use_bg"] else (source_features, mol_features)

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

    model = DefaultCNN.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))

    # Define Data
    lidar_dm = LidarDataModule(train_csv_path=consts["train_csv_path"],
                               test_csv_path=consts["test_csv_path"],
                               stats_csv_path=consts["stats_csv_path"],
                               powers=powers if config['use_power'] else None,
                               top_height=consts["top_height"], X_features_profiles=X_features,
                               Y_features=consts['Y_features'], batch_size=config['bsize'],
                               num_workers=consts['num_workers'],
                               data_filter=config['dfilter'],
                               data_norm=config['dnorm'])

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_epochs=consts['max_epochs'],
                      gpus=[1] if consts['num_gpus'] > 0 else 0)

    lidar_dm.setup('fit')
    trainer.validate(model=model, datamodule=lidar_dm)

    # TODO TEST NOT WORKING
    lidar_dm.setup('test')
    trainer.test(model=model, datamodule=lidar_dm)


if __name__ == '__main__':
    # Override number of workers if debugging
    CONSTS['num_workers'] = 0 if DEBUG_RAY else CONSTS['num_workers']

    logger = create_and_configer_logger(
        log_name=f"{os.path.dirname(__file__)}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log", level=logging.INFO)

    experiment_dir = 'main_2021-05-19_23-17-25'
    trial_dir = r'main_39b80_00000_0_bsize=32,dfilter=None,dnorm=True,fc_size=[16],hsizes=[3, 3, 3, 3],lr=0.001,ltype=MAELoss,source=lidar,use_bg=Tr_2021-05-19_23-17-25'
    check_point_path = 'checkpoint_epoch=0-step=175'

    model_path = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, check_point_path)
    params_path = os.path.join(RESULTS_PATH, experiment_dir, trial_dir, 'params.json')

    with open(params_path) as params_file:
        params = json.load(params_file)

    main(config=params, checkpoint_dir=model_path, consts=CONSTS)
