import logging
import os.path
from datetime import datetime

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from run_params import CONSTS, RAY_HYPER_PARAMS, RESULTS_PATH, NUM_AVAILABLE_GPU, NON_RAY_HYPER_PARAMS

from pytorch_lightning import Trainer, seed_everything

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN
from learning_lidar.utils.utils import create_and_configer_logger

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer


# for pytorch lightning and Ray integration see example at
# https://github.com/ray-project/ray/blob/35ec91c4e04c67adc7123aa8461cf50923a316b4/python/ray/tune/examples/mnist_pytorch_lightning.py

def main(config, checkpoint_dir=None, consts=None):
    # Define X_features
    source_features = (f"{config['source']}_path", "range_corr")
    mol_features = ("molecular_path", "attbsc")
    bg_features = ("bg_path", "p_bg")
    X_features = (source_features, mol_features, bg_features) if config["use_bg"] else (source_features, mol_features)

    # Define Model
    model = DefaultCNN(in_channels=len(X_features),
                       output_size=len(config['Y_features']),
                       hidden_sizes=config['hidden_sizes'],
                       fc_size=consts['fc_size'],
                       loss_type=config['loss_type'],
                       learning_rate=config['lr'])

    # Define Data
    lidar_dm = LidarDataModule(train_csv_path=consts["train_csv_path"], test_csv_path=consts["test_csv_path"],
                               stats_csv_path=consts["stats_csv_path"],
                               powers=consts['powers'] if config['use_power'] else None,
                               top_height=consts["top_height"], X_features_profiles=X_features,
                               Y_features=config['Y_features'], batch_size=config['bsize'],
                               num_workers=consts['num_workers'],
                               data_filter=config['data_filter'],
                               data_norm=config['data_norm'])

    # Define minimization parameter
    metrics = {"loss": f"{config['loss_type']}_val",
               "MARELoss": "MARELoss_val"}
    callbacks = [TuneReportCheckpointCallback(metrics, filename="checkpoint", on="validation_end")]

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_epochs=consts['max_epochs'],
                      callbacks=callbacks,
                      gpus=[1] if consts['num_gpus'] > 0 else 0)

    lidar_dm.setup('fit')
    trainer.fit(model=model, datamodule=lidar_dm)

    # test
    lidar_dm.setup('test')
    trainer.test(model=model, datamodule=lidar_dm)


if __name__ == '__main__':
    USE_RAY = True
    DEBUG_RAY = False

    # Override number of workers if debugging
    CONSTS['num_workers'] = 0 if DEBUG_RAY else CONSTS['num_workers']

    logger = create_and_configer_logger(
        log_name=f"{os.path.dirname(__file__)}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log", level=logging.INFO)

    if DEBUG_RAY:
        ray.init(local_mode=True)

    if USE_RAY:
        reporter = CLIReporter(
            metric_columns=["loss", "MARELoss", "training_iteration"])

        analysis = tune.run(
            tune.with_parameters(main, consts=CONSTS),
            config=RAY_HYPER_PARAMS,
            local_dir=RESULTS_PATH,  # where to save the results
            fail_fast=True,  # if one run fails - stop all runs
            metric="MARELoss",
            mode="min",
            progress_reporter=reporter,
            log_to_file=True,
            resources_per_trial={"cpu": 7, "gpu": NUM_AVAILABLE_GPU})

        logger.info(f"best_trial {analysis.best_trial}")
        logger.info(f"best_config {analysis.best_config}")
        logger.info(f"best_logdir {analysis.best_logdir}")
        logger.info(f"best_checkpoint {analysis.best_checkpoint}")
        logger.info(f"best_result {analysis.best_result}")
    else:
        main(NON_RAY_HYPER_PARAMS, CONSTS)
