import logging
import os.path
import yaml
from datetime import datetime

import ray
from pytorch_lightning import Trainer, seed_everything
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN
from learning_lidar.learning_phase.run_params import USE_RAY, DEBUG_RAY, CONSTS, RAY_HYPER_PARAMS, RESULTS_PATH, \
    NUM_AVAILABLE_GPU, NON_RAY_HYPER_PARAMS, update_params, RESUME_EXP, EXP_NAME, TRIAL_PARAMS, CHECKPOINT_PATH
from learning_lidar.utils.utils import create_and_configer_logger

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer


# for pytorch lightning and Ray integration see example at
# https://github.com/ray-project/ray/blob/35ec91c4e04c67adc7123aa8461cf50923a316b4/python/ray/tune/examples/mnist_pytorch_lightning.py

def main(config, checkpoint_dir=None, consts=None):
    with open('consts.yaml', 'a') as f:
        yaml.dump(consts, f)
    with open('config.yaml', 'a') as f:
        yaml.dump(config, f)
    config, X_features, powers, dfilter = update_params(config, consts)

    # Define Model
    if checkpoint_dir:
        model = DefaultCNN.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))
    else:
        model = DefaultCNN(in_channels=len(X_features),
                           output_size=len(consts['Y_features']),
                           hidden_sizes=eval(config['hsizes']),
                           fc_size=eval(config['fc_size']),
                           loss_type=config['ltype'],
                           learning_rate=config['lr'],
                           X_features_profiles = X_features,
                           powers = powers,
                           do_opt_powers=config['opt_powers'])

    # Define Data
    lidar_dm = LidarDataModule(train_csv_path=consts["train_csv_path"],
                               test_csv_path=consts["test_csv_path"],
                               stats_csv_path=consts["stats_csv_path"],
                               powers=powers if config['use_power'] else None,
                               top_height=consts["top_height"], X_features_profiles=X_features,
                               Y_features=consts['Y_features'], batch_size=config['bsize'],
                               num_workers=consts['num_workers'],
                               data_filter=dfilter,
                               data_norm=config['dnorm'])

    # Define minimization parameter
    metrics = {"loss": f"loss/{config['ltype']}_val",
               "MARELoss": "rel_loss/MARELoss_val"}
    callbacks = [TuneReportCheckpointCallback(metrics, filename="checkpoint", on="validation_end")]

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_steps=consts['max_steps'],
                      max_epochs=consts['max_epochs'],
                      callbacks=callbacks,
                      gpus=[0] if consts['num_gpus'] > 0 else 0)

    lidar_dm.setup('fit')
    trainer.fit(model=model, datamodule=lidar_dm)


if __name__ == '__main__':
    # Override number of workers if debugging
    CONSTS['num_workers'] = 0 if DEBUG_RAY else CONSTS['num_workers']

    logger = create_and_configer_logger(
        log_name=f"{os.path.dirname(__file__)}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.log", level=logging.INFO)

    if DEBUG_RAY:
        ray.init(local_mode=True)

    if USE_RAY:
        reporter = CLIReporter(
            metric_columns=["loss", "MARELoss", "training_iteration"],
            max_progress_rows=50,
            print_intermediate_tables=True)

        analysis = tune.run(
            tune.with_parameters(main, consts=CONSTS),
            config=TRIAL_PARAMS if TRIAL_PARAMS else RAY_HYPER_PARAMS,
            local_dir=RESULTS_PATH,  # where to save the results
            fail_fast=False,  # if one run fails - stop all runs
            metric="MARELoss",
            mode="min",
            progress_reporter=reporter,
            log_to_file=True,
            resources_per_trial={"cpu": 7, "gpu": NUM_AVAILABLE_GPU},
            resume=RESUME_EXP, name=EXP_NAME,
            restore=CHECKPOINT_PATH
        )

        logger.info(f"best_trial {analysis.best_trial}")
        logger.info(f"best_config {analysis.best_config}")
        logger.info(f"best_logdir {analysis.best_logdir}")
        logger.info(f"best_checkpoint {analysis.best_checkpoint}")
        logger.info(f"best_result {analysis.best_result}")
        results_df = analysis.dataframe(metric="MARELoss", mode="min", )
        results_df.to_csv(os.path.join(analysis._experiment_dir, f'output_table.csv'))
    else:
        main(config=NON_RAY_HYPER_PARAMS, consts=CONSTS)
