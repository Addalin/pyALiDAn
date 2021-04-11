from datetime import datetime

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from pytorch_lightning import Trainer, seed_everything

from data_modules.lidar_data_module import LidarDataModule
from models.defaultCNN import DefaultCNN

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer


def main(config, consts):
    # Define Model
    model = DefaultCNN(in_channels=consts['in_channels'], output_size=len(config['Y_features']),
                       hidden_sizes=consts['hidden_sizes'], loss_type=config['loss_type'], learning_rate=config['lr'])

    # Define Data
    csv_path = f"/home/shubi/PycharmProjects/learning_lidar/dataset_{consts['station_name']}_" \
               f"{consts['start_date'].strftime('%Y-%m-%d')}_{consts['end_date'].strftime('%Y-%m-%d')}_shubi_mini.csv"

    lidar_dm = LidarDataModule(csv_path=csv_path, powers=config['powers'], Y_features=config['Y_features'],
                            batch_size=config['batch_size'])

    # Define minimization parameter
    metrics = {"loss": f"{config['loss_type']}_val"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    # Setup the pytorchlighting trainer and run the model
    trainer = Trainer(max_steps=consts['max_steps'], callbacks=callbacks)
    trainer.fit(model, datamodule=lidar_dm)


if __name__ == '__main__':
    # Debug flag to enable debugging
    DEBUG = False
    if DEBUG:
        ray.init(local_mode=True)

    # Constants - should correspond to data, dataloader and model
    consts = {
        'station_name': 'haifa',
        'start_date': datetime(2017, 9, 1),
        'end_date': datetime(2017, 10, 31),
        "hidden_sizes": [16, 32, 8],  # TODO: add options of [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
        'in_channels': 2,
        'max_steps': 30,
    }

    # Defining a search space
    # Note, replace choice with grid_search if want all possible combinations
    hyper_params = {
        "lr": tune.choice([1e-3, 0.5 * 1e-3, 1e-4]),
        "batch_size": tune.choice([8]),
        "wavelengths": tune.grid_search([355, 532, 1064]),
        "loss_type": tune.choice(['MSELoss', 'MAELoss']),  # ['MARELoss']
        "Y_features": tune.choice([['r0', 'r1'], ['r0', 'r1', 'LC'], ['LC']]),
        # TODO with dr - ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']
        "powers": tune.grid_search([None,
                                    {'range_corr': 0.5, 'attbsc': 0.5, 'LC': 0.5,
                                     'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1}])
    }

    analysis = tune.run(
        tune.with_parameters(main, consts=consts),
        config=hyper_params,
        # name="cnn",
        local_dir="./results",
        fail_fast=True,
        metric="loss",
        mode="min",
        resources_per_trial={"cpu": 6, "gpu": 0})

    print(f"best_trial {analysis.best_trial}")
    print(f"best_config {analysis.best_config}")
    print(f"best_logdir {analysis.best_logdir}")
    print(f"best_checkpoint {analysis.best_checkpoint}")
    print(f"best_result {analysis.best_result}")
