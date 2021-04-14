import os.path
from datetime import datetime

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from pytorch_lightning import Trainer, seed_everything

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer


def main(config, consts):
    # Define Model
    model = DefaultCNN(in_channels=consts['in_channels'], output_size=len(config['Y_features']),
                       hidden_sizes=consts['hidden_sizes'], loss_type=config['loss_type'], learning_rate=config['lr'])

    # Define Data
    lidar_dm = LidarDataModule(csv_path=consts["csv_path"],
                               powers=consts['powers'] if config['use_power'] else None,
                               Y_features=config['Y_features'],
                               batch_size=config['bsize'], num_workers=consts['num_workers'])

    # Define minimization parameter
    metrics = {"loss": f"{config['loss_type']}_val"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    # Setup the pytorchlighting trainer and run the model
    trainer = Trainer(max_epochs=consts['max_epochs'], callbacks=callbacks)
    # trainer = Trainer(max_steps=consts['max_steps'])
    trainer.fit(model, datamodule=lidar_dm)


if __name__ == '__main__':
    # Debug flag to enable debugging
    DEBUG = False
    if DEBUG:
        ray.init(local_mode=True)

    data_params = {
        'base_path': r'C:\Users\addalin\Dropbox\Lidar\code',
        'station_name': 'haifa',
        'start_date': datetime(2017, 9, 1),
        'end_date': datetime(2017, 10, 31),
    }

    csv_path = os.path.join(data_params['base_path'], f"dataset_{data_params['station_name']}_"
                                                     f"{data_params['start_date'].strftime('%Y-%m-%d')}_"
                                                     f"{data_params['end_date'].strftime('%Y-%m-%d')}_on_D.csv")

    # Constants - should correspond to data, dataloader and model
    consts = {
        "hidden_sizes": [16, 32, 8],  # TODO: add options of [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
        'in_channels': 2,
        'max_epochs': 3,
        'num_workers': 7,
        'csv_path':csv_path,
        'powers': {'range_corr': 0.5, 'attbsc': 0.5, 'LC': 0.5, 'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1}
    }

    # Defining a search space
    # Note, replace choice with grid_search if want all possible combinations
    use_ray=True
    if use_ray:
        hyper_params =  {
            "lr": tune.grid_search([1e-3, 0.5 * 1e-3, 1e-4]),
            "bsize": tune.choice([8]),
            #"wavelengths": tune.grid_search([355, 532, 1064]),  # TODO change to const - all wavelenghts
            "loss_type": tune.choice(['MSELoss', 'MAELoss']),  # ['MARELoss']
            "Y_features": tune.choice([['LC']] ),
            #tune.choice(
                #[['LC'], ['r0', 'r1', 'LC'], ['r0', 'r1'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']]),
            "use_power": tune.grid_search([False, True])
        }

        analysis = tune.run(
            tune.with_parameters(main, consts=consts),
            config=hyper_params,
            # name="cnn",
            local_dir="../../results",  # where to save the results
            fail_fast=True,  # if one run fails - stop all runs
            metric="loss",
            mode="min",
            resources_per_trial={"cpu": 7, "gpu": 0})

        print(f"best_trial {analysis.best_trial}")
        print(f"best_config {analysis.best_config}")
        print(f"best_logdir {analysis.best_logdir}")
        print(f"best_checkpoint {analysis.best_checkpoint}")
        print(f"best_result {analysis.best_result}")
    else:
        hyper_params = {"Y_features" : [ "r0" , "r1" , "LC" ] , "bsize" : 8 , "loss_type" : "MSELoss" ,
                        "lr" : 0.001 , "powers" : None}
        main ( hyper_params , consts )