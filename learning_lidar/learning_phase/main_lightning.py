import os.path
from datetime import datetime

import ray
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from pytorch_lightning import Trainer, seed_everything

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN

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
    lidar_dm = LidarDataModule(train_csv_path=consts["train_csv_path"],
                               test_csv_path=consts["test_csv_path"],
                               powers=consts['powers'] if config['use_power'] else None,
                               X_features_profiles=X_features,
                               Y_features=config['Y_features'],
                               batch_size=config['bsize'],
                               num_workers=consts['num_workers'],
                               data_filter=config['data_filter'])

    # Define minimization parameter
    metrics = {"loss": f"{config['loss_type']}_val",
               "MARELoss": "MARELoss_val"}
    callbacks = [TuneReportCheckpointCallback(metrics, filename="checkpoint", on="validation_end")]

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_epochs=consts['max_epochs'], callbacks=callbacks, gpus=[1])
    # trainer = Trainer(max_steps=consts['max_steps'])
    lidar_dm.setup('fit')
    trainer.fit(model=model, datamodule=lidar_dm)

    # test
    lidar_dm.setup('test')
    trainer.test(model=model, datamodule=lidar_dm)


if __name__ == '__main__':
    # Debug flag to enable debugging
    max_workers = 6
    DEBUG_RAY = True
    if DEBUG_RAY:
        ray.init(local_mode=True)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_params = {
        'base_path': os.path.join(base_path, 'data'),
        'station_name': 'haifa',
        'start_date': datetime(2017, 9, 1),
        'end_date': datetime(2017, 10, 31),
    }

    csv_base_name = f"dataset_gen_{data_params['station_name']}_" \
                    f"{data_params['start_date'].strftime('%Y-%m-%d')}_" \
                    f"{data_params['end_date'].strftime('%Y-%m-%d')}"
    train_csv_path = os.path.join(data_params['base_path'], f'{csv_base_name}_train.csv')
    test_csv_path = os.path.join(data_params['base_path'], f'{csv_base_name}_test.csv')

    # Constants - should correspond to data, dataloader and model
    consts = {
        "fc_size": [512],
        'max_epochs': 2,
        'num_workers': 0 if DEBUG_RAY else max_workers,
        'train_csv_path': train_csv_path,
        'test_csv_path': test_csv_path,
        'powers': {'range_corr': 0.5, 'attbsc': 0.5, 'p_bg': 0.5,
                   'LC': 0.5, 'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1},
    }

    # Defining a search space
    # Note, replace choice with grid_search if want all possible combinations
    use_ray = True
    if use_ray:
        hyper_params = {
            "hidden_sizes": tune.choice([[16, 32, 8]]), # TODO: add options of [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
            "lr": tune.grid_search([1e-3, 0.5 * 1e-3, 1e-4]),
            "bsize": tune.choice([32]),#[16, 8]),
            "loss_type": tune.choice(['MSELoss', 'MAELoss']),  # ['MARELoss']
            "Y_features": tune.choice([['LC']]),
            # [['LC'], ['r0', 'r1', 'LC'], ['r0', 'r1'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC']]
            "use_power": tune.grid_search([True, False]),
            "use_bg": tune.grid_search([False]),  # True - bg is relevant for 'lidar' case
            "source": tune.grid_search(['signal', 'lidar']),
            'data_filter': tune.grid_search([('wavelength', [355]), None])
        }

        analysis = tune.run(
            tune.with_parameters(main, consts=consts),
            config=hyper_params,
            # name="cnn",
            local_dir=os.path.join(base_path, 'results'),  # where to save the results
            fail_fast=True,  # if one run fails - stop all runs
            metric="loss",
            mode="min",
            resources_per_trial={"cpu": 7, "gpu": 2})

        print(f"best_trial {analysis.best_trial}")
        print(f"best_config {analysis.best_config}")
        print(f"best_logdir {analysis.best_logdir}")
        print(f"best_checkpoint {analysis.best_checkpoint}")
        print(f"best_result {analysis.best_result}")
    else:
        hyper_params = {
            "lr": 1 * 1e-3,
            "bsize": 8,
            "loss_type": 'MSELoss',
            "Y_features": ['LC'],
            "use_power": True,
            "use_bg": False,
            "source": 'signal',
            "hidden_sizes": [16, 32, 8],
            "source": 'signal',
            'data_filter': None,

        }

        main(hyper_params, consts)
