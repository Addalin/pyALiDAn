import logging
import os.path
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything

from learning_lidar.learning_phase.data_modules.lidar_data_module import LidarDataModule
from learning_lidar.learning_phase.models.defaultCNN import DefaultCNN
from learning_lidar.learning_phase.run_params import DEBUG_RAY, CONSTS, update_params, MODEL_PARAMS, \
    PRETRAINED_MODEL_PATH, MODEL_CONSTS
from learning_lidar.utils.utils import create_and_configer_logger

seed_everything(8318)  # Note, for full deterministic result add deterministic=True to trainer


def main(config, checkpoint_dir=None, consts=None):
    config, X_features, powers = update_params(config, consts)

    model = DefaultCNN.load_from_checkpoint(os.path.join(checkpoint_dir, "checkpoint"))

    # Define Data
    lidar_dm = LidarDataModule(nn_data_folder=consts['nn_source_data'], train_csv_path=consts["train_csv_path"],
                               test_csv_path=consts["test_csv_path"], stats_csv_path=consts["stats_csv_path"],
                               powers=powers if config['use_power'] else None, top_height=consts["top_height"],
                               X_features_profiles=X_features, Y_features=consts['Y_features'],
                               batch_size=config['bsize'], num_workers=consts['num_workers'],
                               data_filter=config['dfilter'], data_norm=config['dnorm'])

    # Setup the pytorch-lighting trainer and run the model
    trainer = Trainer(max_epochs=consts['max_epochs'],
                      gpus=[0] if consts['num_gpus'] > 0 else 0)

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

    main(config=MODEL_PARAMS, checkpoint_dir=PRETRAINED_MODEL_PATH, consts=MODEL_CONSTS)
