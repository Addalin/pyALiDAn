import warnings

import ray

from data_modules.lidar_data_module import lidarDataSet, MyDataModule
from models.defaultCNN import DefaultCNN
from utils_.custom_operations import PowTransform, LidarToTensor
from ray.tune.integration.pytorch_lightning import TuneReportCallback

warnings.filterwarnings("ignore")
import pandas as pd
import os
from datetime import datetime, time
import numpy as np
from ray import tune

import torch, torchvision
import torch.utils.data
import xarray as xr
import preprocessing as prep
import torch.nn as nn
import time
from utils import create_and_configer_logger
from torch.utils.tensorboard import SummaryWriter
from torch import functional as F

# from ignite.contrib.metrics.regression import MeanAbsoluteRelativeError #This method is not working yet
torch.manual_seed(8318)
import json
from pytorch_lightning import Trainer


def calculate_statistics(model, criterion, run_params, loader, device, Y_features=None, wavelengths=None):
    """

    :param model:torch.nn.Module
    :param criterion: loss criterion function
    :param run_params: dict of running parameters
    :param loader: torch.utils.data.DataLoader
    :param device: torch.device ()
    :param Y_features: features list to calculate loss separately, for debug (this is an optional input)
    :param wavelengths: wavelengths list to calculate loss separately, for debug (this is an optional input)
    :return: stats - a dict containing train/validation loss criterion, and separately feature losses (for debug)
    """
    model.eval()  # Evaluation mode
    criterion = criterion

    if Y_features:
        # Initializing FeatureLoss - For debug. this loss is not affecting the model.
        count_less = {}
        feature_loss = {}
        for idx_f, feature in enumerate(Y_features):
            feature_loss.update({feature: {}})
            feature_loss[feature].update({'all': 0.0})
            for wav in wavelengths:
                feature_loss[feature].update({wav: 0.0})
                count_less.update({wav: 0.0})

    running_loss = 0.0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            y_pred = model(x)
            loss = criterion(y, y_pred)
            running_loss += loss.data.item()

            if Y_features:
                wavelength = sample['wavelength']
                for idx_f, feature in enumerate(Y_features):
                    feature_loss[feature]['all'] += mare_loss(y_pred[:, idx_f], y[:, idx_f]).data.item()
                    for wav in wavelengths:
                        idx_w = torch.where(wavelength == wav)[0]
                        if idx_w.numel() > 0:
                            feature_loss[feature][wav] += \
                                mare_loss(y_pred[idx_w][:, idx_f], y[idx_w][:, idx_f]).data.item()
                        else:
                            count_less[wav] += 1
    running_loss /= len(loader)
    stats = {f"{run_params['loss_type']}": running_loss}
    if Y_features:
        for idx_f, feature in enumerate(Y_features):
            feature_loss[feature]['all'] /= len(loader)
            for wav in wavelengths:
                feature_loss[feature][wav] /= (len(loader) - count_less[wav])
        stats.update({'FeatureLoss': feature_loss})
    return stats


def update_stats(writer, model, loaders, device, run_params, criterion, epoch):
    """
    Update current epoch's state to writer
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param model: torch.nn.Module
    :param loaders: a list of [train_loader,val_loader], each of type torch.utils.data.DataLoader
    :param device:  torch.device ()
    :param run_params: dict of running parameters
    :param criterion: loss criterion function
    :param epoch: current epoch
    :return: curr_loss - current loss for train and for validation sets
    """
    curr_loss = {}
    feature_loss = {}
    for loader, mode in zip(loaders, ['train', 'val']):
        # calc current epoch statistics: train and val model, and feature loss for debug.
        stats = calculate_statistics(model, criterion, run_params, loader, device,
                                     Y_features=run_params['Y_features'],
                                     wavelengths=run_params['wavelengths'])

        # add loss for current epoch's model
        field_name = f"{run_params['loss_type']}/{mode}"
        field_value = stats[f"{run_params['loss_type']}"]
        writer.add_scalar(field_name, field_value, epoch)
        curr_loss.update({mode: field_value})

        # add feature losses for debug
        feature_loss.update({mode: {}})
        for feature in run_params['Y_features']:
            for wav in run_params['wavelengths']:
                # add FeatureLoss (per wavelength per feature) , currently the metric is MAERLoss.
                # For debug , this loss is not affecting the model.
                field_name = f"FeatureLoss_{mode}/{feature}_{wav}"
                field_value = stats['FeatureLoss'][feature][wav]
                writer.add_scalar(field_name, field_value, epoch)

            # add common FeatureLoss, currently the metric is MAERLoss.
            # For debug , this loss is not affecting the model.
            field_name = f"FeatureLoss_{mode}/{feature}"
            field_value = stats['FeatureLoss'][feature]['all']
            writer.add_scalar(field_name, field_value, epoch)
            feature_loss[mode].update({feature: field_value})
    return curr_loss, feature_loss


def write_hparams(writer, run_params, run_name, cur_loss, best_loss, cur_loss_feature, best_loss_feature):
    results = {'hparam_last/loss_train': cur_loss['loss']['train'],
               'hparam_last/loss_val': cur_loss['loss']['val'],
               'hparam_last/epoch': cur_loss['epoch'],
               'hparam_best/loss_train': best_loss['loss']['train'],
               'hparam_best/epoch_train': best_loss['epoch']['train'],
               'hparam_best/loss_val': best_loss['loss']['val'],
               'hparam_best/epoch_val': best_loss['epoch']['val'],
               }
    for mode in ['train', 'val']:
        for feature in run_params['Y_features']:
            results.update({f"{feature}_last/loss_{mode}": cur_loss_feature['loss'][mode][feature]})
            results.update({f"{feature}_last/epoch_{mode}": cur_loss_feature['epoch']})
            results.update({f"{feature}_best/loss_{mode}": best_loss_feature['loss'][mode][feature]})
            results.update({f"{feature}_best/epoch_{mode}": best_loss_feature['epoch'][mode][feature]})

    run_params['Y_features'] = '_'.join(run_params['Y_features'])
    run_params['hidden_sizes'] = '_'.join([str(val) for val in run_params['hidden_sizes']])
    run_params['wavelengths'] = '_'.join([str(val) for val in run_params['wavelengths']])
    run_params['powers'] = run_params['powers']['LC'] if run_params['powers'] is not None else 1

    writer.add_hparams(hparam_dict=run_params, metric_dict=results, run_name=run_name)


def get_model_dirs(run_name, model_n, s_model):
    model_dir = f"model_{model_n}"
    submodel_dir = f"model_{model_n}.{s_model}"
    main_dir = os.path.join(os.getcwd(), "cnn_models", model_dir, submodel_dir, run_name)
    checkpoints_dir = os.path.join(main_dir, "checkpoints")
    run_dir = os.path.join(main_dir, "run")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    return {'main': main_dir, 'checkpoints': checkpoints_dir, 'run': run_dir}


def main(config, consts):

    # Define Model
    model = DefaultCNN(in_channels=consts['in_channels'], output_size=len(config['Y_features']),
                       hidden_sizes=consts['hidden_sizes'], loss_type=config['loss_type'], learning_rate=config['lr'])

    # Define Data
    lidar_dm = MyDataModule(station_name=consts['station_name'], start_date=consts['start_date'],
                            end_date=consts['end_date'], powers=config['powers'], Y_features=config['Y_features'],
                            batch_size=config['batch_size'])

    # Define minimization parameter
    metrics = {"loss": f"{config['loss_type']}_val"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]

    # Setup the pytorchlighting trainer and run the model
    trainer = Trainer(max_steps=consts['max_steps'], callbacks=callbacks)
    trainer.fit(model, datamodule=lidar_dm)


if __name__ == '__main__':
    DEBUG = False
    if DEBUG:
        ray.init(local_mode=True)

    consts = {
        'station_name': 'haifa',
        'start_date': datetime(2017, 9, 1),
        'end_date': datetime(2017, 10, 31),
        "hidden_sizes": [16, 32, 8],
        'in_channels': 2,
        'max_steps': 30,
    }

    # Defining a search space TODO replace choice with grid_search if want all possible combinations
    hyper_params = {
        "lr": tune.choice([1e-3, 0.5 * 1e-3, 1e-4]),
        "batch_size": tune.choice([8]),
        "wavelengths": tune.choice([355, 532, 1064]),
        # TODO: add option - hidden_sizes = [ 8, 16, 32], [16, 32, 8], [ 64, 32, 16]
        "loss_type": tune.choice(['MSELoss', 'MAELoss']),  # ['MARELoss']
        # "Y_features": tune.choice([['r0', 'r1'], ['r0', 'r1', 'LC'], ['r0', 'r1', 'dr'], ['r0', 'r1', 'dr', 'LC'], ['LC']]), # TODO with dr
        "Y_features": tune.choice([['r0', 'r1'], ['r0', 'r1', 'LC'], ['LC']]),
        "powers": tune.grid_search([None, {'range_corr': 0.5, 'attbsc': 0.5, 'LC': 0.5,
                                           'LC_std': 0.5, 'r0': 1, 'r1': 1, 'dr': 1}])
    }
    #
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

    #
    # PATH="results/_inner_2021-03-29_17-37-01/_inner_3917a_00000_0_Y_features=['LC'],batch_size=8,loss_type=MSELoss,lr=0.0005,powers=None,wavelengths=532_2021-03-29_17-37-01/lightning_logs/version_0/checkpoints/epoch=1-step=19.ckpt"
    # model = DefaultCNN.load_from_checkpoint(PATH)
    # trainer = Trainer()
    # lidar_dm = MyDataModule(station_name=consts['station_name'], start_date=consts['start_date'],
    #                         end_date=consts['end_date'], powers=None, Y_features=['LC'],
    #                         batch_size=8)
    # x=trainer.test(model, datamodule=lidar_dm)
    # x=trainer.test(model, ckpt_path=PATH, test_dataloaders=[lidar_dm.train_dataloader(), lidar_dm.val_dataloader()])

    DEBUG_WITHOUT_RAY = False
    if DEBUG_WITHOUT_RAY:
        run_param = {
            "Y_features": [
                "r0",
                "r1",
            ],
            "batch_size": 8,
            "hidden_sizes": [
                16,
                32,
                8
            ],
            "lr": 0.0005,
            "loss_type": "MSELoss",
            "powers": None,
            "wavelengths": 532
        }
        main(config=run_param, consts=consts)
