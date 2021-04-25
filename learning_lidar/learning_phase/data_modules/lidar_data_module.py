import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from learning_lidar.preprocessing import preprocessing as prep
from learning_lidar.learning_phase.utils_.custom_operations import PowTransform, LidarToTensor


class LidarDataSet(torch.utils.data.Dataset):
    """
    TODO: add usage
    """

    def __init__(self, csv_file, transform, top_height, X_features_profiles, Y_features, wavelengths=None):
        """
        Args:
            csv_file (string): Path to the csv file of the database.
            :param transform:
        """
        self.data = pd.read_csv(csv_file)
        self.key = ['idx', 'date', 'wavelength', 'cali_method', 'telescope',
                    'cali_start_time', 'cali_stop_time',
                    'start_time_period', 'end_time_period']
        self.X_features, self.profiles = map(list, zip(*X_features_profiles))  # unpack list of tuples into two lists
        self.Y_features = Y_features
        self.wavelengths = wavelengths  # TODO: make option to load data by desired wavelength/s
        self.top_height = top_height
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load data
        X = self.load_X(idx)
        Y = self.load_Y(idx)
        wavelength = self.get_key_val(idx, 'wavelength').astype(np.int32)

        sample = {'x': X, 'y': Y}
        if self.transform:
            sample = self.transform(sample)
            wavelength = torch.from_numpy(np.asarray(wavelength))
        sample['wavelength'] = wavelength
        return sample

    def get_splits(self, n_test=0.2, n_val=0.2):
        if n_test:
            test_size = round(n_test * len(self))
            val_size = round(n_val * len(self))
            train_size = len(self) - val_size - test_size
            train_set, val_set, test_set = random_split(self, [train_size, val_size, test_size])
            return train_set, val_set, test_set
        else:
            val_size = round(n_val * len(self))
            train_size = len(self) - val_size
            train_set, val_set = random_split(self, [train_size, val_size])
            return train_set, val_set

    def load_X(self, idx):
        """
        Returns X samples - measurements of lidar, and molecular
        :param idx: index of the sample
        :return: A list of two element each is of type xarray.core.dataarray.DataArray.
        0 - is for lidar measurements, 1 - is for molecular measurements
        """
        row = self.data.loc[idx, :]

        # Load X datasets
        X_paths = row[self.X_features]
        datasets = [prep.load_dataset(path) for path in X_paths]

        # Calc sample height and time slices
        hslices = [
            slice(ds.Height.min().values.tolist(), ds.Height.min().values.tolist() + self.top_height)
            for ds in datasets]
        tslice = slice(row.start_time_period, row.end_time_period)

        # Crop slice from the datasets
        X_ds = [ds.sel(Time=tslice, Height=hslice)[profile]
                for ds, profile, hslice in zip(datasets, self.profiles, hslices)]

        return X_ds

    def load_Y(self, idx):
        """
        Returns Y features for estimation
        :param idx: index of the sample
        :return: pandas.core.series.Series
        """

        row = self.data.loc[idx, :]
        Y = row[self.Y_features]
        return Y

    def get_key_val(self, idx, key='idx'):
        """

        :param idx: index of the sample
        :param key: key is any of self.key values. e.g, 'idx', 'wavelength' etc...
        :return: The values of the required key for the sample
        """
        row = self.data.loc[idx, :]
        key_val = row[self.key.index(key)]
        return key_val


class LidarDataModule(LightningDataModule):
# TODO - This is failing :(
    def __init__(self, train_csv_path, test_csv_path,
                 powers, X_features_profiles, Y_features, batch_size, num_workers,
                 val_length=0.2, test_length=0.2):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.powers = powers
        self.X_features_profiles = X_features_profiles
        self.Y_features = Y_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_length = val_length
        self.test_length = test_length

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage=None):
        # called on every GPU

        # Step 1. Load Dataset
        # TODO: add option - y = {bin(r0),bin(r1)}
        transformations_list = [PowTransform(self.powers), LidarToTensor()] if self.powers else [LidarToTensor()]
        lidar_transforms = torchvision.transforms.Compose(transformations_list)

        if stage == 'fit' or stage is None:
            trainable_dataset = LidarDataSet(csv_file=self.train_csv_path, transform=lidar_transforms, top_height=15.3,
                                             X_features_profiles=self.X_features_profiles, Y_features=self.Y_features)

            self.train, self.val = trainable_dataset.get_splits(n_val=self.val_length, n_test=0)

        if stage == 'test' or stage is None:
            self.test = LidarDataSet(csv_file=self.test_csv_path, transform=lidar_transforms, top_height=15.3,
                                     X_features_profiles=self.X_features_profiles, Y_features=self.Y_features)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
