import numpy as np
import pandas as pd
import torch
import torchvision
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from learning_lidar.preprocessing import preprocessing as prep
from learning_lidar.learning_phase.utils_.custom_operations import PowTransform, SampleXR2Tensor


class LidarDataSet(torch.utils.data.Dataset):
    """
    TODO: add usage
    """

    def __init__(self, dataset_csv_file, transform, torch_transforms, top_height,
                 X_features, profiles, Y_features, filter_by, filter_values):
        """

        :param dataset_csv_file: string, Path to the csv file of the database
        :param transform:
        :param transform:
        :param top_height: np.float(). The Height[km] **above** ground (Lidar) level - up to which slice the samples.
        Note: default is 15.3 [km]. IF ONE CHANGES IT - THAN THIS WILL AFFECT THE INPUT DIMENSIONS AND STATISTICS !!!
        :param X_features:
        :param profiles:
        :param Y_features:
        :param filter_by: string, should be one of the features names in the data. E.g. 'wavelength' / 'date' / ...
        :param filter_values: list, values to keep. E.g. [355,
        1064] for wavelengths, ['9/1/2017', '9/2/2017', '9/5/2017',...] for  dates
        """

        self.data = pd.read_csv(dataset_csv_file)
        self.key = list(self.data.keys())
        if filter_by:
            if filter_by not in self.key:
                raise KeyError(f'{filter_by} is not a valid feature name in the data. Should be one of {self.key}')
            self.data = self.data.loc[self.data[filter_by].isin(filter_values)]
            if self.data.empty:
                raise Exception('Dataframe is empty! Make sure filter_by and filter_values are correct.')
        self.X_features = X_features
        self.profiles = profiles
        self.Y_features = Y_features
        self.top_height = top_height
        self.transform = transform
        self.torch_transforms = torch_transforms

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
            if self.torch_transforms:
                sample['x'] = self.torch_transforms(sample['x'])
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
        key_val = row[key]
        return key_val


class LidarDataModule(LightningDataModule):
    def __init__(self, train_csv_path, test_csv_path, stats_csv_path,
                 powers, top_height, X_features_profiles, Y_features, batch_size, num_workers,
                 val_length=0.2, test_length=0.2, data_filter=None, data_norm=False):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.stats_csv_path = stats_csv_path
        self.powers = powers
        self.top_height = top_height
        self.X_features, self.profiles = map(list, zip(*X_features_profiles))  # unpack list of tuples into two lists
        self.Y_features = Y_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_length = val_length
        self.test_length = test_length
        if data_filter:
            self.filter_by = data_filter[0]
            self.filter_values = data_filter[1]
        else:
            self.filter_by = None
            self.filter_values = None
        self.data_norm = data_norm
        self.stats = self.calc_stats()

    def calc_stats(self):
        stats_df = pd.read_csv(self.stats_csv_path)
        if self.filter_by:
            if self.filter_by == 'wavelength':
                stats_df = stats_df.loc[stats_df[self.filter_by].isin(self.filter_values)]
            if stats_df.empty:
                raise Exception('Dataframe is empty! Make sure filter_by and filter_values are correct.')

        sources = [x_feature.split('_path')[0] for x_feature in self.X_features]
        X_mean = [stats_df[f"{profile}_{source}_mean"].mean() for profile, source in zip(self.profiles, sources)]
        X_std = [stats_df[f"{profile}_{source}_std"].mean() for profile, source in zip(self.profiles, sources)]
        Y_mean = [stats_df[f"{y_feature}_mean"].mean() for y_feature in self.Y_features]
        Y_std = [stats_df[f"{y_feature}_std"].mean() for y_feature in self.Y_features]
        stats = {'x': {'mean': X_mean, 'std': X_std}, 'y': {'mean': Y_mean, 'std': Y_std}}

        if self.powers:
            stats = self.pow_stats(stats)

        return stats

    def pow_stats(self, stats):
        if self.powers:
            x_powers = [self.powers[profile] for profile in self.profiles]
            y_powers = [self.powers[feature] for feature in self.Y_features]
            for op in ['mean', 'std']:
                stats['x'][op] = [val_i ** pow_i for val_i, pow_i in zip(stats['x'][op], x_powers)]
                stats['y'][op] = [val_i ** pow_i for val_i, pow_i in zip(stats['y'][op], y_powers)]
        return stats

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage=None):
        # called on every GPU

        # Step 1. Load Dataset
        # TODO: add option - y = {bin(r0),bin(r1)}
        transformations_list = [PowTransform(self.Y_features, self.profiles, self.powers), SampleXR2Tensor()] \
            if self.powers else [SampleXR2Tensor()]
        lidar_transforms = torchvision.transforms.Compose(transformations_list)
        torch_transforms = torchvision.transforms.Normalize(mean=tuple(self.stats['x']['mean']),
                                                            std=tuple(self.stats['x']['std'])) \
            if self.data_norm else None

        if stage == 'fit' or stage is None:
            trainable_dataset = LidarDataSet(dataset_csv_file=self.train_csv_path,
                                             transform=lidar_transforms,torch_transforms=torch_transforms,
                                             top_height=self.top_height, X_features=self.X_features,
                                             profiles=self.profiles, Y_features=self.Y_features,
                                             filter_by=self.filter_by, filter_values=self.filter_values)

            self.train, self.val = trainable_dataset.get_splits(n_val=self.val_length, n_test=0)

        if stage == 'test' or stage is None:
            self.test = LidarDataSet(dataset_csv_file=self.test_csv_path,
                                     transform=lidar_transforms,torch_transforms=torch_transforms,
                                     top_height=self.top_height, X_features=self.X_features, profiles=self.profiles,
                                     Y_features=self.Y_features, filter_by=self.filter_by,
                                     filter_values=self.filter_values)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
