import torch
import numpy as np
import xarray as xr

class PowTransform(object):
    def __init__(self, Y_features, profiles, powers={'range_corr': 0.5, 'attbsc': 0.5,
                                                     'LC': 0.5, 'LC_std': 0.5, 'r0': 1.0, 'r1': 1.0, 'dr': 1.0}):
        # TODO: Pass Y_features to the constructor
        self.Y_features = Y_features
        self.profiles = profiles
        self.X_powers = [powers[profile] for profile in self.profiles]
        self.Y_powers = [powers[feature] for feature in self.Y_features]

    def __call__(self, sample):
        X, Y = sample['x'], sample['y']
        X = [self.pow_X(x_i, pow_i) for (x_i, pow_i) in zip(X, self.X_powers)]
        Y = self.pow_Y(Y)
        return {'x': X, 'y': Y}

    def pow_X(self, x_i, pow_i):
        """

        :param x_i: xr.dataset: a lidar or a molecular dataset
        :return: The dataset is raised (shrink in this case) by the powers set.
        Acts similarly to gamma correction aims to reduce the input values.
        """
        # trim negative values
        x_i = x_i.where(x_i >= 0, np.finfo(np.float).eps)
        # apply power - using apply_ufunc function to accelerate
        x_i = xr.apply_ufunc(lambda x: x ** pow_i, x_i, keep_attrs=True)
        return x_i

    def pow_Y(self, Y):
        """

        :param Y: pandas.core.series.Series of np.float values to be estimates (as LC, ro, r1)
        :return: The values raised by the relevant powers set.
        """
        return [y_i ** pow for (pow, y_i) in zip(self.Y_powers, Y)]


class TrimNegative(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        X, Y = sample['x'], sample['y']
        # trim negative values
        X = [x_i.where(x_i >= 0, np.finfo(np.float).eps) for x_i in X]
        return {'x': X, 'y': Y}


class SampleXR2Tensor(object):
    """Convert a lidar sample {x,y}  to Tensors."""

    def __call__(self, sample):
        X, Y = sample['x'], sample['y']

        # convert X from xr.dataset to concatenated a np.ndarray, and then to torch.tensor
        X = torch.dstack([torch.from_numpy(X[i].values) for i in range(len(X))])

        # swap channel axis
        # numpy image: H x W x C
        # torch image: C X H X W
        X = X.permute(2, 0, 1)

        # convert Y from pd.Series to np.array, and then to torch.tensor
        Y = torch.from_numpy(np.array(Y).astype(np.float32))

        return {'x': X, 'y': Y}
    # TODO : apply poisson noise (miscLidar.generate_poisson_signal(mu, n)), add gaussian noise for augmentation
