import warnings

warnings.filterwarnings ( "ignore" )
import pandas as pd
import os
from datetime import datetime , timedelta , time , date
import glob
import numpy as np
import re
from netCDF4 import Dataset
import sqlite3
import fnmatch
import matplotlib.pyplot as plt
import global_settings as gs
import torch , torchvision
import torch.utils.data
import xarray as xr
import matplotlib.dates as mdates
from pytictoc import TicToc
from multiprocessing import Pool , cpu_count
import preprocessing as prep



class lidarDataSet ( torch.utils.data.Dataset ) :
    """TODO"""

    def __init__ ( self , csv_file , transform = None,
                   top_height = 15.0 ) :
        """
        Args:
            csv_file (string): Path to the csv file of the database.
            :param transform:
        """
        self.data = pd.read_csv ( csv_file )
        self.key = [ 'date' , 'wavelength' , 'cali_method' , 'telescope' , 'cali_start_time' , 'cali_stop_time' ,
                     'start_time_period' , 'end_time_period' ]
        self.Y_features = [ 'LC' , 'r0' , 'r1' ]  # , 'LC_std'
        self.X_features = [ 'lidar_path' , 'molecular_path' ]
        self.profiles = [ 'range_corr' , 'attbsc' ]
        self.top_height = top_height
        self.transform = transform

    def __len__ ( self ) :
        return len ( self.data )

    def __getitem__ ( self , idx ) :
        # load data
        X = self.load_X ( idx )
        Y = self.load_Y ( idx )

        sample = {'x' : X , 'y' : Y}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_splits( self, n_test = 0.2 ):
        test_size = round (n_test * len(self))
        train_size = len(self) - test_size
        train_set , test_set = torch.utils.data.random_split ( self , [ train_size , test_size ] )
        return train_set , test_set

    def load_X( self, idx ):
        """
        Returns X samples - measurements of lidar, and molecular
        :param idx: index of the sample
        :return: A list of two element each is of type xarray.core.dataarray.DataArray.
        0 - is for lidar measurements, 1 - is for molecular measurements
        """
        row = self.data.loc [ idx , : ]

        # Load X datasets
        X_paths = row [ self.X_features ]
        datasets = [ prep.load_dataset ( path ) for path in X_paths ]

        # Calc sample height and time slices
        hslices = [
            slice ( ds.Height.min ( ).values.tolist ( ) , ds.Height.min ( ).values.tolist ( ) + self.top_height )
            for ds in datasets ]
        tslice = slice ( row.start_time_period , row.end_time_period )

        # Crop slice from the datasets
        X_ds = [ ds.sel ( Time = tslice , Height = hslice ) [ profile ]
                 for ds , profile , hslice in zip ( datasets , self.profiles , hslices ) ]

        return X_ds

    def load_Y( self, idx ):
        """
        Returns Y features for estimation
        :param idx: index of the sample
        :return: pandas.core.series.Series
        """

        row = self.data.loc [ idx , : ]
        Y = row [ self.Y_features ]
        return Y



class PowTransform(object):
    def __init__( self,powers = {'range_corr' : 0.5 , 'attbsc' : 0.5 ,
                             'LC' : 0.5 , 'LC_std' : 0.5 , 'r0' : 1 , 'r1' : 1} ):
        self.Y_features = [ 'LC' , 'r0' , 'r1' ]  # , 'LC_std'
        self.profiles = [ 'range_corr' , 'attbsc' ]
        self.X_powers = [ powers [ profile ] for profile in self.profiles ]
        self.Y_powers = [ powers [ feature ] for feature in self.Y_features ]

    def __call__ ( self , sample ) :
        X , Y = sample [ 'x' ] , sample [ 'y' ]
        X = [ self.pow_X(x_i, pow_i) for (x_i, pow_i) in zip ( X, self.X_powers)]
        Y  = self.pow_Y(Y)
        return {'x' : X , 'y' : Y}

    def pow_X ( self , x_i, pow_i ) :
        """

        :param x_i: xr.dataset: a lidar or a molecular dataset
        :return: The dataset is raised (shrink in this case) by the powers set.
        Acts similarly to gamma correction aims to reduce the input values.
        """
        # trim negative values
        x_i = x_i.where ( x_i >= 0 , np.finfo ( np.float ).eps )
        # apply power - using apply_ufunc function to accelerate
        x_i = xr.apply_ufunc ( lambda x : x ** pow_i , x_i , keep_attrs = True )
        return x_i

    def pow_Y ( self , Y ) :
        """

        :param Y: pandas.core.series.Series of np.float values to be estimates (as LC, ro, r1)
        :return: The values raised by the relevant powers set.
        """
        return [ y_i ** pow for (pow , y_i) in zip ( self.Y_powers , Y ) ]


class ToTensor ( object ) :
    """Convert a lidar sample {x,y}  to Tensors."""

    def __call__ ( self , sample ) :
        X , Y = sample [ 'x' ] , sample [ 'y' ]


        # convert X from xr.dataset to concatenated a np.ndarray, and then to torch.tensor
        X = torch.dstack ( (torch.from_numpy ( X [ 0 ].values ) ,
                            torch.from_numpy ( X [ 1 ].values )) )
        # swap channel axis
        # numpy image: H x W x C
        # torch image: C X H X W
        X = X.permute ( 2 , 0 , 1 )

        # convert Y from pd.Series to np.array, and then to torch.tensor
        Y = torch.from_numpy ( np.array ( Y ).astype ( float ) )

        return {'x' : X ,'y' : Y}


def main( station_name = 'haifa' , start_date = datetime ( 2017 , 9 , 1 ) , end_date = datetime ( 2017 , 9 , 2 ) ):

    # Step 1. Load Dataset
    csv_path = f"dataset_{station_name}_{start_date.strftime ( '%Y-%m-%d' )}_{end_date.strftime ( '%Y-%m-%d' )}.csv"
    powers = {'range_corr' : 0.5 , 'attbsc' : 0.5 , 'LC' : 0.5 , 'LC_std' : 0.5 , 'r0' : 1 , 'r1' : 1}
    lidar_transforms = torchvision.transforms.Compose([PowTransform(powers),ToTensor()])
    dataset = lidarDataSet ( csv_path , lidar_transforms )
    train_set , test_set = dataset.get_splits(n_test = 0.2)



    # Step 2. Split and Make Datasets Iterable


    train_loader = torch.utils.data.DataLoader (
        train_set , batch_size = 4 , shuffle = True , num_workers = 7 )

    test_loader = torch.utils.data.DataLoader (
        test_set , batch_size = 4 , shuffle = False , num_workers = 7 )


    for i_batch , sample_batched in enumerate ( train_loader ) :
        print ( i_batch , sample_batched )

    for i_batch , sample_batched in enumerate ( test_loader ) :
        print ( i_batch , sample_batched )



if __name__ == '__main__' :
    station_name = 'haifa'
    start_date = datetime ( 2017 , 9 , 1 )
    end_date = datetime ( 2017 , 10 , 31 )
    main ( station_name , start_date , end_date )