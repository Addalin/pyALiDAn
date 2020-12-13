#!/usr/bin/env python
# MIT License
# Copyright (c) 2020  Adi Vainiger
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Modules for physical and pollyXT constants.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass

# %% Basic physics constants

eps = np.finfo(np.float).eps
C_km_s = 299792.458  # Speed of light [Km/sec]
C_m_s = C_km_s * 1E+3  # Speed of light [m/sec]
h_plank = 6.62606e-34  # plank constant [J sec]

# %% pollyXT Lidar info
n_chan = 13


# %% Haifa station info


@dataclass()
class Station:
    def __init__(self, station_name='haifa', stations_csv_path='stations.csv'):
        """
        A station class that stores all the below information

        :param station_name: str, should match a station_name in the csv file
        :param stations_csv_path: str, path to stations csv file
        """

        stations_df = pd.read_csv(stations_csv_path, index_col='station_name', sep=',',skipinitialspace=True)
        try:
            station_df = stations_df.loc[station_name.lower()]
        except KeyError as e:
            print(f"{station_name.lower()} not in {stations_csv_path}. Available stations: {stations_df.index.values}")
            raise e
        self.name = station_name
        self.location = station_df['location']
        self.lon = np.float(station_df['longitude'])
        self.lat = np.float(station_df['latitude'])
        self.altitude = np.float(station_df['altitude'])  # [m] The Lidar's altitude   ( above sea level, see 'altitude' in ' *_att_bsc.nc)
        self.start_bin_height = np.float(station_df['start_bin_height'])  # [m] The first bin's height ( above ground level - a.k.a above the lidar, see height[0]  in *_att_bsc.nc)
        self.end_bin_height = np.float(station_df['end_bin_height'])  # [m] The last bin's height  ( see height[-1] in *_att_bsc.nc)
        self.n_bins = np.int(station_df['n_bins'])      # [#] Number of height bins         ( see height.shape  in  *_att_bsc.nc)
        self.dt = np.float(eval(station_df['dt']))     # [sec] temporal pulse width of the lidar
        self.gdas1_folder = station_df['gdas1_folder']
        self.gdastxt_folder = station_df['gdastxt_folder']
        self.lidar_src_folder = station_df['lidar_src_folder']
        self.molecular_src_folder = station_df['molecular_src_folder']
        self.db_file = station_df['db_file']
        self.lidar_parent_folder = station_df['lidar_parent_folder']


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


#### TODO: station parameters load from station_info.csv
#### TODO: create pre-proc class, having the following internal fields:
#### location, lat, lon, gdas_name_format, gdas_txt_format, min_height, top_height, h_bins,


class CHANNELS():
    def __init__(self):
        pass

    ''' Class of pollyXT lidar channel numbers '''
    UV = 0  # UV channel - 355[nm]
    UVs = 1  # UV polarized chnanel - 355 [nm]
    V_1 = 2  # V_1 Ramman channel of Nitrogen N2 - 387[nm]
    V_2 = 3  # V_2 Ramman channel of water H2O - 407[nm]
    G = 4  # Green channel - 532[nm]
    Gs = 5  # Green channel - 532[nm]
    R = 6  # Red Raman channel - 607[nm]
    IR = 7  # IR channel - 1064[nm]
    GNF = 8  # Near Field green channel - 532[nm]
    RNF = 9  # Near Field red channel - 607[nm]
    UVNF = 10  # Near Field UV channel - 355[nm]
    V1NF = 11  # Near field Ramman channel - 387[nm]


class LAMBDA_nm( object ):
    def __init__(self, scale=1):
        # pass
        """ Class of pollyXT lidar wavelengths, values are in micro meters
        :param scale: unit scaling to [nm]
        """
        self.UV = 355 * scale  # UV channel - 355[nm]
        self.V_1 = 387 * scale  # V_1 Ramman channel of Nitrogen N2 - 386[nm]
        self.V_2 = 407 * scale  # V_2 Ramman channel of water H2O - 407[nm]
        self.G = 532 * scale  # Green channel - 532[nm]
        self.R = 607 * scale  # Red Raman channel - 607[nm]
        self.IR = 1064 * scale  # IR channel - 1064[nm]

    def get_elastic(self):
        return [self.UV, self.G, self.IR]

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

class LAMBDA_m( LAMBDA_nm ):
    def __init__(self):
        LAMBDA_nm.__init__( self , 1E-9 )


# %%DEBUG -----------------------------
if __name__ == '__main__':
    print('This files contains some useful constants')
    wavelengths = LAMBDA_nm( )
    print(wavelengths)
    wavelengths_m = LAMBDA_m()
    print(wavelengths_m.get_elastic())
    haifa_station = Station()
    print(haifa_station)
