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
from dataclasses import dataclass
import os
from datetime import datetime, timedelta
import numpy as np
# %% Basic physics constants
import seaborn as sns
from matplotlib import pyplot as plt, dates as mdates

eps = np.finfo(float).eps
C_km_s = 299792.458  # Speed of light [Km/sec]
C_m_s = C_km_s * 1E+3  # Speed of light [m/sec]
h_plank = 6.62606e-34  # plank constant [J sec]

# %% pollyXT Lidar info
n_chan = 13


# %% Haifa station info


@dataclass()
class Station:
    def __init__(self, station_name='haifa',
                 stations_csv_path=
                 os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              'data',
                              'stations.csv')):
        """
        A station class that stores all the below information

        :param station_name: str, should match a station_name in the csv file
        :param stations_csv_path: str, path to stations csv file
        """

        stations_df = pd.read_csv(stations_csv_path, index_col='station_name', sep=',', skipinitialspace=True)
        try:
            station_df = stations_df.loc[station_name.lower()]
        except KeyError as e:
            print(f"{station_name.lower()} not in {stations_csv_path}. Available stations: {stations_df.index.values}")
            raise e
        self.name = station_name
        self.location = station_df['location']
        self.lon = float(station_df['longitude'])
        self.lat = float(station_df['latitude'])
        self.altitude = float(
            station_df['altitude'])  # [m] The Lidar's altitude   ( above sea level, see 'altitude' in ' *_att_bsc.nc)
        self.start_bin_height = float(station_df[
                                          'start_bin_height'])  # [m] The first bin's height ( above ground level - a.k.a above the lidar, see height[0]  in *_att_bsc.nc)
        self.end_bin_height = float(
            station_df['end_bin_height'])  # [m] The last bin's height  ( see height[-1] in *_att_bsc.nc)
        self.n_bins = int(
            station_df['n_bins'])  # [#] Number of height bins         ( see height.shape  in  *_att_bsc.nc)
        self.dt = float(eval(station_df['dt']))  # [sec] temporal pulse width of the lidar note: dr = C*dt/2
        self.freq = 30  # [sec] Frequency of measurments, currently every 30 sec, if this value changes, add it to the stations.csv
        self.total_time_bins = 2880  # Total measurment per day, currently 2880 time bins, if this value changes, add it to the stations.csv
        self.gdas1_folder = station_df['gdas1_folder']
        self.gdastxt_folder = station_df['gdastxt_folder']
        self.lidar_src_folder = station_df['lidar_src_folder']
        self.molecular_dataset = station_df['molecular_dataset']
        self.lidar_dataset = station_df['lidar_dataset']
        self.db_file = station_df['db_file']
        self.aeronet_folder = station_df['aeronet_folder']
        self.aeronet_name = station_df['aeronet_name']
        self.generation_folder = station_df['generation_folder']
        self.gen_lidar_dataset = station_df['gen_lidar_dataset']
        self.gen_signal_dataset = station_df['gen_signal_dataset']
        self.gen_aerosol_dataset = station_df['gen_aerosol_dataset']
        self.gen_bg_dataset = station_df['gen_bg_dataset']
        self.gen_density_dataset = station_df['gen_density_dataset']

    def __str__(self):
        return ("\n " + str(self.__class__) + ": " + str(self.__dict__)).replace(" {", "\n  {").replace(",", ",\n  ")

    def get_height_bins_values(self, USE_KM_UNITS=True):
        """
        Setting height vector above ground level
        (for lidar functions that uses height bins).
        :param USE_KM_UNITS: Boolean. True - get values in [km], False - get values in [m]
        :return: height_bins. np.array of height bins above ground level (distances of measurements relative to the sensor)
        """
        if USE_KM_UNITS:
            scale = 1E-3
        else:
            scale = 1
        min_height = self.start_bin_height
        top_height = self.end_bin_height
        height_bins = np.linspace(min_height * scale, top_height * scale, self.n_bins)
        # Note: another option:
        # dr = scale*gs.C_m_s*sel.dt/2.
        # height_bins = np.arange(min_height, top_height, step = dr)* scale
        # But this retrieves different values than TROPOS' netcdf-s height indexes.
        return height_bins

    def calc_height_index(self, USE_KM_UNITS=True):
        """
        Setting height vector above see level
        For interpolation of radiosonde / gdas files.
        And for Height indexes of xr.Dataset and pd.Dataframe objects
        :param USE_KM_UNITS: USE_KM_UNITS: Boolean. True - get values in [km], False - get values in [m]
        :return: heights. np.array of height bins above see level
        """
        if USE_KM_UNITS:
            scale = 1E-3
        else:
            scale = 1
        min_height = self.altitude + self.start_bin_height
        top_height = self.altitude + self.end_bin_height
        heights = np.linspace(min_height * scale, top_height * scale, self.n_bins)
        return heights

    def calc_daily_time_index(self, cur_day):
        end_t = cur_day + timedelta(hours=24) - timedelta(seconds=self.freq)
        time_index = pd.date_range(start=cur_day, end=end_t, freq=f'{self.freq}S')
        assert self.total_time_bins == len(time_index)
        return time_index


class CHANNELS():
    def __init__(self):
        ''' Class of pollyXT lidar channel numbers '''
        self.UV = 0  # UV channel - 355[nm]
        self.UVs = 1  # UV polarized channel - 355 [nm]
        self.V_1 = 2  # V_1 Raman channel of Nitrogen N2 - 387[nm]
        self.V_2 = 3  # V_2 Raman channel of water H2O - 407[nm]
        self.G = 4  # Green channel - 532[nm]
        self.Gs = 5  # Green channel - 532[nm]
        self.R = 6  # Red Raman channel - 607[nm]
        self.IR = 7  # IR channel - 1064[nm]
        self.GNF = 8  # Near Field green channel - 532[nm]
        self.RNF = 9  # Near Field red channel - 607[nm]
        self.UVNF = 10  # Near Field UV channel - 355[nm]
        self.V1NF = 11  # Near field Raman channel - 387[nm]

    def get_elastic(self):
        return [self.UV, self.G, self.IR]

    def __str__(self):
        return ("\n " + str(self.__class__) + ": " + str(self.__dict__)).replace(" {", "\n  {").replace(",", ",\n  ")


class LAMBDA_nm(object):
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
        return ("\n " + str(self.__class__) + ": " + str(self.__dict__)).replace(" {", "\n  {").replace(",", ",\n  ")


class LAMBDA_m(LAMBDA_nm):
    def __init__(self):
        LAMBDA_nm.__init__(self, 1E-9)


# %% VISUALIZATIONS
# TODO : move to vis_utils.py
FIGURE_DPI = 300
SAVEFIG_DPI = 300
SMALL_FONT_SIZE = 14
MEDIUM_FONT_SIZE = 16
BIG_FONT_SIZE = 18
TITLE_FONT_SIZE = 18
SUPTITLE_FONT_SIZE = 20
TIMEFORMAT = mdates.DateFormatter('%H:%M')
MONTHFORMAT = mdates.DateFormatter('%Y-%m')
COLORS = ["darkblue", "darkgreen", "darkred"]

def set_visualization_settings():
    # TODO : move to vis_utils.py
    # TODO make sure this actually propagates to other functions
    plt.rcParams['figure.dpi'] = FIGURE_DPI
    plt.rcParams['savefig.dpi'] = SAVEFIG_DPI

    # Create an array with the colors to use

    # Set a custom color palette
    sns.set_palette(sns.color_palette(COLORS))

    plt.rc('font', size=SMALL_FONT_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=TITLE_FONT_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_FONT_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_FONT_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_FONT_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SUPTITLE_FONT_SIZE)  # fontsize of the figure title


# %%DEBUG -----------------------------
if __name__ == '__main__':
    print('This files contains some useful constants')
    wavelengths = LAMBDA_nm()
    print(wavelengths)
    wavelengths_m = LAMBDA_m()
    print(wavelengths_m.get_elastic())
    haifa_station = Station()
    print(haifa_station)
