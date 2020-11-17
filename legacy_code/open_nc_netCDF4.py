# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:45:31 2019

@author: hofer
"""


from netCDF4 import Dataset
file_path="C:\\Users\\hofer\\Documents\\haifa_170422\\2017_04_22_Sat_TROPOS_00_00_01.nc"
ncfile = Dataset(file_path,'r')
raw_signal355 = ncfile.variables['raw_signal'][:,:,0]
raw_signal532 = ncfile.variables['raw_signal'][:,:,4]
raw_signal1064 = ncfile.variables['raw_signal'][:,:,7]

##################
##SHAPE OF NC FILE
##################
#dimensions:
#	time = UNLIMITED ; // (717 currently)
#	height = 6400 ;
#	channel = 8 ;
#	date_time = 2 ;
#	coordinates = 2 ;
#	polynomial = 6 ;
#variables:
#	int raw_signal(time, height, channel) ;
#		raw_signal:long_name = "signal from the channels" ;
#		raw_signal:units = "counts" ;
#	int measurement_shots(time, channel) ;
#		measurement_shots:long_name = "shots per dataset" ;
#		measurement_shots:units = "counts" ;
#	int measurement_time(time, date_time) ;
#		measurement_time:long_name = "date and time of each dataset" ;
#		measurement_time:units = "YYYYMMDD and seconds of day" ;
#	float depol_cal_angle(time) ;
#		depol_cal_angle:long_name = "Angle of Polarizer for Depol " ;
#		depol_cal_angle:units = "degree" ;
#	float measurement_height_resolution ;
#		measurement_height_resolution:long_name = "HR of data acquisition" ;
#		measurement_height_resolution:units = "ns" ;
#	float laser_rep_rate ;
#		laser_rep_rate:long_name = "repetition rate of laser" ;
#		laser_rep_rate:units = "Hz" ;
#	float laser_power ;
#		laser_power:long_name = "mean pulse energy of laser" ;
#		laser_power:units = "mJ" ;
#	float laser_flashlamp ;
#		laser_flashlamp:long_name = "used flashlamp shots of laser" ;
#		laser_flashlamp:units = "counts" ;
#	float location_height ;
#		location_height:long_name = "location height asl" ;
#		location_height:units = "m" ;
#	float location_coordinates(time, coordinates) ;
#		location_coordinates:long_name = "latitude(-180..180) and longitude(-90..90)" ;
#		location_coordinates:units = "degree.minutes" ;
#	float neutral_density_filter(channel) ;
#		neutral_density_filter:long_name = "log_10 of attenuation of each channel" ;
#		neutral_density_filter:units = "1" ;
#	float if_center(channel) ;
#		if_center:long_name = "central wavelength of interference filters" ;
#		if_center:units = "nm" ;
#	float if_fwhm(channel) ;
#		if_fwhm:long_name = "FWHM of interference filters" ;
#		if_fwhm:units = "nm" ;
#	int polstate(channel) ;
#		polstate:long_name = "Polarization state of the channels. 0=total, 1=co, 2=cross." ;
#	int telescope(channel) ;
#		telescope:long_name = "Telescope for each channel. 0=far range, 1=near range." ;
#	float deadtime_polynomial(polynomial, channel) ;
#		deadtime_polynomial:long_name = "5-order polynomial coefficients for dead time correction at MCPS scale" ;
#		deadtime_polynomial:units = "1" ;
#	float deadtime_polynomial_error(polynomial, channel) ;
#		deadtime_polynomial_error:long_name = "error of dead time polynomial coefficients" ;
#		deadtime_polynomial_error:units = "1" ;
#	float discr_level(channel) ;
#		discr_level:long_name = "discriminator level" ;
#		discr_level:units = "mV" ;
#	float pm_voltage(channel) ;
#		pm_voltage:long_name = "photomultiplier voltage" ;
#		pm_voltage:units = "V" ;
#	float pinhole ;
#		pinhole:long_name = "diameter of pinhole" ;
#		pinhole:units = "mm" ;
#	float zenithangle ;
#		zenithangle:long_name = "Zenith angle of measurement" ;
#		zenithangle:units = "deg" ;