# %% Imports

import pandas as pd
import os
from datetime import datetime, timedelta, time, date
import glob
import numpy as np
import learning_lidar.preprocessing.preprocessing_utils as prep_utils
import learning_lidar.utils.global_settings as gs
from learning_lidar.utils.utils import create_and_configer_logger
import logging
import xarray as xr
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from learning_lidar.preprocessing.fix_gdas_errors import download_from_noa_gdas_files
from zipfile import ZipFile


def preprocessing_main(station_name='haifa', start_date=datetime(2017, 9, 1), end_date=datetime(2017, 9, 2)):
    logging.getLogger('matplotlib').setLevel(logging.ERROR)  # Fix annoying matplotlib logs
    logging.getLogger('PIL').setLevel(logging.ERROR)  # Fix annoying PIL logs
    logger = create_and_configer_logger(f"{os.path.basename(__file__)}.log", level=logging.INFO)
    DOWNLOAD_GDAS = False
    CONV_GDAS = False
    GEN_MOL_DS = False
    GEN_LIDAR_DS = False
    GEN_LIDAR_DS_RAW = False
    USE_KM_UNITS = True
    UNZIP_TROPOS_LIDAR = True

    """set day,location"""
    station = gs.Station(station_name=station_name)
    logger.info(f"Loading {station.location} station")
    logger.debug(f"Station info: {station}")
    logger.info(
        f"\nStart preprocessing for period: [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Generate molecular datasets for required period'''
    if DOWNLOAD_GDAS:
        # TODO Test that this works as expected
        download_from_noa_gdas_files(
            dates_to_retrieve=pd.date_range(start=start_date, end=end_date, freq=timedelta(days=1)),
            save_folder=station.gdas1_folder)

    gdas_paths = []
    if CONV_GDAS:
        # Convert gdas files for a period
        gdas_paths.extend(convert_periodic_gdas(station, start_date, end_date))

    # get all days having a converted (to txt) gdas files in the required period
    if (GEN_MOL_DS or GEN_LIDAR_DS) and not gdas_paths:
        logger.debug('\nGet all days in the required period that have a converted gdas file')
        dates = pd.date_range(start=start_date, end=end_date, freq='D').to_pydatetime().tolist()
        for day in tqdm(dates):
            _, curpath = prep_utils.get_daily_gdas_paths(station, day, f_type='txt')
            if curpath:
                gdas_paths.extend(curpath)
        timestamps = [prep_utils.get_gdas_timestamp(station, path) for path in gdas_paths]
        df_times = pd.DataFrame(data=timestamps, columns=['timestamp'])
        days_g = df_times.groupby([df_times.timestamp.dt.date]).groups
        valid_gdas_days = list(days_g.keys())

    if GEN_MOL_DS:
        # TODO: Check for existing molecular paths, to avoid creation for them (Since it takes long time to generate these datasest)
        '''molpaths = [ ]
        for day_date in valid_gdas_days:
            cpath = get_prep_dataset_paths ( station = station ,
                                             day_date = day_date ,
                                             lambda_nm = '*',
                                             file_type = 'attbsc')
            if cpath:
                molpaths.extend(cpath)

        timestamps = [
            datetime.strptime ( '_'.join ( (os.path.basename ( cpath )).split ( '_' ) [ 0 :3 ] ) , '%Y_%m_%d' ) for
            cpath in molpaths ]
        df_times = pd.DataFrame ( data = timestamps , columns = [ 'timestamp' ] )
        days_g = df_times.groupby ( [ df_times.timestamp.dt.date ] ).groups
        mol_days = list ( days_g.keys ( ) )
        # separate exiting days from required period (not to genarate again)
        inds_exist = [ valid_gdas_days.index(mol_day)  for mol_day in mol_days ] 
        logger.debug ( f"Existing days of 'attbsc' profiles: {mol_days}" )
        '''
        # Generate and save molecular dataset for each valid day in valid_gdas_days :
        logger.info(
            f"Start generating molecular datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        len_mol_days = len(valid_gdas_days)
        num_processes = np.min((cpu_count() - 1, len_mol_days))
        chunksize = np.ceil(float(len_mol_days) / num_processes).astype(int)
        # TODO: add here tqdm
        with Pool(num_processes) as p:
            p.map(gen_daily_molecular_ds, valid_gdas_days, chunksize=chunksize)

        logger.info(
            f"\nFinished generating and saving of molecular datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

    ''' Extract all zip files of raw signals from TROPOS'''
    if UNZIP_TROPOS_LIDAR:
        """
        Expects all zip files to be in station.lidar_src_folder, and saves them under the appropriate year/month/day
        """
        path_pattern = os.path.join(station.lidar_src_folder, '*.zip')
        for file_name in sorted(glob.glob(path_pattern)):
            # extract day of the file. 0 - year, 1 - month, 2 - day
            nc_year, nc_month, nc_day = os.path.basename(file_name).split("_")[0:3]
            save_path = os.path.join(station.lidar_src_folder, nc_year, nc_month, nc_day)

            logger.info(f"extracting {file_name} to {save_path}")
            ZipFile(file_name).extractall(save_path)

    ''' Generate lidar datasets for required period'''
    if GEN_LIDAR_DS:
        lidarpaths = []
        logger.info(
            f"\nStart generating lidar datasets for period [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        for day_date in tqdm(valid_gdas_days):
            # Generate daily range corrected
            lidar_ds = get_daily_range_corr(station, day_date, height_units='km', optim_size=False, verbose=False,
                                            USE_KM_UNITS=USE_KM_UNITS)

            # Save lidar dataset
            lidar_paths = save_prep_dataset(station, lidar_ds, data_source='lidar', save_mode='single',
                                            profiles=['range_corr'])
            lidarpaths.extend(lidar_paths)
        logger.info(
            f"\nDone creation of lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        logger.debug(f'\nLidar paths: {lidarpaths}')
    ''' Obtaining lidar datasets for required period'''
    if GEN_LIDAR_DS_RAW:
        lidarpaths = []
        logger.info(
            f"\nStart obtaining raw lidar datasets for period [{start_date.strftime('%Y-%m-%d')},"
            f"{end_date.strftime('%Y-%m-%d')}]")

        for day_date in tqdm(valid_gdas_days):
            # Generate daily range corrected
            lidar_ds = get_daily_measurements(station, day_date, use_km_units=USE_KM_UNITS)

            # Save lidar dataset
            lidar_paths = save_prep_dataset(station, lidar_ds, data_source='lidar', save_mode='single',
                                            profiles=['range_corr'])
            lidarpaths.extend(lidar_paths)
        logger.info(
            f"\nDone creation of lidar datasets for period [{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}]")

        logger.debug(f'\nLidar paths: {lidarpaths}')


def save_dataset(dataset, folder_name='', nc_name='', nc_path=None, optim_size=True):
    """
    Save the input dataset to netcdf file

    :param nc_path: full path to netcdf file if already known
    :param dataset: array.Dataset()
    :param folder_name: folder name
    :param nc_name: netcdf file name
    :param optim_size: Boolean. False: the saved dataset will be type 'float64', True: the saved dataset will be type 'float64'(default).
    :return: nc_path - full path to netcdf file created if succeeded, else none

    """
    logger = logging.getLogger()
    if nc_path:
        folder_name, nc_name = os.path.dirname(nc_path), os.path.basename(nc_path)
    if optim_size:
        try:
            # Separate whether its a dataset (has data_vars) or dataarray
            if hasattr(dataset, 'data_vars'):
                for var in dataset.data_vars:
                    if dataset[var].dtype == np.float64 and len(dataset[var].dims) >= 2:
                        dataset[var] = dataset[var].astype(np.float32,
                                                           casting='same_kind',
                                                           copy=False,
                                                           keep_attrs=True)
            else:
                if dataset.dtype == np.float64 and len(dataset.dims) >= 2:
                    dataset = dataset.astype(np.float32,
                                             casting='same_kind',
                                             copy=False,
                                             keep_attrs=True)

            logger.debug(f"\nCasting float64 to float32 in file - {folder_name} {nc_name}")
        except Exception:
            logger.exception(f"\nFailed casting float64 to float32")
            return None
    if not os.path.exists(folder_name):
        try:
            os.makedirs(folder_name)
            logger.debug(f"\nCreating folder: {folder_name}")
        except Exception:
            logger.exception(f"\nFailed to create folder: {folder_name}")
            return None

    nc_path = os.path.join(folder_name, nc_name)
    try:
        dataset.to_netcdf(nc_path, mode='w', format='NETCDF4', engine='netcdf4')
        dataset.close()
        logger.debug(f"\nSaving dataset file: {nc_path}")
    except Exception:
        logger.exception(f"\nFailed to save dataset file: {nc_path}")
        nc_path = None
    return nc_path


def load_dataset(ncpath):
    """
    Load Dataset stored in the netcdf file path (ncpath)
    :param ncpath: a netcdf file path
    :return: xarray.Dataset, if fails return none
    """
    logger = logging.getLogger()
    try:
        dataset = xr.load_dataset(ncpath, engine='netcdf4').expand_dims()
        dataset.close()
        logger.debug(f"\nLoading dataset file: {ncpath}")
    except Exception as e:
        logger.exception(f"\nFailed to load dataset file: {ncpath}")
        raise e
    return dataset


def convert_periodic_gdas(station, start_day, end_day):
    logger = logging.getLogger()

    day_dates = pd.date_range(start=start_day, end=end_day, freq=timedelta(days=1))
    expected_file_no = len(day_dates) * 8  # 8 timestamps per day
    gdastxt_paths = []
    for day in day_dates:
        gdastxt_paths.extend(prep_utils.convert_daily_gdas(station, day))
    total_converted = len(gdastxt_paths)
    logger.debug(f"\nDone conversion of {total_converted} gdas files for period [{start_day.strftime('%Y-%m-%d')},"
                 f"{end_day.strftime('%Y-%m-%d')}], {(expected_file_no - total_converted)} failed.")
    return gdastxt_paths


def generate_daily_molecular(station, day_date, time_res='30S', height_units='km',
                             optim_size=False, verbose=False, USE_KM_UNITS=True):
    """
    Generating daily molecular profiles for all elastic channels (355,532,1064)
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param time_res: Output time resolution required. default is 30sec (according to time resolution of pollyXT measurements)
    :param height_units:  Output units of height grid in 'km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
                                True: the retrieved values are of type 'float'.
    :param verbose: Boolean. False(default). True: prints information regarding size optimization.
    :param USE_KM_UNITS: Boolean flag, to set the scale of units of the output data ,True - km units, False - meter units
    :return: xarray.Dataset() holding 5 data variables:
             3 daily dataframes: beta,sigma,att_bsc with shared dimensions(Height, Time, Wavelength)
             and 2 shared variables: lambda_nm with dimension (Wavelength), and date
    """

    date_datetime = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date

    wavelengths = gs.LAMBDA_nm().get_elastic()
    ds_list = []
    # t = TicToc()
    # t.tic()
    for lambda_nm in wavelengths:
        ds_chan = prep_utils.generate_daily_molecular_chan(station, date_datetime, lambda_nm, time_res=time_res,
                                                height_units=height_units, optim_size=optim_size,
                                                verbose=verbose)
        ds_list.append(ds_chan)
    # t.toc()
    '''concatenating molecular profiles of all channels'''
    mol_ds = xr.concat(ds_list, dim='Wavelength')
    mol_ds['date'] = date_datetime
    mol_ds.attrs = {'info': 'Daily molecular profiles',
                    'location': station.name,
                    'source_type': 'gdas'}
    if USE_KM_UNITS:
        mol_ds = prep_utils.convert_profiles_units(mol_ds, units=['1/m', '1/km'], scale=1e+3)

    return mol_ds


def get_daily_range_corr(station, day_date, height_units='km',
                         optim_size=False, verbose=False, USE_KM_UNITS=True):
    """
    Retrieving daily range corrected lidar signal (pr^2) from attenuated_backscatter signals in three channels (355,532,1064).

    The attenuated_backscatter are from 4 files of 6-hours *att_bsc.nc for a given day_date and station
    
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param height_units:  Output units of height grid in 'km' (default) or 'm'
    :param optim_size: Boolean. False(default): the retrieved values are of type 'float64',
                                True: the retrieved values are of type 'float'.
    :param verbose: Boolean. False(default). True: prints information regarding size optimization.
    :param USE_KM_UNITS: Boolean flag, to set the scale of units of the output data ,True - km units, False - meter units
    :return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
             1 daily dataset of range_corrected signal in 3 channels, with dimensions of : Height, Time, Wavelength
             3 variables : lambda_nm, plot_min_range, plot_max_range, with dimension of : Wavelength
             1 shared variable: date
    """

    """ get netcdf paths of the attenuation backscatter for given day_date"""
    date_datetime = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date

    bsc_paths = prep_utils.get_TROPOS_dataset_paths(station, date_datetime, file_type='att_bsc')
    bsc_ds0 = load_dataset(bsc_paths[0])
    altitude = bsc_ds0.altitude.values[0]
    profiles = [dvar for dvar in list(bsc_ds0.data_vars) if 'attenuated_backscatter' in dvar]
    wavelengths = [np.uint(pname.split(sep='_')[-1].strip('nm')) for pname in profiles]

    min_range = np.empty((len(wavelengths), len(bsc_paths)))
    max_range = np.empty((len(wavelengths), len(bsc_paths)))

    ds_range_corrs = []
    for ind_path, bsc_path in enumerate(bsc_paths):
        cur_ds = load_dataset(bsc_path)
        '''get 6-hours range corrected dataset for three channels [355,532,1064]'''
        ds_chans = []
        for ind_wavelength, (pname, lambda_nm) in enumerate(zip(profiles, wavelengths)):
            cur_darry = cur_ds.get(pname).transpose(transpose_coords=True)
            ds_chan, LC = prep_utils.get_range_corr_ds_chan(cur_darry, altitude, lambda_nm, height_units, optim_size,
                                                 verbose)
            min_range[ind_wavelength, ind_path] = LC * cur_darry.attrs['plot_range'][0]
            max_range[ind_wavelength, ind_path] = LC * cur_darry.attrs['plot_range'][1]
            ds_chans.append(ds_chan)

        cur_ds_range_corr = xr.concat(ds_chans, dim='Wavelength')
        ds_range_corrs.append(cur_ds_range_corr)

    '''merge range corrected of lidar through 24-hours'''
    range_corr_ds = xr.merge(ds_range_corrs, compat='no_conflicts')

    # Fixing missing timestamps values:
    time_indx = station.calc_daily_time_index(date_datetime)
    range_corr_ds = range_corr_ds.reindex({"Time": time_indx}, fill_value=0)
    range_corr_ds = range_corr_ds.assign({'plot_min_range': ('Wavelength', min_range.min(axis=1)),
                                          'plot_max_range': ('Wavelength', max_range.max(axis=1))})
    range_corr_ds['date'] = date_datetime
    range_corr_ds.attrs = {'location': station.location,
                           'info': 'Daily range corrected lidar signal',
                           'source_type': 'att_bsc'}
    if USE_KM_UNITS:
        range_corr_ds = prep_utils.convert_profiles_units(range_corr_ds, units=[r'$m^2$', r'$km^2$'], scale=1e-6)
    return range_corr_ds


def get_raw_lidar_signal(station: gs.Station, day_date: datetime.date, height_slice: slice, ds_attr: dict,
                         use_km_units: bool) -> xr.Dataset:
    """
        Retrieving daily raw lidar signal (p / bg) from attenuated_backscatter signals in three channels
     (355,532,1064).

    The attenuated_backscatter are from 4 files of 6-hours *.nc for a given day_date and station

    Height slice determines if it is background - slice(0, station.pt_bin) or
     p - slice(station.pt_bin, station.pt_bin + station.n_bins)

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param height_slice: slice object deterining the heights to keep
    :param ds_attr: dict, the attributes of the dataset
    :param use_km_units: datetime.date object of the required date
    :return: xarray.Dataset() a daily raw lidar signal, holding 5 data variables:
             1 daily dataset of background or raw lidar signal in 3 channels,
             with dimensions of : Height, Time, Wavelength
             1 variables : lambda_nm, with dimension of : Wavelength
             1 shared variable: date
    """
    date_datetime = datetime.combine(date=day_date, time=time.min) if isinstance(day_date, date) else day_date
    raw_paths = prep_utils.get_TROPOS_dataset_paths(station, date_datetime, file_type=None)

    profile = 'raw_signal'
    num_times = int(station.total_time_bins / 4)
    channel_ids = gs.CHANNELS().get_elastic()
    wavelengths = gs.LAMBDA_nm().get_elastic()
    all_times = station.calc_daily_time_index(date_datetime)
    heights_ind = station.calc_height_index(USE_KM_UNITS=use_km_units)

    dss = []
    for part_of_day_indx, bsc_path in enumerate(raw_paths):
        cur_ds = load_dataset(bsc_path)
        # get 6-hours range corrected dataset for three channels [355,532,1064]
        cur_darry = cur_ds.get(profile).transpose(transpose_coords=True)
        times = list(all_times)[num_times * part_of_day_indx:num_times * (part_of_day_indx + 1)]

        darray = cur_darry.sel(channel=channel_ids, height=height_slice)
        ''' Create p dataset'''
        ds_partial = xr.Dataset(
            data_vars={'p': (('Wavelength', 'Height', 'Time'), darray.values)},
            coords={'Height': heights_ind,
                    'Time': times,
                    'Wavelength': wavelengths})

        dss.append(ds_partial)

    # merge range corrected of lidar through 24-hours
    ds = xr.merge(dss, compat='no_conflicts')

    # Fixing missing timestamps values:
    ds = ds.reindex({"Time": all_times}, fill_value=0)

    ds.attrs = ds_attr
    ds.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    ds.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'}  

    ds['date'] = date_datetime

    if use_km_units:
        ds = prep_utils.convert_profiles_units(ds, units=[r'$m^2$', r'$km^2$'], scale=1e-6)

    return ds


def get_daily_measurements(station: gs.Station, day_date: datetime.date, use_km_units: bool = True) -> xr.Dataset:
    """
    Retrieving daily range corrected lidar signal (pr^2), background and raw lidar signal
     from attenuated_backscatter signals in three channels (355,532,1064).

    :param station: gs.station() object of the lidar station
    :param day_date: datetime.date object of the required date
    :param use_km_units: whether to use km units or not. If False  uses 'm'
    :return: xarray.Dataset() a daily range corrected lidar signal, holding 5 data variables:
             3 daily datasets of range_corrected signal, background and raw lidar signal in 3 channels,
             with dimensions of : Height, Time, Wavelength
             1 variables : lambda_nm, with dimension of : Wavelength
             1 shared variable: date
    """

    # Raw Lidar Signal Dataset
    pn_ds_attr = {'info': 'Raw Lidar Signal',
                  'long_name': r'$p$', 'name': 'pn',
                  'units': r'$\rm$' + r'$photons$',
                  'location': station.location, }

    pn_ds = get_raw_lidar_signal(station=station,
                                 day_date=day_date,
                                 height_slice=slice(station.pt_bin, station.pt_bin + station.n_bins),
                                 ds_attr=pn_ds_attr,
                                 use_km_units=use_km_units)

    # Raw Background Measurement Dataset
    bg_ds_attr = {'info': 'Raw Background Measurement',
                  'long_name': r'$<p_{bg}>$',
                  'units': r'$photons$',
                  'name': 'pbg'}
    # TODO should bg data also be called p?
    bg_ds = get_raw_lidar_signal(station=station,
                                 day_date=day_date,
                                 height_slice=slice(0, station.pt_bin),
                                 ds_attr=bg_ds_attr,
                                 use_km_units=use_km_units)

    bg_mean = bg_ds.mean(dim='Height')
    bg_ds = bg_mean.broadcast_like(pn_ds.p)

    # Raw Range Corrected Lidar Signal
    r2_ds = prep_utils.calc_r2_ds(station, day_date)
    pr2n_ds = (pn_ds.copy(deep=True) * r2_ds)  # calc_range_corr_measurement
    pr2n_ds.attrs = {'info': 'Raw Range Corrected Lidar Signal',
                     'long_name': r'$p$' + r'$\cdot r^2$', 'name': 'range_corr',
                     'units': r'$\rm$' + r'$photons$' + r'$\cdot km^2$',
                     'location': station.location, }
    pr2n_ds.Height.attrs = {'units': r'$km$', 'info': 'Measurements heights above sea level'}
    pr2n_ds.Wavelength.attrs = {'long_name': r'$\lambda$', 'units': r'$nm$'} 
    pr2n_ds['date'] = day_date

    # Daily raw lidar measurement from TROPOS.
    lidar_ds = xr.Dataset().assign(p=pn_ds, range_corr=pr2n_ds, p_bg=bg_ds)
    lidar_ds['date'] = day_date
    lidar_ds.attrs = {'location': station.location,
                      'info': 'Daily raw lidar measurement from TROPOS.',
                      'source_file': os.path.basename(__file__)}

    return lidar_ds


def get_prep_dataset_file_name(station, day_date, data_source='molecular',
                               lambda_nm='*', file_type='*', time_slice=None):
    """
     Retrieves file pattern name of preprocessed dataset according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm] or all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or 'all' (meaning the dataset contains several profile types)

    :return: dataset file name (netcdf) file of the data_type required per given day and wavelength, data_source and file_type
    """
    dt_str = day_date.strftime('%Y_%m_%d')
    if time_slice:
        dt_str += f"_{time_slice.start.strftime('%H%M%S')}_{time_slice.stop.strftime('%H%M%S')}"
    if lambda_nm == 'all':
        file_type = ''
    if file_type == '*' or lambda_nm == '*':  # * for lambd_nm - means any wavelength and profile
        # retrieves any file of this date
        file_name = f"{dt_str}_{station.location}_{file_type}_{lambda_nm}*{data_source}.nc"
    else:
        # this option is mainly to set new file names
        file_name = f"{dt_str}_{station.location}_{file_type}_{lambda_nm}_{data_source}.nc"
    file_name = file_name.replace('all', '').replace('__', '_').replace('__', '_')
    return file_name


def get_prep_dataset_paths(station, day_date, data_source='molecular', lambda_nm='*', file_type='*'):
    """
     Retrieves file paths of preprocessed datasets according to date, station, wavelength dataset source, and profile type.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the measuring date
    :param lambda_nm: wavelength [nm] e.g., for the green channel 532 [nm], or 'all' (meaning the dataset contains several profile types)
    :param data_source: string object: 'molecular' or 'lidar'
    :param file_type: string object: e.g., 'attbsc' for molecular_dataset or 'range_corr' for a lidar_dataset, or 'all' (meaning the dataset contains several profile types)

    :return: paths to all datasets netcdf files of the data_type required per given day and wavelength
    """
    if data_source == 'molecular':
        parent_folder = station.molecular_dataset
    elif data_source == 'lidar':
        parent_folder = station.lidar_dataset

    month_folder = prep_utils.get_month_folder_name(parent_folder, day_date)
    file_name = get_prep_dataset_file_name(station, day_date, data_source, lambda_nm, file_type)

    # print(os.listdir(month_folder))
    file_pattern = os.path.join(month_folder, file_name)

    paths = sorted(glob.glob(file_pattern))

    return paths


def gen_daily_molecular_ds(day_date):
    """
    Generating and saving a daily molecular profile.
    The profile is of type xr.Datatset().
    Having 3 variables: sigma (extinction) ,beta(backscatter) and attbsc(beta*exp(-2tau).
    Each profile have dimensions of: Wavelength, Height, Time.
    :param day_date: datetime.date object of the required day
    :return:
    """
    # TODO: Find a way to pass: optim_size, save_mode, USE_KM_UNITS
    #  as variables when running with multiprocessing.
    logger = logging.getLogger()
    optim_size = False
    save_mode = 'single'
    USE_KM_UNITS = True

    logger.debug(f"\nStart generation of molecular dataset for {day_date.strftime('%Y-%m-%d')}")
    station = gs.Station(station_name='haifa')
    # generate molecular dataset
    mol_ds = generate_daily_molecular(station, day_date,
                                      optim_size=optim_size, USE_KM_UNITS=USE_KM_UNITS)

    # save molecular dataset
    ncpaths = save_prep_dataset(station, mol_ds, data_source='molecular', save_mode=save_mode, profiles=['attbsc'])
    logger.debug(f"\nDone saving molecular datasets for {day_date.strftime('%Y-%m-%d')}, to: {ncpaths}")


def save_prep_dataset(station, dataset, data_source='lidar', save_mode='both',
                      profiles=None, time_slices=None):
    """
    Save the input dataset to netcdf file
    :param time_slices:
    :param profiles: The name of profile desired to be saved separately.
    If this name is not provided, then the first profile is automatically selected
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily preprocessed lidar or molecular signals.
    Having dimensions of : Wavelength, Height, Time.
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'molecular' - molecular dataset.
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' - saving both options
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    date_datetime = prep_utils.get_daily_ds_date(dataset)
    if data_source == 'lidar':
        base_folder = station.lidar_dataset
    elif data_source == 'molecular':
        base_folder = station.molecular_dataset
    month_folder = prep_utils.get_month_folder_name(base_folder, date_datetime)

    prep_utils.get_daily_ds_date(dataset)
    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []

    if save_mode in ['both', 'sep']:
        if not profiles:
            profiles = [list(dataset.data_vars)[0]]
        for profile in profiles:
            for wavelength in dataset.Wavelength.values:
                ds_profile = dataset.sel(Wavelength=wavelength)[profile]
                ds_profile['date'] = date_datetime
                if time_slices:
                    for time_slice in tqdm(time_slices,
                                           desc=f"Split and save time slices for: {profile}, {wavelength}"):
                        file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                                               lambda_nm=wavelength, file_type=profile,
                                                               time_slice=time_slice)
                        ds_slice = ds_profile.sel(Time=time_slice)
                        ncpath = save_dataset(ds_slice, month_folder, file_name)
                        if ncpath:
                            ncpaths.append(ncpath)
                else:
                    file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                                           lambda_nm=wavelength, file_type=profile)
                    ncpath = save_dataset(ds_profile, month_folder, file_name)
                    if ncpath:
                        ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_prep_dataset_file_name(station, date_datetime, data_source=data_source,
                                               lambda_nm='all', file_type='all')
        ncpath = save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths


if __name__ == '__main__':
    station_name = 'haifa_shubi'
    start_date = datetime(2017, 9, 1)
    end_date = datetime(2017, 9, 1)
    preprocessing_main(station_name, start_date, end_date)
    # TODO: Add flags as args.
