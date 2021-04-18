import learning_lidar.preprocessing.preprocessing as prep


def get_gen_dataset_file_name(station, day_date, wavelength='*', data_source='lidar', file_type='range_corr'):
    """
     Retrieves file pattern name of generated lidar dataset according to:
      station, date, wavelength.
    :param station: gs.station() object of the lidar station
    :param day_date: datetime.datetime object of the generated date
    :param wavelength: wavelength [nm] e.g., for the green channel 532 [nm] or all (meaning the dataset contains all elastic wavelengths)
    :param data_source: string object: 'aerosol' or 'lidar'
    :param file_type: string object: e.g., 'range_corr'/'lidar_power' for separated files per wavelength (355,532, or 1064) or 'lidar'/'aerosol' for all wavelengths

    :return: dataset file name (netcdf) file of the data_type required per given day and wavelength, data_source and file_type
    """
    if wavelength == '*':
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{data_source}.nc"
    else:
        file_name = f"{day_date.strftime('%Y_%m_%d')}_{station.location}_generated_{file_type}_{wavelength}_{data_source}.nc"

    return file_name


def save_generated_dataset(station, dataset, data_source='lidar', save_mode='both'):
    """
    Save the input dataset to netcdf file
    :param station: station: gs.station() object of the lidar station
    :param dataset: array.Dataset() a daily generated lidar signal, holding 5 data variables:
             4 daily dataset, with dimensions of : Height, Time, Wavelength.
              name of profiles: 'range_corr','range_corr_p', 'lidar_sig','lidar_sig_p'
             1 shared variable: date
    :param data_source: source type of the file, i.e., 'lidar' - for lidar dataset, and 'aerosol' - aerosols dataset.
    :param save_mode: save mode options:
                    'sep' - for separated profiles (each is file is per profile per wavelength)
                    'single' - save the dataset a single file per day
                    'both' -saving both options
    :return: ncpaths - the paths of the saved dataset/s . None - for failure.
    """
    date_datetime = prep.get_daily_ds_date(dataset)
    if data_source == 'lidar':
        base_folder = station.gen_lidar_dataset
    elif data_source == 'aerosol':
        base_folder = station.gen_aerosol_dataset
    else:
        base_folder = station.generation_folder
    month_folder = prep.get_month_folder_name(base_folder, date_datetime)

    prep.get_daily_ds_date(dataset)
    '''save the dataset to separated netcdf files: per profile per wavelength'''
    ncpaths = []

    # NOTE: Currently saving to separated profiles is only for `range_corr_p` - used in the learning phase.cur_day
    # if one needs other separated profile, add it as an an input term.
    profile = list(dataset.data_vars)[1]
    if save_mode in ['both', 'sep']:
        for wavelength in dataset.Wavelength.values:
            ds_profile = dataset.sel(Wavelength=wavelength)[profile]
            ds_profile['date'] = date_datetime
            file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                                  wavelength=wavelength, file_type=profile)
            ncpath = prep.save_dataset(ds_profile, month_folder, file_name)
            if ncpath:
                ncpaths.append(ncpath)

    '''save the dataset to a single netcdf'''
    if save_mode in ['both', 'single']:
        file_name = get_gen_dataset_file_name(station, date_datetime, data_source=data_source,
                                              wavelength='*')
        ncpath = prep.save_dataset(dataset, month_folder, file_name)
        if ncpath:
            ncpaths.append(ncpath)
    return ncpaths
