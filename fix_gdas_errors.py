from time import sleep

import ARLreader as Ar
from datetime import datetime
import logging
import csv

from utils import create_and_configer_logger


def extract_dates_to_retrieve(failed_gdas_files_path='gdas2radiosonde_failed_files.csv', mark_as_downloaded=False):
    """
    Extracts a set of datetimes from the failed_gdas_files_path csv file.

    :param failed_gdas_files_path: str, path to csv file.
                                    Expects format to be with header,
                                    then each entry e.g: /gdas/2017/09/haifa_20170901_00_32.8_35.0.gdas1,Conversion Fail

    :return: set, each entry is a datetime, e.g. datetime(2017,09,17)
    """
    logger = logging.getLogger()
    dates_to_retrieve = set()
    with open(failed_gdas_files_path, 'r') as failed_files:
        reader = csv.reader(failed_files)
        file_data = [['gdas_source_file', 'Failure Reason', 'status']]
        for indx, failed_file in enumerate(reader):
            if indx == 0:
                continue
            path = failed_file[0]
            corruption_reason = failed_file[1]
            status = failed_file[2]
            if status == 'Broken':
                YYYYMMDD = path.split('/')[-1].split('_')[1]
                yearmonthday_to_retrieve = datetime.strptime(YYYYMMDD, '%Y%m%d')
                dates_to_retrieve.add(yearmonthday_to_retrieve)
            if mark_as_downloaded:
                status = 'Downloaded'
            file_data.append([path, corruption_reason, status])

    with open(failed_gdas_files_path, 'w') as failed_files:
        writer = csv.writer(failed_files)
        writer.writerows(file_data)

    logger.debug(
        f"Extracted {len(dates_to_retrieve)} failed gdas files from dates: {[date_.strftime('%d-%m-%Y') for date_ in dates_to_retrieve]}. mark_as_fixed? {mark_as_fixed}")
    return dates_to_retrieve


def download_gdas_files(dates_to_retrieve, save_folder='downloaded_gdas'):
    """
    Downloads all gdas files matching dates_to_retrieve from NOA.

    :param dates_to_retrieve: iterable, datetimes to get gdas files for
    :param save_folder: str, where to save the files
    """
    logger = logging.getLogger()
    downloader = Ar.Downloader(saveFolder=save_folder)
    for gdas_date in dates_to_retrieve:
        logger.debug(f"Starting download gdas {gdas_date.strftime('%d-%m-%Y')} from NOA")
        downloader.download(gdas_date)
        logger.debug(f"Complete download gdas {gdas_date.strftime('%d-%m-%Y')} from NOA to {save_folder}")


def extract_single_profile(failed_gdas_files_path='gdas2radiosonde_failed_files.csv'):
    gdas_file = Ar.fname_from_date(datetime(2019, 4, 17))
    print('name of input file ', gdas_file)
    gdas = Ar.reader('downloaded_gdas/'+gdas_file)
    profile, sfcdata, indexinfo, ind = gdas.load_profile(17, 12, (32.8, 35.0))
    Ar.write_profile('testfile.txt', indexinfo, ind, (32.8, 35.0), profile, sfcdata)

    logger = logging.getLogger()
    dates_to_retrieve = set()
    with open(failed_gdas_files_path, 'r') as failed_files:
        reader = csv.reader(failed_files)
        file_data = [['gdas_source_file', 'Failure Reason', 'status']]
        for indx, failed_file in enumerate(reader):
            if indx == 0:
                continue
            path = failed_file[0]
            corruption_reason = failed_file[1]
            status = failed_file[2]
            if status == 'Downloaded':
                parsed_path = path.split('/')[-1]
                station_name = parsed_path[0]
                date = parsed_path[1]
                hour = parsed_path[2]
                lat = parsed_path[3]
                lon, ending = parsed_path[4].split('.')
                new_file_name = f"{station_name}_"
                YYYYMMDD = path.split('/')[-1].split('_')[1]
                yearmonthday_to_retrieve = datetime.strptime(YYYYMMDD, '%Y%m%d')
                dates_to_retrieve.add(yearmonthday_to_retrieve)
            if mark_as_fixed:
                status = 'Downloaded'
            file_data.append([path, corruption_reason, status])

    with open(failed_gdas_files_path, 'w') as failed_files:
        writer = csv.writer(failed_files)
        writer.writerows(file_data)

    logger.debug(
        f"Extracted {len(dates_to_retrieve)} failed gdas files from dates: {[date_.strftime('%d-%m-%Y') for date_ in dates_to_retrieve]}. mark_as_fixed? {mark_as_fixed}")
    return dates_to_retrieve




if __name__ == '__main__':

    #logging.getLogger('ARLreader').setLevel(logging.ERROR)  # Fix annoying ARLreader logs
    logger = create_and_configer_logger('preprocessing_log.log')
    _ = extract_dates_to_retrieve(mark_as_fixed=False)
    sleep(0.1)
    flag = input("Are you sure you want to download all these files? [y/n]")
    if flag == 'y':
        dates_to_retrieve = extract_dates_to_retrieve(mark_as_fixed=True)
        download_gdas_files(dates_to_retrieve)
