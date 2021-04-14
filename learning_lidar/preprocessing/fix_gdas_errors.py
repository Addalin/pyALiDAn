import os
from time import sleep

import ARLreader as Ar
from datetime import datetime
import logging
import csv

from utils import create_and_configer_logger


def extract_dates_to_retrieve(failed_gdas_files_path='gdas2radiosonde_failed_files.csv', mark_as_downloaded=False):
    """
    Extracts a set of datetimes from the failed_gdas_files_path csv file.
    updates the file status as downloaded if mark_as_downloaded flag is True

    :param failed_gdas_files_path: str, path to csv file.
                                    Expects format to be with header,
                                    then each entry e.g: /gdas/2017/09/haifa_20170901_00_32.8_35.0.gdas1,Conversion Fail
    :param mark_as_downloaded: whether to mark the file as downloaded

    :return: set, each entry is a datetime, e.g. datetime(2017,09,17)
    """
    logger = logging.getLogger()
    dates_to_retrieve = set()
    with open(failed_gdas_files_path, 'r') as failed_files:
        reader = csv.reader(failed_files)
        file_data = [['gdas_source_file', 'failure_reason', 'status']]
        for indx, failed_file in enumerate(reader):
            if indx == 0 or not failed_file:
                continue
            path = failed_file[0]
            corruption_reason = failed_file[1]
            status = failed_file[2]
            if status == 'Broken':
                YYYYMMDD = path.split(os.sep)[-1].split('_')[1]
                yearmonthday_to_retrieve = datetime.strptime(YYYYMMDD, '%Y%m%d')
                dates_to_retrieve.add(yearmonthday_to_retrieve)
            if mark_as_downloaded:
                status = 'Downloaded'
            file_data.append([path, corruption_reason, status])

    with open(failed_gdas_files_path, 'w') as failed_files:
        writer = csv.writer(failed_files)
        writer.writerows(file_data)

    logger.debug(
        f"Extracted {len(dates_to_retrieve)} failed gdas files from dates: {[date_.strftime('%d-%m-%Y') for date_ in dates_to_retrieve]}. mark_as_downloaded? {mark_as_downloaded}")
    return dates_to_retrieve


def download_from_noa_gdas_files(dates_to_retrieve, save_folder='downloaded_gdas'):
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


def extract_profiles(failed_gdas_files_path='gdas2radiosonde_failed_files.csv'):
    """
    Coverts gdas1 files to relevant txt files by calling extract_single_profile,
    for each record in failed_gdas_files_path with 'status'=Downloaded

    Marks status ss Fixed for each such record
    :param failed_gdas_files_path: path to csv file
    """
    logger = logging.getLogger()
    with open(failed_gdas_files_path, 'r') as failed_files:
        reader = csv.reader(failed_files)
        file_data = [['gdas_source_file', 'Failure Reason', 'status']]
        for indx, failed_file in enumerate(reader):
            # skip first row
            if indx == 0 or not failed_file:
                continue
            # parse the csv row
            path = failed_file[0]
            corruption_reason = failed_file[1]
            status = failed_file[2]

            if status == 'Downloaded':
                logger.debug(f"Converting {path}...")
                parsed_path = path.split(os.sep)[-1].split('_')
                station_name = parsed_path[0]
                date = parsed_path[1]
                day = int(date[-2:])
                hour = int(parsed_path[2])
                lat = float(parsed_path[3])
                lon = float(parsed_path[4].rstrip('.gdas1'))
                YYYYMMDD = path.split(os.sep)[-1].split('_')[1]
                yearmonthday_to_retrieve = datetime.strptime(YYYYMMDD, '%Y%m%d')
                new_file_name = f"{station_name}_{date}_{hour}_{lat}_{lon}.txt"
                folder = os.path.dirname(path)
                extract_single_profile(day, hour, lat, lon, yearmonthday_to_retrieve, new_file_name,folder)
                status = 'Fixed'

            file_data.append([path, corruption_reason, status])

    with open(failed_gdas_files_path, 'w') as failed_files:
        writer = csv.writer(failed_files)
        writer.writerows(file_data)


def extract_single_profile(day, hour, lat, lon, yearmonthday_to_retrieve, new_file_name, save_path='converted_gdas'):
    """
    Conversion from gdas to txt, using ARLreader - reader, laad_profile and write profile
    :param day: int, the desired day
    :param hour: int, the desired hour
    :param lat: float, the latitude
    :param lon: float, the longitude
    :param yearmonthday_to_retrieve: datetime, includes they year, month and day
    :param new_file_name: str, name of converted text file
    :param save_path: where to save the converted gdas1 files
    """
    logger = logging.getLogger()

    gdas_file = Ar.fname_from_date(yearmonthday_to_retrieve)
    logger.debug(f'{gdas_file} converted to {os.path.join(save_path,new_file_name)}')
    gdas = Ar.reader(os.path.join('../../data/downloaded_gdas', gdas_file))
    profile, sfcdata, indexinfo, ind = gdas.load_profile(day, hour, (lat, lon))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    Ar.write_profile(os.path.join(save_path, new_file_name), indexinfo, ind, (lat, lon), profile, sfcdata)


if __name__ == '__main__':

    logging.getLogger('ARLreader').setLevel(logging.ERROR)  # Fix annoying ARLreader logs
    logger = create_and_configer_logger('preprocessing_log.log')
    download_from_noa = True
    if download_from_noa:
        # get list of 'Broken' dates from the csv file
        _ = extract_dates_to_retrieve(mark_as_downloaded=False)
        sleep(0.1)  # so log will show up before input below
        flag = input("Are you sure you want to download all these files? [y/n]")
        if flag == 'y':
            # extracts the list again, this marking the status as 'downloaded'
            dates_to_retrieve = extract_dates_to_retrieve(mark_as_downloaded=True)

            # downloads the files
            download_from_noa_gdas_files(dates_to_retrieve)

    # For files with 'Downloaded' status - converts to txt
    extract_profiles()
