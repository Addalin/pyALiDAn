import csv
import logging


def create_and_configer_logger(log_name):
    """
    TODO
    Args:
        log_name ():
    Returns:
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.DEBUG,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)
    return logger


def write_row_to_csv(csv_path, msg):
    """
    TODO
    Args:
        forward_dir ():
        msg ():
    Returns:
    """
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(msg)
    logger = logging.getLogger()
    logger.debug(f"Wrote to tracker{csv_path} message - {msg}")