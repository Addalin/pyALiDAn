import csv
import logging


def create_and_configer_logger(log_name='log_file.log'):
    """
    Sets up a logger that works across files.
    The logger prints to console, and to log_name log file. 
    
    Example usage:
        In main function:
            logger = create_and_configer_logger(log_name='myLog.log')

        Then in all other files:
            logger = logging.getLogger(__name__)
            
        To add records to log:
            logger.debug(f"New Log Message. Value of x is {x}")
    
    Args:
        log_name: str, log file name
        
    Returns: logger
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
    Writes msg to a new row in csv_path which is a csv file.
    Logs message in logger.
    
    Args:
        csv_path: str, path to the csv file. Creates it if file doesn't exist.
        msg: list, message to write to file. Each entry in list represents a column.
    """
    with open(csv_path, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(msg)
    logger = logging.getLogger()
    logger.debug(f"Wrote to tracker{csv_path} message - {msg}")
