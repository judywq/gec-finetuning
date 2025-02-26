import os
import datetime
from settings import DEFAULT_LOG_LEVEL

import logging
logger = logging.getLogger(__name__)


def get_date_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def setup_log(level=None, log_path='./log/txt', need_file=True):
    if not level:
        level = logging.getLevelName(DEFAULT_LOG_LEVEL)
    if not os.path.exists(log_path):
        os.makedirs(log_path)    
        
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s: %(message)s")
    
    handlers = []
    if need_file:
        filename = get_date_str()
        file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, filename))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level=level)
    handlers.append(console_handler)

    # https://stackoverflow.com/a/11111212
    logging.basicConfig(level=logging.DEBUG,
                        handlers=handlers)

def ask_if_delete_file(filename):
    """Ask if the user wants to delete the output file.
    Args:
        filename: str, the path to the file to be deleted
    Returns:
        str, 'y' if the file is deleted or not exists, 
            'a' if the user wants to abort, 
            'n' otherwise
    """
    if os.path.exists(filename):
        logger.info(f"Output file {filename} already exists. Delete it? (y)es/(n)o/(A)bort")
        answer = input().lower()
        if answer in ['y', 'yes']:
            os.remove(filename)
            return 'y'
        elif answer in ['a', 'abort', '']:
            return 'a'
        elif answer in ['n', 'no']:
            return 'n'
        else:
            raise ValueError(f"Invalid answer: {answer}")
    return 'y'
