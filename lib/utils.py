import os
import datetime

import logging
logger = logging.getLogger(__name__)

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")

def setup_log(level=None, log_path='./log/txt', need_file=True):
    from settings import DEFAULT_LOG_LEVEL
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

def backup_output_file(filename, padding=3):
    """Backup the output file.
    Args:
        filename: str, the path to the file to be backed up
    Returns: str, the path to the backup file
    """
    def find_next_backup_filename(filename):
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        count = 1
        while True:
            new_filename = f"{base_name}.bk{str(count).zfill(padding)}{ext}"
            if not os.path.exists(new_filename):
                break
            count += 1
        return new_filename

    if os.path.exists(filename):
        new_filename = find_next_backup_filename(filename)
        # logger.info(f"Backup {filename} to {new_filename}")
        os.rename(filename, new_filename)
        return new_filename
    return None


def test_backup_output_file():
    filename = "data/test.txt"
    print(backup_output_file(filename))


if __name__ == "__main__":
    test_backup_output_file()
