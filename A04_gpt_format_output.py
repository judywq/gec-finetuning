from lib.utils import setup_log
from lib.data_formatter import DataFormatter
import settings
import os


def main():
    
    formatter = DataFormatter(settings)
    formatter.run()


if __name__ == "__main__":
    setup_log()
    main()
