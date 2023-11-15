import logging
import os
import sys
from datetime import datetime

# ------------------------LOGGING-------------------------------------
# logs folder name
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

# store logs in folder path
logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Creating log folder path
os.makedirs(logs_path, exist_ok=True)

# log filepath
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Generate logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# ------------------------CustomException-------------------------------------


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
