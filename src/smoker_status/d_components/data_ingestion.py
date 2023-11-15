import os
import sys
from pathlib import Path
import urllib.request as request
import zipfile

from smoker_status import logging, CustomException

from smoker_status.c_config.configuration import DataIngestionConfig
from smoker_status.f_utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.syn_URL):

            filename, headers = request.urlretrieve(
                url=self.config.syn_URL,
                filename=self.config.file_path_syn
            )
            logging.info(
                f"{filename} download! with following info: \n{headers}")

            filename, headers = request.urlretrieve(
                url=self.config.act_URL,
                filename=self.config.file_path_act
            )

            logging.info(
                f"{filename} download! with following info: \n{headers} into downloaded and extracted folder")
        else:
            logging.info(
                f"File already exists of size: {get_size(Path(self.config.file_path))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.extracted_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.file_path_syn, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        with zipfile.ZipFile(self.config.file_path_act, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
