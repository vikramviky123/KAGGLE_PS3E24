import os
import sys
from smoker_status import logging, CustomException
from smoker_status.c_config.configuration import ConfigurationManager
from smoker_status.d_components.data_ingestion import DataIngestion


STAGE_NAME = "DATA -- INGESTION -- STAGE"


class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logging.info(
            f"\n\nx==========x\n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logging.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x\n\n")
    except Exception as e:
        logging.exception(CustomException(e, sys))
        raise CustomException(e, sys)
