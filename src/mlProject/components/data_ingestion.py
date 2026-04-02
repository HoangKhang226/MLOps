import os
import urllib.request as request
import zipfile
from src.mlProject import logger
from src.mlProject.utils.common import get_size
from src.mlProject.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(
            self.config.local_data_file
        ):  # check if the file already exists
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,  # url of the file
                filename=self.config.local_data_file,  # local path to save the file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        # Check if the downloaded file is a zip file
        if self.config.local_data_file.endswith(".zip"):
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Unzipped {self.config.local_data_file} to {unzip_path}")
        elif self.config.local_data_file.endswith(".csv"):
            import shutil

            # If it's a CSV, we copy it to the expected unzipped location
            # Note: We name it the same as the expected file in the template
            dest_path = os.path.join(unzip_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
            shutil.copy(self.config.local_data_file, dest_path)
            logger.info(f"Copied CSV {self.config.local_data_file} to {dest_path}")
