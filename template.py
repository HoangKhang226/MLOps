import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "mlProject"

list_of_files = [
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "research/01_data_ingestion.ipynb",
    "research/02_eda_model.ipynb",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/logger.py",
    "tests/__init__.py",
    "artifacts/.gitkeep",
    "logs/.gitkeep",
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "app.py",
    "main.py",
    "requirements.txt",
]
for file_path in list_of_files:
    # Converting the file path string into a Path object for cross-platform compatibility
    file_path = Path(file_path)
    
    # Separating the directory and the filename from the path
    file_dir, file_name = os.path.split(file_path)

    # If a directory is specified, create it if it doesn't already exist
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file: {file_name}")

    # Create an empty file if it doesn't exist or if it is currently empty
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass # Creating an empty file
        logging.info(f"Creating empty file: {file_path}")


    else:
        # Log if the file already exists to avoid overwriting existing data
        logging.info(f"{file_name} already exists")
   