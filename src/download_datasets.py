import os
from tqdm import tqdm
import requests
import zipfile
from utils import logger
import shutil
import gdown


# Create necessary directories
base_dir = 'orddc2022/'
datasets_dir = os.path.join(base_dir, 'datasets/')
rdd_dir = os.path.join(datasets_dir, 'RDD2022/')

os.makedirs(base_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(rdd_dir, exist_ok=True)

# Change to the datasets directory
os.chdir(rdd_dir)

# List of dataset URLs and corresponding folder names
datasets = [
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip", "Norway"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip", "Japan"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip", "India"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip", "Czech"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip", "United_States"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip", "China_MotorBike"),
    ("https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_Drone.zip", "China_Drone"),
]

# Function to download and unzip a dataset
def download_and_unzip(url, folder_name):
    local_filename = url.split('/')[-1]
    
    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=f"Downloading {local_filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    
    # Unzip with progress bar
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(
            desc=f"Unzipping {local_filename}",
            total=total_files,
            unit='file',
        ) as bar:
            for file_info in zip_ref.infolist():
                zip_ref.extract(file_info, folder_name)
                bar.update(1)
    
    # Remove the zip file
    # os.remove(local_filename)

# Download and unzip each dataset
# for url, folder in datasets:
#     folder_name = os.path.join(rdd_dir, folder)
#     os.makedirs(folder_name, exist_ok=True)
#     logger.info(f" downloading_datasets | download_and_unzip | folder : {folder}  - folder_name: {folder_name}")
#     download_and_unzip(url, folder_name)

logger.info("All datasets downloaded and unzipped successfully!")



# URLs or file IDs of the files to be downloaded
file_ids = [
    '1oFS2hjdY_tMS9hkPoB3KtQ9SWJvnCn-h',
    '12FIs5j_tRoGS_jAtZW1XTR_IrvWnotQu',
    '1MnWlibYWBo6aPcg-53OOPl9fCsOPWgcp',
    '1ukHpidxCEIF20SnMDFu1ddEO3QYbqudl'
]

# Base URL for gdown
source_dir = 'ordcc2022'
base_url = 'https://drive.google.com/uc?id='

# Download each file
for file_id in file_ids:
    url = base_url + file_id
    logger.info(f' downloading_datasets | download_files | url: {url}')
    gdown.download(url, quiet=False)


# List of files to be moved
files_to_move = [
    'damage_classes.txt',
    'gene_train_val.py',
    'gene_file_list.py',
    'xml2yolo.py'
]

# Move each file from the source to the destination
for file_name in files_to_move:
    source_file = os.path.join(source_dir, file_name)
    dest_file = os.path.join(rdd_dir, file_name)
    
    # Check if the file exists in the source directory
    if os.path.exists(source_file):
        shutil.move(source_file, dest_file)
        logger.info(f" downloading_datasets | move_files | Moved {file_name} to {rdd_dir}")
    else:
        print(f"File not found: {source_file}")

