import os
from PIL import Image
import ultralytics
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from utils import logger
import time
from pathlib import Path
import torch

def timeit(method):
    def timed(*args, **kw):
        logger.info(f'{method.__name__} started ...')
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f'{method.__name__} finished in {te-ts:.2f} seconds')
        return result
    return timed

damage_classes = [
    'D00',
    'D10',
    'D20',
    'D40'
]

model_name = 'yolov8m_data_aug'


def clean_gpu_util():
    # this function is used to clean the gpu utilisation after each prediction
    # this is to avoid the gpu memory from being filled up
    torch.cuda.empty_cache()



@timeit
def predict():
    # Load SETTINGS.json
    settings_path = 'SETTINGS.json'
    if not os.path.exists(settings_path):
        logger.error(f'{settings_path} does not exist.')
        return
    with open(settings_path) as f:
        settings = json.load(f)
    logger.info('Loaded settings from SETTINGS.json')

    # Load model from MODEL_DIR
    model_dir = os.path.join(os.getcwd(), settings['MODEL_DIR'])
    if not os.path.exists(model_dir):
        logger.error(f'Model directory {model_dir} does not exist.')
        return
    model = ultralytics.YOLO(model_dir)

    # Distribute model across all available CUDA devices
    if torch.cuda.is_available():
        device_ids = list(range(torch.cuda.device_count()))
        model.to(f'{device_ids[0]}')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info(f'Using CUDA devices: {device_ids}')
    else:
        logger.warning('CUDA is not available. Running on CPU.')

    logger.info(f'Loaded model from {settings["MODEL_DIR"].split("orddc")[-1]}')

    # Get the test samples directory
    test_dir = os.path.join(os.getcwd(), settings['TEST_DATA_CLEAN_DIR'])
    if not os.path.exists(test_dir):
        logger.error(f'Test samples directory {test_dir} does not exist.')
        return
    logger.info(f'Loading test samples from {test_dir.split("orddc")[-1]}')

    # Prepare submission directory
    submission_path = settings['SUBMISSION_DIR']
    if not os.path.exists(submission_path):
        os.makedirs(submission_path)
        logger.info(f'Created submission directory {submission_path}')

    # submission file name = predictions_{model backbone}.csv

    submission_file = os.path.join(submission_path, f'predictions_{model_name}.csv')
    if not os.path.exists(submission_file):
        #create a file
        os.mknod(submission_file) # create a new file
        logger.info(f'Created new submission file {submission_file}')

    # Initialize empty predictions list
    predictions = []

    # Iterate over all image files in the testidation directory with progress bar
    for root, dirs, files in os.walk(test_dir):
        for file in tqdm(files, desc="Processing images", unit="image", dynamic_ncols=True):
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                if not os.path.exists(image_path):
                    tqdm.write(f'Error: Image file {image_path} does not exist.')
                    continue

                try:
                    image = Image.open(image_path)
                except Exception as e:
                    tqdm.write(f'Error opening image {image_path}: {e}')
                    continue
                
                results = model(image, save=False, verbose=False)

                if len(results[0].boxes) == 0:
                    predictions.append([Path(image_path).name, ''])
                    continue

                boxes = results[0].boxes.xyxy.tolist()
                classes = results[0].boxes.cls.tolist()
                confidences = results[0].boxes.conf.tolist()

                prediction_string = []
                for box, cls, conf in zip(boxes, classes, confidences):
                    if conf > 0.3:
                        x1, y1, x2, y2 = box
                        detected_class = int(cls) + 1
                        prediction_string.append(f'{detected_class} {int(x1)} {int(y1)} {int(x2)} {int(y2)}')
                    else:
                        prediction_string.append('')

                image_id = Path(image_path).name
                if prediction_string:
                    predictions.append([image_id, ' '.join(prediction_string)])
                # Free up GPU memory
                torch.cuda.empty_cache()

    # Convert predictions to DataFrame and save as CSV without headers
    df = pd.DataFrame(predictions, columns=['ImageID', 'PredictionString'])
    #remove headers
    df.to_csv(submission_file, index=False, header=False)
    logger.info(f'Predictions saved to {submission_file}')
    logger.info('Predict finished ...')

    # Free up GPU memory after completing predictions
    torch.cuda.empty_cache()

@timeit
def sort_csv(path):
    # sort csv by column 0 file at path
    df = pd.read_csv(path)
    # Sort the DataFrame by column 0
    df = df.sort_values(df.columns[0])
    df.to_csv(path, index=False)
    logger.info(f'Sorted CSV file at {path}')


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU.")
    return device



if __name__ == '__main__':
    clean_gpu_util()
    get_device()
    predict()
    #add_empty_images('submissions/predictions.csv')
    sort_csv('submissions/predictions.csv')