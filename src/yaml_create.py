import os
import yaml

# Define the dataset configuration
dataset_config = {
    'path': 'datasets/RDD2022',
    'train': 'datasets/RDD_2022/train.txt',
    'val': 'datasets/RDD_2022/val.txt',
    'test': 'datasets/RDD_2022/test1_images',
    'nc': 4,
    'names': ['longitudinal cracks', 'transverse cracks', 'alligator cracks', 'potholes']
}

# Specify the path for the YAML file
yaml_path = 'datasets/RDD_2022/data.yaml'

# Create the directory if it doesn't exist
os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

# Write the dataset configuration to a YAML file
with open(yaml_path, 'w') as file:
    yaml.dump(dataset_config, file, sort_keys=False)