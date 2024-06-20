#!/bin/bash

# Install necessary Python packages
pip install Pillow
pip install tqdm

# Change directory to datasets/RDD2022
cd datasets/RDD2022

# Run the gene_train_val.py script
python gene_train_val.py

# Change back to the previous directory
cd ../..

# Run the xml2yolo.py script with specified arguments
python datasets/RDD2022/xml2yolo.py --class_file datasets/RDD2022/damage_classes.txt --input_file datasets/RDD2022/train.txt

python datasets/RDD2022/gene_file_list.py

#!/bin/bash

# Create the target directory
mkdir -p datasets/RDD2022/test1_images

# Copy images from various directories to the target directory
cp datasets/RDD2022/China_MotorBike/test/images/* datasets/RDD2022/test1_images/
cp datasets/RDD2022/Czech/test/images/* datasets/RDD2022/test1_images/
cp datasets/RDD2022/India/test/images/* datasets/RDD2022/test1_images/
cp datasets/RDD2022/Norway/test/images/* datasets/RDD2022/test1_images/
cp datasets/RDD2022/Japan/test/images/* datasets/RDD2022/test1_images/
cp datasets/RDD2022/United_States/test/images/* datasets/RDD2022/test1_images/
