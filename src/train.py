from ultralytics import YOLO
import torch
from utils import logger
import wandb

# Load YOLOv10n model from scratch
model = YOLO("workspace/yolov8m.pt")
logger.info(f' train.py | Installed YOLOv8 weights')


def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU.")
    return device

device = get_device()
# Define the dry run mode
WANDB_MODE='disabled'

# Initialize WandB in dry run mode
wandb.init(mode=WANDB_MODE)


# Set the device, ensure it's valid
available_devices = list(range(torch.cuda.device_count()))
device_str = ','.join(map(str, available_devices))
print(f"Using devices: {device_str}")

logger.info(f' train.py  | training init | Training is ready to go!')
# Train the model
model.train(
    data='src/data.yaml',
    task='detect',
    epochs=300,
    verbose=True,
    batch=128,
    imgsz=640,
    patience=10,
    save=True,
    workers=8,
    cos_lr=True,
    lr0=0.0001,
    lrf=0.00001,
    warmup_epochs=3,
    warmup_bias_lr=0.000001,
    optimizer='Adam',
    seed=42,
    device=[0,1,2,3],  # device should be a string
    
    # Data Augmentation
    degrees=45,
    translate=0.5,
    shear=180,
    fliplr=0.5,
    mixup=0.0,
    copy_paste=0.2,
    hsv_h=0.05,
    hsv_s=0.05,
    hsv_v=0.5,
)

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
