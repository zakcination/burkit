#yolo predict model='runs/detect/train34/weights/best.pt' source='datasets/RDD2022/test1_images' imgsz=320 device=0

python3 src/inference.py