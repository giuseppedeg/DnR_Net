import os
from ultralytics import YOLO
import _config as C
from src.datasets.yolo_dataset import create_YOLO_trainset


dataset_path = "data/preprocessed_YOLO_75_1"
epoch = 1
img_size = 640
batch = 12
lr=1e-3
workers=4
project_path="checkpoints/YOLO/test"
name = "YOLO_test"

checkpoint = None
#checkpoint = "checkpoints/YOLO/test/YOLO_test2/weights/last.pt"

if not os.path.exists(os.path.join(dataset_path, "dataset.yaml")):
    categories_dict = {}
    for cat in C.CATEGORIES:
        categories_dict[C.ENCODING_CATEGORIES[cat['id']]] = cat['name']

    create_YOLO_trainset(dataset_path, categories_dict)

if checkpoint:
    model = YOLO(checkpoint, verbose=C.YOLO_VERBOSE)
    resume = True
else:
    model = YOLO("yolov8l.yaml", verbose=C.YOLO_VERBOSE)          # yolov8n.pt | yolov8s.pt | yolov8m.pt | yolov8l.pt | yolov8x.pt
    resume = False

model.train(data=os.path.join(dataset_path, "dataset.yaml"), 
            epochs=epoch, 
            patience=20,
            iou = 0.7,
            imgsz=img_size, 
            batch=batch, 
            lr0=lr,
            workers=workers,
            project=project_path,
            name=name,
            resume=resume,
            pretrained=True)

model.save(os.path.join("checkpoints", "YOLO", "yolov8m_01.pt"))