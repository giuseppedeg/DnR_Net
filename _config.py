import os
import json

SEED = 42

model_type = "FRCNN"  # "FRCNN"

if model_type == "YOLO":
    IMAGE_SIZE = 2560
    PATCH_SIZE = 640
    PADDING = 0
    REF_BOX_HEIGHT = 100 # 150 # 75 150 

elif model_type == "FRCNN":
    IMAGE_SIZE = 1600#1600 # 800 # 1600 
    PATCH_SIZE = 400#400 # 800 # 400
    PADDING = 15
    REF_BOX_HEIGHT = 75 # 96 # 75

FRCNN_mod = "resnet50"

from src.datasets.papdata import PADDING_BIG_IMAGE
#PADDING_BIG_IMAGE = 20
PADDING_VALUE = 1

CHECKPOINT_RESIZER = "checkpoints/resizer/resizer.pth"



DATASET_CUSTOM = "data/custom_dataset"
ICDAR_DATASET_PATH = "data/icdar"
ICDAR_TESTSET_PATH = "data/icdar_test"
PREPROCESSED_DATASET_PATH = "data/preprocessed"
PREPROCESSED_DATASET_PATH_TRAIN = os.path.join(PREPROCESSED_DATASET_PATH, "train")
PREPROCESSED_DATASET_PATH_VALID = os.path.join(PREPROCESSED_DATASET_PATH, "val")
VAL_SIZE = 0.2
PROB_INCLUDE_LABELS = 1 # Probability to include in patched dataset
PROB_INCLUDE_EMPTYLABELS = 0 # Probability to include in patched dataset images wit no labels

CATEGORIES_DS = ICDAR_DATASET_PATH
if os.path.exists(os.path.join(CATEGORIES_DS, "categories.json")):
    f_name = "categories.json"
else:
    f_name = "data.json"

with open(os.path.join(CATEGORIES_DS, f_name), encoding="utf-8") as f:
    CATEGORIES = json.load(f)["categories"]

ENCODING_CATEGORIES = {}
DECODING_CATEGORIES = {}
for i, cat in enumerate(CATEGORIES):
    ENCODING_CATEGORIES[cat['id']] = i + 1
    DECODING_CATEGORIES[i] = cat['id']

## DELETE!!!! Forse No
if model_type == "FRCNN":
    ENCODING_CATEGORIES = {}
    DECODING_CATEGORIES = {}
    for i, cat in enumerate(CATEGORIES):
        ENCODING_CATEGORIES[cat['id']] = i + 1
        DECODING_CATEGORIES[i+1] = cat['id']

N_CLASSES = len(CATEGORIES)

train_workers = 4
batch_size = 1
DROPOUT = 0.5


# FUSE ANNOTATION JSON
IOU_THR = 0.5
IOU_SAME_THR = 0.25
SCORE_THR = 0.05
MIN_VOTERS = 1

# VERBOSE
DISABLE_TQDM = False
YOLO_VERBOSE = True
