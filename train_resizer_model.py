import torch
from torch.utils.data import DataLoader
from src.datasets.papdata import ICDARIliad, CustomImages, collate_fn

from src.models.frcnn_dr import FasterRCNN_DR
from src.models.frcnn_dr import Trainer as FRCNN_Trainer 
import _config as C

torch.manual_seed(C.SEED)

"""
Train the model for the resizer operation

This model is trained to detect letter bounding-boxes on images of full size.
The prediction is used to provide the avarage shape of detected bounding boxes.
"""

train_workers = 4
batch_size = 1
dropout = 0.5
lr = 4e-4
epochs = 150

checkpoint = None
checkpoint_out_dir = "checkpoints/resizer"

image_size = C.IMAGE_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## ICDAR23 Dataset
training = ICDARIliad(dataset_path=C.ICDAR_DATASET_PATH, is_training=True, val_size=C.VAL_SIZE,  image_size=image_size, transforms=None, seed=C.SEED)
validation = ICDARIliad(dataset_path=C.ICDAR_DATASET_PATH, is_training=False, val_size=C.VAL_SIZE, image_size=image_size, transforms=None, seed=C.SEED)

## To use with custom datasets
#training = CustomImages(dataset_path=C.DATASET_CUSTOM, is_training=True, val_size=C.VAL_SIZE,  image_size=image_size, transforms=None, seed=C.SEED)
#validation = CustomImages(dataset_path=C.DATASET_CUSTOM, is_training=False, val_size=C.VAL_SIZE, image_size=image_size, transforms=None, seed=C.SEED)


model_image_size = image_size

loader_train = DataLoader(training, shuffle=True, num_workers=train_workers, collate_fn=collate_fn, persistent_workers=True,  batch_size=batch_size, drop_last=True, pin_memory=True)
loader_val = DataLoader(validation, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=batch_size)

model = FasterRCNN_DR("resnet50", C.N_CLASSES, model_image_size, dropout=dropout)

optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

trainer = FRCNN_Trainer(model, device)

trainer.train(checkpoint_out_dir, optim, loader_train, loader_val, checkpoint=checkpoint, epochs=epochs, iters_verb=30)
