import torch
from torch.utils.data import DataLoader
from src.datasets.papdata import PreprocessedDataset, collate_fn

from src.models.frcnn_dr import FasterRCNN_DR
from src.models.frcnn_dr import Trainer as FRCNN_Trainer 
import _config as C

torch.manual_seed(C.SEED)

"""
Train for my network for D&R
is the training of the network that can be used for the resizing and the D&R

"""

train_workers = 1
batch_size = 8
dropout = 0.5
lr = 4e-3
epochs = 50

checkpoint = None #set this to continue a training

checkpoint_out_dir = "checkpoints/dnr"

image_size = C.PATCH_SIZE + C.PADDING*2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocessed images for training
training = PreprocessedDataset(dataset_path=C.PREPROCESSED_DATASET_PATH_TRAIN, image_size=image_size, is_training=True, transforms=None, seed=C.SEED)
validation = PreprocessedDataset(dataset_path=C.PREPROCESSED_DATASET_PATH_VALID, image_size=image_size, is_training=False, transforms=None, seed=C.SEED)

loader_train = DataLoader(training, shuffle=True, num_workers=train_workers, collate_fn=collate_fn, persistent_workers=True,  batch_size=batch_size, drop_last=True, pin_memory=True)
loader_val = DataLoader(validation, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=batch_size)


model = FasterRCNN_DR(C.FRCNN_mod, C.N_CLASSES, image_size, dropout=dropout)

optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
#optim = torch.optim.Adam(model.parameters(), lr=lr)

trainer = FRCNN_Trainer(model, device)

trainer.train(checkpoint_out_dir, optim, loader_train, loader_val, checkpoint=checkpoint, epochs=epochs, iters_verb=30)

print("Done")