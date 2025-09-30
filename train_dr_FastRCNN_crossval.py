import torch
import os
import shutil
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from src.datasets.papdata import PreprocessedDataset, collate_fn

from src.models.frcnn_dr import FasterRCNN_DR
from src.models.frcnn_dr import Trainer as FRCNN_Trainer
from sklearn.model_selection import KFold 
import _config as C

torch.manual_seed(C.SEED)

"""
Train for my network for D&R
is the training of the network that can be used for the resizing and the D&R

"""

N_fold = 3
train_workers = 1
batch_size = 4
dropout = 0.5
lr = 4e-3
epochs = 2

checkpoint = None # set this to continue a training

checkpoint_out_dir = "checkpoints/dnr_crossval"
checkpoint_out_dir = os.path.join(checkpoint_out_dir, datetime.today().strftime('%Y-%m-%d_%H-%M'))

image_size = C.PATCH_SIZE + C.PADDING*2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# preprocessed images for training
training = PreprocessedDataset(dataset_path=C.PREPROCESSED_DATASET_PATH_TRAIN, image_size=image_size, is_training=True, transforms=None, seed=C.SEED)

loader_train = DataLoader(training, shuffle=True, num_workers=train_workers, collate_fn=collate_fn, persistent_workers=True,  batch_size=batch_size, drop_last=True, pin_memory=True)


kf = KFold(n_splits=N_fold, shuffle=True, random_state=C.SEED)

# Store results for each fold
fold_epoch = []
fold_losses = []

# --- Loop Through Folds ---
for fold, (train_index, val_index) in enumerate(kf.split(training)):

    train_subset = Subset(training, train_index)
    val_subset = Subset(training, val_index)

    loader_train = DataLoader(train_subset, shuffle=True, num_workers=train_workers, collate_fn=collate_fn, persistent_workers=True,  batch_size=batch_size, drop_last=True, pin_memory=True)
    loader_val = DataLoader(val_subset, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=batch_size)

    model = FasterRCNN_DR(C.FRCNN_mod, C.N_CLASSES, image_size, dropout=dropout)

    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    #optim = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = FRCNN_Trainer(model, device)

    if checkpoint:
        ch = os.path.join(checkpoint, f"fold_{fold+1}", "models", "last.pth")
        dst = dst = os.path.join(checkpoint, f"fold_{fold+1}", "models", f"last_start_0.pth") 
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(checkpoint, f"fold_{fold+1}", "models", f"last_start_{i}.pth") 
            i += 1 

        shutil.copyfile(os.path.join(checkpoint, f"fold_{fold+1}", "models", "last.pth"), dst)
    else:
        ch = None

    best_model_info = trainer.train(checkpoint_out_dir, optim, loader_train, loader_val, checkpoint=ch, epochs=epochs, iters_verb=30)
    
    fold_epoch.append(best_model_info['best_val_epoch'])
    fold_losses.append(best_model_info['best_val_loss'])

    if not os.path.exists(os.path.join(checkpoint_out_dir, f"fold_{fold+1}")):
        os.rename(os.path.join(checkpoint_out_dir, best_model_info['name']),
                  os.path.join(checkpoint_out_dir, f"fold_{fold+1}"))

# --- Aggregate and Report Results ---
with open(os.path.join(checkpoint_out_dir, "cross_result.txt"), "w") as f:
    f.write("--- Cross-Validation Results ---\n\n")
    for i, loss in enumerate(fold_losses):
        f.write(f"Fold {i+1}  \t-  Best Epoch: {fold_epoch[i]} \tValidation Loss: {fold_losses[i]:.4f}\n")

    
    best_loss = min(fold_losses)
    best_fold = np.argmin(fold_losses)
    f.write(f"\n\nAverage Validation Loss: {np.mean(fold_losses):.4f} +/- {np.std(fold_losses):.4f}\n")
    f.write(f"Best Fold: {best_fold}  -  Validation Loss: {best_loss:.4f} Epoch {fold_epoch[best_fold]}\n")


        
print("Done")