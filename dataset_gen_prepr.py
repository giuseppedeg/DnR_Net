import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.papdata import ICDARIliad, CustomImages, collate_fn
from src.operations.opt_resizing import Resizer
from src.operations.patchhfy import Patchify, Patchifier
from src.models.processer import Preprocesser
from src.models.frcnn_dr import FasterRCNN_DR
from src.utils.batch_utils import save_batch_images, save_batch_labelledImages
from src.utils.transforms import Compose, PaddingImage, FixedImageResize, ToTensor
import os
import _config as C
import random
random.seed(C.SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out_dataset_path = C.PREPROCESSED_DATASET_PATH
batch_size = 1
dropout = 0.5
seed = 1
padding = C.PADDING


transform =  Compose([
                #LongRectangleCrop(),
                FixedImageResize(C.IMAGE_SIZE),
                PaddingImage(padding_size=C.PADDING_BIG_IMAGE),
                ToTensor()
            ])

val_size = 0

## ICDAR23 Dataset
training = ICDARIliad(dataset_path=C.ICDAR_DATASET_PATH, val_size=val_size, is_training=True, image_size=C.IMAGE_SIZE, transforms=transform, seed=seed)
validation = ICDARIliad(dataset_path=C.ICDAR_DATASET_PATH, val_size=val_size, is_training=False, image_size=C.IMAGE_SIZE, transforms=transform, seed=seed)

## To use with custom datasets
# training = CustomImages(dataset_path=C.DATASET_CUSTOM, is_training=True, val_size=val_size,  image_size=C.IMAGE_SIZE, transforms=transform, seed=seed)
# validation = CustomImages(dataset_path=C.DATASET_CUSTOM, is_training=False, val_size=val_size, image_size=C.IMAGE_SIZE, transforms=transform, seed=seed)


training_dl = DataLoader(training, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=batch_size)
validation_dl = DataLoader(validation, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=batch_size)

opts_chain = []

# resizing
model = FasterRCNN_DR("resnet50", C.N_CLASSES, C.IMAGE_SIZE, dropout=dropout)
model.eval()
resizer = Resizer(model, device, C.REF_BOX_HEIGHT)
resizer.load_model_state(C.CHECKPOINT_RESIZER)
opts_chain.append(resizer)


# Patchiging
patcher = Patchify(patch_size=C.PATCH_SIZE, padding=padding, padding_value=C.PADDING_VALUE)
patchifier = Patchifier(patcher, device)
opts_chain.append(patchifier)


preprocesser = Preprocesser(opts_chain)
#preprocesser.to(device)


out_dataset = os.path.join(out_dataset_path, "train") 
for id, batch in enumerate(tqdm(training_dl, desc=f"training", leave=False)):
    preproc_data = preprocesser(batch)

    save_batch = ([], [])
    for img_tensor, label in zip(preproc_data['data'][0], preproc_data['data'][1]):
        if len(label['boxes']) == 0:
           if random.random() <= C.PROB_INCLUDE_EMPTYLABELS:
                save_batch[0].append(img_tensor)
                save_batch[1].append(label)
        else:
            if random.random() <= C.PROB_INCLUDE_LABELS:
                save_batch[0].append(img_tensor)
                save_batch[1].append(label)

    # save image
    save_batch_images(save_batch, out_dataset)
    # save_batch_labelledImages(save_batch, out_dataset)


out_dataset = os.path.join(out_dataset_path, "val") 
for id, batch in enumerate(tqdm(validation_dl, desc=f"validation", leave=False)):
    preproc_data = preprocesser(batch)

    save_batch = ([], [])
    for img_tensor, label in zip(preproc_data['data'][0], preproc_data['data'][1]):
        if len(label['boxes']) == 0:
           if random.random() <= C.PROB_INCLUDE_EMPTYLABELS:
                save_batch[0].append(img_tensor)
                save_batch[1].append(label)
        else:
            if random.random() <= C.PROB_INCLUDE_LABELS:
                save_batch[0].append(img_tensor)
                save_batch[1].append(label)

    # save image
    save_batch_images(save_batch, out_dataset)
    #save_batch_labelledImages(save_batch, out_dataset)
