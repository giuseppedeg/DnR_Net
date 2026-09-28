import os
import json
import torch
from PIL import Image
from torch.utils.data import DataLoader
from src.datasets.papdata import ImagesDataset, collate_fn
from src.operations.opt_resizing import Resizer
from src.operations.patchhfy import Patchify, Patchifier, depatchify, plot_imgannot, plot_patches
from src.models.processer import Preprocesser
from src.models.frcnn_dr import FasterRCNN_DR
from src.utils.target_utils import scale_targets, create_JSON, fuse_annotations, delete_overlapping_boxes, create_pap
from tqdm import tqdm
from datetime import datetime
import _config as C


torch.manual_seed(C.SEED)

"""
Train for my network for D&R
is the training of the network that can be used for the resizing and the D&R

"""


checkpoints_dir = "checkpoints/weights"
out_dir = "out/"
img_dirs = "data/images_to_test"
batch_test = 8


image_size = C.PATCH_SIZE + C.PADDING*2
padding = C.PADDING

add_custom_bt = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
assert os.path.exists(img_dirs), f"Image directory {img_dirs} does not exist."
dataset_test = ImagesDataset(dataset_path=img_dirs, image_size=C.IMAGE_SIZE, transforms=None) 
loader_test = DataLoader(dataset_test, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=collate_fn, batch_size=1)

categories = C.DECODING_CATEGORIES

## PREPROCESSOR
opts_chain = []

# Resizer
model = FasterRCNN_DR("resnet50", C.N_CLASSES, C.IMAGE_SIZE, dropout=C.DROPOUT)
model.eval()
resizer = Resizer(model, device, C.REF_BOX_HEIGHT)
resizer.load_model_state(C.CHECKPOINT_RESIZER)
opts_chain.append(resizer)

# Patchiging
patcher = Patchify(patch_size=C.PATCH_SIZE, padding=C.PADDING, padding_value=C.PADDING_VALUE)
patchifier = Patchifier(patcher, device)
opts_chain.append(patchifier)

preprocesser = Preprocesser(opts_chain)


## D&R Model
model = FasterRCNN_DR(C.FRCNN_mod, C.N_CLASSES, image_size, dropout=C.DROPOUT)


ids_image = {}
all_images = []
all_annotations = {}
stard_id_ann = 0

with torch.no_grad():
    for img_tensor, target, m_data in tqdm(loader_test, disable=C.DISABLE_TQDM):
        preprocessed_data = preprocesser((img_tensor, target))
        batch_patches = preprocessed_data['data']


        images, targets = batch_patches
        images = [x.to(device, non_blocking=True) for x in images]

        original_images_shapes = []
        for m in m_data:
            img_n = m['img_path']
            img = Image.open(img_n).convert("RGB")
            # original_images_shapes.append([3, img.size[1]+2*C.PADDING_BIG_IMAGE, img.size[0]+2*C.PADDING_BIG_IMAGE])
            original_images_shapes.append([3, img.size[1], img.size[0]])

        for model_weights in tqdm(os.listdir(checkpoints_dir), desc="Batch", leave=False, colour="GREEN", disable=C.DISABLE_TQDM):
            checkpoint = torch.load(os.path.join(checkpoints_dir, model_weights), weights_only=False)
            model.load_state_dict(checkpoint['model'])
            model.to(device).eval()

            batched_images = [images[i * batch_test:(i + 1) * batch_test] for i in range((len(images) + batch_test - 1) // batch_test )] 
            region_predictions = []
            for batch_i in tqdm(batched_images, desc="Run", leave=False, colour="RED", disable=C.DISABLE_TQDM):
                batch_predictions = model(batch_i)
                region_predictions.extend(batch_predictions)

            for t,p in zip(targets, region_predictions):
                t['boxes'] = p['boxes']
                t['labels'] = p['labels']
                t['scores'] = p['scores']

            # depatchify
            resized_images_shapes = preprocessed_data['scald_sizes']
            rec_img, rec_target = depatchify(batch_patches, resized_images_shapes, padding)
            
            # rescale to original image size
            scaled_target = scale_targets(rec_target, resized_images_shapes, original_images_shapes)

            # Remove overlapped boxes
            del_target = delete_overlapping_boxes(scaled_target, iou_threshold=0.7)
           
            ## SAVE OUTPUT
            for m, t, size in zip(m_data, scaled_target, original_images_shapes):
                img_p = m['img_path']
                img_name = os.path.splitext(os.path.basename(img_p))[0]
                current_out_path = os.path.join(out_dir, img_name)
                if img_name not in ids_image:
                    ids_image[img_name] = len(ids_image)
                id_image = ids_image[img_name]
                
                os.makedirs(current_out_path, exist_ok=True)
                os.makedirs(os.path.join(current_out_path, "annotations"), exist_ok=True)
                
                image = {
                    "id": id_image,
                    "file_name": os.path.basename(img_p),
                    "img_url": os.path.basename(img_p),
                    "height": size[1],
                    "width": size[2],
                    "license": None
                }
                all_images.append(image)
                all_annotations[id_image] = t

                curr_json = create_JSON(C.CATEGORIES, {id_image: image}, {id_image: t}, encoding_categories=categories, padding=0)

                ann_filename = str(len(os.listdir(os.path.join(current_out_path, "annotations")))).zfill(3)
                with open(os.path.join(current_out_path, "annotations", f"{ann_filename}.json"), 'w', encoding='utf-8') as f:
                    json.dump(curr_json, f, ensure_ascii=False, indent=4)

        all_ann_files = [os.path.join(current_out_path, "annotations",x) for x in os.listdir(os.path.join(current_out_path, "annotations"))]
        annotations = fuse_annotations(all_ann_files, iou_threshold=C.IOU_THR, iou_sameclass_threshold=C.IOU_SAME_THR, score_threshold=C.SCORE_THR, min_voters=C.MIN_VOTERS, stard_id=stard_id_ann)
        stard_id_ann = len(annotations['annotations'])
        with open(os.path.join(current_out_path, "annotations.json"), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4)

        create_pap(m_data[0]['img_path'], os.path.join(current_out_path, "annotations.json"), save_img=False)

    
## Creare il JSON grande con tutte le immagini
general_json = {
    "info": {
            "version": 0,
            "description": "Automatic result of Detection and Recognition",
            "date_created": datetime.today().strftime('%Y-%m-%d__%H:%M:%S')
        },
        "lincenses": {
            "id": 0,
            "name": "__LICENZENAME__",
            "url": "__LICENZEURL__"
        },
        "categories": C.CATEGORIES
    }

annotations = []
for img_name in ids_image:
    with open(os.path.join(out_dir, img_name, "annotations.json"), "r", encoding="utf-8") as f:
        predictions = json.load(f)
    for ann in predictions['annotations']:
        annotations.append(ann)

general_json['images'] = all_images
general_json['annotations'] = annotations

with open(os.path.join(out_dir, f"annotations_{datetime.today().strftime('%Y-%m-%d__%H-%M-%S')}.json"), 'w', encoding='utf-8') as f:
    json.dump(general_json, f, ensure_ascii=False, indent=4)


print("Done")
