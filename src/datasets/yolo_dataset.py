import os
import json, shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict

def make_label_map(all_labels):
    """Create stable int ids for each unique character label."""
    return {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}


def convert_bbox_to_yolo(box, img_w, img_h):
    """(x_min,y_min,x_max,y_max)->(x_c,y_c,w,h) normalized [0,1]."""
    x_min, y_min, x_max, y_max = box
    w = x_max-x_min
    h = y_max-y_min
    x_c = x_min + w / 2
    y_c = y_min + h / 2
    return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]


def create_YOLO_trainset(dataset_dir, encoding_categories):
    dataset_dir = Path(dataset_dir)

    for data_set_ty in ["train", "val"]:
        image_dir = dataset_dir / data_set_ty / "images" 
        ann_dir   = dataset_dir / data_set_ty / "jsons" 
        
        # Gather all annotations & labels
        label_set = set()
        items = []
        for ann_path in ann_dir.glob("*.json"):
            with open(ann_path) as f:
                ann = json.load(f)
            img_path = image_dir / (ann_path.stem + ann_path.suffix.replace("json", "jpg"))
            if not img_path.exists():
                img_path = image_dir / (ann_path.stem + ".png")
            if not img_path.exists():
                print(f"⚠️ Image for {ann_path.name} not found, skipping.")
                continue
            label_set.update(ann["labels"])
            items.append((img_path, ann))

        # Build label → id map
        # lbl2id = make_label_map(label_set)
        # print("Detected labels:", lbl2id)

        # Prepare YOLO folders
        yolo_labels_dir = dataset_dir / data_set_ty / "labels" 
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)

        for i, (img_path, ann) in enumerate(sorted(items)):
            with Image.open(img_path) as im:
                w, h = im.size # PATCH SIZE
            out_lines = []
            for bbox, lbl in zip(ann["boxes"], ann["labels"]):
                xc, yc, bw, bh = convert_bbox_to_yolo(bbox, w, h)
                out_lines.append(f"{lbl} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            
            # write txt
            dest_lbl = (yolo_labels_dir / (img_path.stem + ".txt"))
            dest_lbl.write_text("\n".join(out_lines))

    # Write dataset.yaml
    names_list = [encoding_categories[lbl] for lbl in sorted(label_set)]
    (dataset_dir / "dataset.yaml").write_text(
        f"path: {dataset_dir}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        "names:\n" + "\n".join([f"  {i}: {n}" for i, n in enumerate(names_list)])
    )


