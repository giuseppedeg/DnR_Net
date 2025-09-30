import torch
from datetime import datetime
import json
import os
import torchvision
from .fileformat_handler import FFHandler
from PIL import Image


def _add_items_to_group(items, groups):
    """
    Add list of items to groups,
    If there are no groups that match with the items, create a new group and put those item in this new group
    If there is only one matching group, add all these items to this group
    If there is more than one matching group, add all these items to the first group, then move items from
                other matching groups to this first group
    """
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))


def scale_targets(targets, original_size, goal_size):
    
    new_targets = []

    for t, orig_s, goal_s in zip(targets, original_size, goal_size):

        scale_factor = goal_s[1] / orig_s[1]
        
        boxes = torch.empty(t['boxes'].shape, dtype=torch.float32)
        area = []
        new_t = t.copy()

        for ind, box in enumerate(t['boxes']):
            new_box = box * scale_factor
            new_area = (new_box[2]-new_box[0]) * (new_box[3]-new_box[1])

            boxes[ind] = new_box
            area.append(new_area.item())

        new_t['boxes'] = boxes
        new_t['area'] = torch.tensor(area)

        new_targets.append(new_t)
    
    return new_targets


def create_JSON(categories, images, annotations, encoding_categories=None, padding=0):
    """
    creates a JSON for annotations in COCO format
    """
    json_data = {
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
        "categories": categories
    }

    json_images = []
    for id_img, img in images.items():
        json_images.append(img)

    json_annotations = []
    # CREATE ANNOTATIONS IN COCO FORMAT
    for id_img, ann in annotations.items():
        for b, l, s, a in zip(ann["boxes"], ann["labels"], ann["scores"], ann["area"]):
            box = b.tolist()
            if encoding_categories is None:
                categ =  l.item()
            else:
                categ = encoding_categories[l.item()]
            json_annotations.append({
                "image_id": id_img,
                "category_id": categ,
                # "bbox": [round(box[0]), round(box[1]), round(box[2]-box[0]), round(box[3]-box[1])],
                "bbox": [box[0]-padding, box[1]-padding, box[2]-box[0], box[3]-box[1]],
                "area": a.item(),
                "score": s.item()
            })

    json_data['images'] = json_images
    json_data['annotations'] = json_annotations


    return json_data


def fuse_annotations(prediction_files, iou_threshold=0.7, iou_sameclass_threshold=0.25, score_threshold=0.05, min_voters=1, stard_id=0):
    """
    Merges in a single JSON all annotations in all files in prediction_files list

    params:
      - prediction_files: list of all JSON path to marge
      - iou_threshold: tharshold to consider two annotations to be overlapped
      - score_threshold: only annotation with a scor hoigher will be saved in the final file
    """
    all_predictions = {}
    for idx, prediction_file in enumerate(prediction_files):
        with open(prediction_file, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        for annotation in predictions['annotations']:
            all_predictions.setdefault(annotation['image_id'], {}).setdefault(idx, []).append(annotation)
    
    out_json = predictions.copy()

    preds = {}
    for image in all_predictions:
        preds.setdefault(image, {})
        boxes, labels, scores = [], [], []
        for f_id, annotations in all_predictions[image].items():
            preds[image].setdefault(f_id, {})
            for annotation in annotations:
                if annotation['score'] < score_threshold:
                    continue
                box = annotation['bbox']
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                boxes.append(box)
                labels.append(annotation['category_id'])
                scores.append(annotation['score'])

        preds[image]['boxes'] = torch.tensor(boxes)
        preds[image]['labels'] = torch.tensor(labels)
        preds[image]['scores'] = torch.tensor(scores)

    output_annotations = []

    for image in preds:
        boxes_1 = preds[image]['boxes']
        boxes_2 = preds[image]['boxes']

        # if OOM:
        #     # save boxes
        #     torch.save(boxes_1, os.path.join("OOM","boxes.pt"))

        #     # load IOU
        #     iou = torch.load(os.path.join("OOM","iou_tensor.pt"), weights_only=True)
        #     os.remove(os.path.join("OOM","iou_tensor.pt"))
        # else:
        # compute the IoU between the two sets of bounding boxes
        iou = torchvision.ops.box_iou(boxes_1, boxes_2)

        # find the indices of the overlapping bounding boxes
        overlapping_indices = torch.where(iou > iou_threshold)

        groups = []
        for indicate_1, indicate_2 in zip(overlapping_indices[0], overlapping_indices[1]):
            _add_items_to_group([indicate_1.item(), indicate_2.item()], groups)
        
        group_filtered = [x for x in groups if len(x) >= min_voters]
        im_boxes, im_labels, im_scores = [], [], []
        for ids in group_filtered:
            sample_ids = torch.tensor(list(ids))
            boxes = preds[image]['boxes'][sample_ids]
            labels = preds[image]['labels'][sample_ids]
            scores = preds[image]['scores'][sample_ids]
            label_ids, label_counts = torch.unique(labels, return_counts=True)

            label = label_ids[torch.argmax(label_counts)]
            selected_ids = labels == label
            boxes = boxes[selected_ids]
            scores = scores[selected_ids]

            box = boxes.mean(dim=0)
            score = scores.mean()

            im_boxes.append(box)
            im_labels.append(label)
            im_scores.append(score)

        im_boxes = torch.stack(im_boxes)
        im_labels = torch.stack(im_labels)
        im_scores = torch.stack(im_scores)

        # Overlapping same class
        if len(im_boxes) > 2:
            boxes_1 = im_boxes
            boxes_2 = im_boxes
            iou = torchvision.ops.box_iou(boxes_1, boxes_2)
            overlapping_indices = torch.where(iou > iou_sameclass_threshold)
            groups = []
            for indicate_1, indicate_2 in zip(overlapping_indices[0], overlapping_indices[1]):
                _add_items_to_group([indicate_1.item(), indicate_2.item()], groups)
            group_filtered = [x for x in groups if len(x) >= min_voters]

            def_im_boxes, def_im_labels, def_im_scores = [], [], []
            for ids in group_filtered:
                sample_ids = torch.tensor(list(ids))
                boxes = im_boxes[sample_ids]
                labels = im_labels[sample_ids]
                scores = im_scores[sample_ids]
                label_ids, label_counts = torch.unique(labels, return_counts=True)

                label = label_ids[torch.argmax(label_counts)]
                selected_ids = labels == label
                selected_boxes = boxes[selected_ids]
                selected_scores = scores[selected_ids]

                box = selected_boxes.mean(dim=0)
                score = selected_scores.mean()

                def_im_boxes.append(box)
                def_im_labels.append(label)
                def_im_scores.append(score)

                selected_ids = torch.logical_not(selected_ids)

                selected_boxes = boxes[selected_ids]
                selected_labels = labels[selected_ids]
                selected_scores = scores[selected_ids]

                for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
                    def_im_boxes.append(box)
                    def_im_labels.append(label)
                    def_im_scores.append(score)
        else:
            def_im_boxes = im_boxes
            def_im_labels = im_labels
            def_im_scores = im_scores


        for box, label, score in zip(def_im_boxes, def_im_labels, def_im_scores):
            box_np = box.numpy().astype(float)
            annotation = {
                'id': stard_id,
                'image_id': image,
                'category_id': label.item(),
                'bbox': [round(box_np[0]), round(box_np[1]), round(box_np[2] - box_np[0]), round(box_np[3] - box_np[1])],
                'area':  round(box_np[2] - box_np[0]) * round(box_np[3] - box_np[1]),
                'score': float(score.item())
            }
            output_annotations.append(annotation)
            stard_id += 1
    
    out_json["annotations"] = output_annotations

    return out_json


def delete_overlapping_boxes(predictions, iou_threshold=0.3, additional_keys=()):
    out = []
    for pred in predictions:
        boxes_1 = pred['boxes']
        boxes_2 = pred['boxes']

        output = {}

        if len(boxes_1) != 0:
            scores_1 = pred['scores']
            scores_2 = pred['scores']

            # compute the IoU between the two sets of bounding boxes
            iou = torchvision.ops.box_iou(boxes_1, boxes_2)

            # find the indices of the overlapping bounding boxes
            overlapping_indices = torch.where(iou > iou_threshold)

            b1_scores = scores_1[overlapping_indices[0]]
            b2_scores = scores_2[overlapping_indices[1]]
            b1_lt_b2 = torch.less_equal(b1_scores, b2_scores)

            b1_remove = overlapping_indices[0][b1_lt_b2]
            b2_remove = overlapping_indices[1][torch.logical_not(b1_lt_b2)]
        else:
            b2_remove = []
            
        k_to_reduce = ["boxes", "scores", "labels", "area"] 
        for key_ in k_to_reduce:
            el = pred[key_]
            el_red = delete_in_tensor(el, b2_remove)
            output[key_] = el_red

        for key_ in pred:
            if key_ not in k_to_reduce:
                output[key_] = pred[key_]
        
        # Problem 'iscrowd' - dimension dose not match  
        iscrowd = torch.zeros((output["boxes"].shape[0]), dtype=torch.int64)
        output["iscrowd"] = iscrowd

        out.append(output)

    return out


def delete_in_tensor(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def create_pap(or_image_path, prediction_path, save_img=False, image_kb_max=1300000):
    file_h = FFHandler()

    out_fld = os.path.dirname(prediction_path)
    img_name = os.path.basename(or_image_path)

    image = Image.open(or_image_path)
    image = image.convert("RGB") 

    image_path = os.path.join(out_fld, img_name)

    quality = 100
    image.save(image_path,quality=quality,optimize=True)

    file_size = os.path.getsize(image_path)

    while (file_size > image_kb_max and quality > 1):
        quality -= 5
        image.save(image_path, quality=quality, optimize=True)
        file_size = os.path.getsize(image_path)
        #print(name, file_size)
   
    file_h.save_formattedfile(image_path, prediction_path, out_fld)

    if not save_img:
        os.remove(image_path)
    
