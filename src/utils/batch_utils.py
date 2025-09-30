import os
import json
from torchvision.utils import save_image
from PIL import ImageDraw
import torch
from torchvision.transforms import ToPILImage

OUT_IMG_FORMAT = ".png"

def get_imgannot(imgtensor, target):
    """
    retrn an image with the relative boundingboxes
    """
    if len(imgtensor.shape) > 3:
        imgtensor = imgtensor.squeeze()
    image = ToPILImage()(imgtensor)
    image = image.convert("RGB")

    if target is not None:

        img_dr = ImageDraw.Draw(image)

        for box in target['boxes']:

            img_dr.rectangle(box.tolist(), outline=(255,0,0), width=2) 

    return image


def save_batch_images(batch, out_path, onefolder=True):
    images, targets = batch
    if onefolder:
        os.makedirs(os.path.join(out_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_path, "jsons"), exist_ok=True)
    else:
        os.makedirs(out_path, exist_ok=True)

    for image, target in zip(images, targets):
        img_name = os.path.splitext(target["image_name"])[0]
        
        if onefolder:
           img_path = os.path.join(out_path,"images", f'{str(len(os.listdir(os.path.join(out_path, "images")))).zfill(4)}{OUT_IMG_FORMAT}')
           json_path = os.path.join(out_path,"jsons", f'{str(len(os.listdir(os.path.join(out_path, "jsons")))).zfill(4)}.json')
        else:
            folder_out = os.path.join(out_path, img_name)
            os.makedirs(os.path.join(folder_out, "images"), exist_ok=True)
            os.makedirs(os.path.join(folder_out, "jsons"), exist_ok=True)
            img_path = os.path.join(folder_out, "images" f'{str(len(os.listdir(folder_out))).zfill(4)}{OUT_IMG_FORMAT}')
            json_path = os.path.join(folder_out,"jsons", f'{str(len(os.listdir(folder_out))).zfill(4)}.json')

        save_image(image, img_path)
        with open(json_path, 'w') as f:
            if torch.is_tensor(target["image_part"]) :
                image_part = target["image_part"].tolist()
            else:
                image_part = None
            
            save_data = {
                "boxes": target["boxes"].tolist(),
                "labels": target["labels"].tolist(),
                "image_id": target["image_id"].tolist(),
                "image_name": target["image_name"],
                "area": target["area"].tolist(),
                "iscrowd": target["iscrowd"].tolist(),
                "image_part": image_part,
            }
            json.dump(save_data, f, indent=4)


def save_batch_labelledImages(batch, out_path, onefolder=True):
    images, targets = batch
    os.makedirs(out_path, exist_ok=True)

    for image, target in zip(images, targets):
        img_name = os.path.splitext(target["image_name"])[0]

        pil_labelledImage = get_imgannot(image, target)

        if onefolder:
           img_path = os.path.join(out_path, f'{str(len(os.listdir(out_path))).zfill(4)}{OUT_IMG_FORMAT}')
        else:
            folder_out = os.path.join(out_path, img_name)
            os.makedirs(folder_out, exist_ok=True)
            img_path = os.path.join(folder_out, f'{str(len(os.listdir(folder_out))).zfill(4)}{OUT_IMG_FORMAT}')

        pil_labelledImage.save(img_path)

