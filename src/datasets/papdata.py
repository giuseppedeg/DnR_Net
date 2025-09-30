import glob
import json
import os
import random
import torch
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from src.utils.transforms import Compose, PaddingImage, FixedImageResize, ImageTransformCompose, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

PADDING_BIG_IMAGE = 20


def collate_fn(batch):
    return tuple(zip(*batch))


class ICDARIliad(Dataset):

    def __init__(self, dataset_path: str, is_training, test=False, image_size=1500, val_size=0.2, transforms=None, seed=42):
        self.seed = seed
        random.seed(seed)

        self.image_size = image_size
        
        self.dataset_path = dataset_path
        self.is_training = is_training
        self.test = test
        self.val_size = val_size
        self.transforms = transforms if transforms is not None else self.get_transforms(is_training)

        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))

        #self.images = sorted([os.path.basename(x) for x in images])

        random.shuffle(images)
        ind_split = len(images) - int(len(images)*self.val_size)
        if is_training:
            images = images[:ind_split]
        else:
            images = images[ind_split:]

        self.images = ["/".join(x.split(os.path.sep)[-2:]) for x in images]

        with open(os.path.join(dataset_path, "data.json"), encoding="utf-8") as f:
            self.data = json.load(f)

        if os.path.exists(os.path.join(dataset_path, "categories.json")):
            with open(os.path.join(dataset_path, "categories.json"), encoding="utf-8") as f:
                self.categories = json.load(f)["categories"]
        else:
            self.categories = self.data["categories"]

        self.letter_mapping = {}
        for i, cat in enumerate(self.categories):
            self.letter_mapping[cat['id']] = i + 1

        boxes = {}
        labels = {}
        imgs = set()
        for annotation in self.data['annotations']:
            try:
                labels.setdefault(annotation['image_id'], []).append(self.letter_mapping[int(annotation['category_id'])])
                imgs.add(annotation['image_id'])
            except:
                continue
            x, y, w, h = annotation['bbox']
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            boxes.setdefault(annotation['image_id'], []).append([xmin, ymin, xmax, ymax])

        self.boxes, self.labels = {}, {}
        for key in boxes:
            self.boxes[key] = torch.as_tensor(boxes[key], dtype=torch.float32)
            self.labels[key] = torch.as_tensor(labels[key], dtype=torch.int64)

        self.imgs = [] #= self.data['images']

        for img in self.data['images']:
            url_data = img["img_url"].split('/')
            img_path = "/".join(url_data[-2:])

            if img_path in self.images:
                self.imgs.append(img)


    def __len__(self):
        return len(self.imgs)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                #LongRectangleCrop(),
                # PaddingImage(padding_size=PADDING_BIG_IMAGE),
                FixedImageResize(self.image_size),
                ImageTransformCompose([
                    torchvision.transforms.RandomGrayscale(p=0.3),
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ], p=0.5)
                ]),
                ToTensor()
            ])
        else:
            return Compose([
                #LongRectangleCrop(),
                PaddingImage(padding_size=PADDING_BIG_IMAGE),
                FixedImageResize(self.image_size),
                ToTensor()
            ])


    def __getitem__(self, idx):
        return self.__get_item_by_idx(idx)

    def __get_item_by_idx(self, idx):
        image = self.imgs[idx]       

        #image = self.data['images'][image_idx]
        img_url = image['img_url'].split('/')
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image['id']

        boxes = self.boxes[image_id].clone()
        labels = self.labels[image_id].clone()

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        src_folder = os.path.join(self.dataset_path, "images", "homer2")
        fname = os.path.join(src_folder, image_folder, image_file)
        
        with Image.open(fname) as f:
            img = f.convert('RGB')

        target = {
            "boxes": boxes,
            "original_boxes": boxes.clone(),
            "labels": labels,
            "image_id": image_id,
            "image_name":image_file,
            "image_path": fname,
            "area": area,
            "iscrowd": iscrowd,
            "image_part": "None",
            "width": img.size[0],
            "height": img.size[1]
        }

        # img = self.transforms(img)
        img,target = self.transforms(img, target)
        
        if self.test:
            return img, target, {"img_path":fname, "img_id":image['id']}
        else:
            return img, target
        
    
class CustomImages(Dataset):

    def __init__(self, dataset_path: str, is_training, test=False, image_size=1500, val_size=0.2, transforms=None, seed=42):
        self.seed = seed
        random.seed(seed)

        self.image_size = image_size
        
        self.dataset_path = dataset_path
        self.is_training = is_training
        self.test = test
        self.val_size = val_size
        self.transforms = transforms if transforms is not None else self.get_transforms(is_training)

        images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))

        #self.images = sorted([os.path.basename(x) for x in images])

        random.shuffle(images)
        ind_split = len(images) - int(len(images)*self.val_size)
        if is_training:
            images = images[:ind_split]
        else:
            images = images[ind_split:]

        self.images = ["/".join(x.split(os.path.sep)[-2:]) for x in images]

        with open(os.path.join(dataset_path, "data.json"), encoding="utf-8") as f:
            self.data = json.load(f)

        if os.path.exists(os.path.join(dataset_path, "categories.json")):
            with open(os.path.join(dataset_path, "categories.json"), encoding="utf-8") as f:
                self.categories = json.load(f)["categories"]
        else:
            self.categories = self.data["categories"]

        self.letter_mapping = {}
        for i, cat in enumerate(self.categories):
            self.letter_mapping[cat['id']] = i + 1

        boxes = {}
        labels = {}
        imgs = set()
        for annotation in self.data['annotations']:
            try:
                labels.setdefault(annotation['image_id'], []).append(self.letter_mapping[int(annotation['category_id'])])
                imgs.add(annotation['image_id'])
            except:
                continue
            x, y, w, h = annotation['bbox']
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            boxes.setdefault(annotation['image_id'], []).append([xmin, ymin, xmax, ymax])

        self.boxes, self.labels = {}, {}
        for key in boxes:
            self.boxes[key] = torch.as_tensor(boxes[key], dtype=torch.float32)
            self.labels[key] = torch.as_tensor(labels[key], dtype=torch.int64)

        self.imgs = [] #= self.data['images']

        for img in self.data['images']:
            url_data = img["img_url"].split('/')
            img_path = "/".join(url_data[-2:])

            if img_path in self.images:
                self.imgs.append(img)


    def __len__(self):
        return len(self.imgs)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                #LongRectangleCrop(),
                # PaddingImage(padding_size=PADDING_BIG_IMAGE),
                FixedImageResize(self.image_size),
                ImageTransformCompose([
                    torchvision.transforms.RandomGrayscale(p=0.3),
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ], p=0.5)
                ]),
                ToTensor()
            ])
        else:
            return Compose([
                #LongRectangleCrop(),
                PaddingImage(padding_size=PADDING_BIG_IMAGE),
                FixedImageResize(self.image_size),
                ToTensor()
            ])


    def __getitem__(self, idx):
        return self.__get_item_by_idx(idx)

    def __get_item_by_idx(self, idx):
        image = self.imgs[idx]       

        #image = self.data['images'][image_idx]
        img_url = image['img_url'].split('/')
        image_file = img_url[-1]
        image_folder = img_url[-2]
        image_id = image['id']

        boxes = self.boxes[image_id].clone()
        labels = self.labels[image_id].clone()

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        num_objs = labels.shape[0]
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)


        fname = os.path.join(self.dataset_path, image_folder, image_file)
        
        with Image.open(fname) as f:
            img = f.convert('RGB')

        target = {
            "boxes": boxes,
            "original_boxes": boxes.clone(),
            "labels": labels,
            "image_id": image_id,
            "image_name":image_file,
            "image_path": fname,
            "area": area,
            "iscrowd": iscrowd,
            "image_part": "None",
            "width": img.size[0],
            "height": img.size[1]
        }

        # img = self.transforms(img)
        img,target = self.transforms(img, target)
        
        if self.test:
            return img, target, {"img_path":fname, "img_id":image['id']}
        else:
            return img, target

    

class PreprocessedDataset(Dataset):
    def __init__(self, dataset_path: str, is_training, image_size=1500, transforms=None, seed=42) -> None:
        super().__init__()
        self.seed = seed
        random.seed(seed)
        self.image_size = image_size        
        self.dataset_path = dataset_path
        self.is_training = is_training
        self.transforms = transforms if transforms is not None else self.get_transforms(is_training)

        self.images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        self.images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        self.images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))

        # self.val_size = val_size
        # random.shuffle(images)
        # ind_split = len(images) - int(len(images)*self.val_size)
        # if is_training:
        #     self.images = images[:ind_split]
        # else:
        #     self.images = images[ind_split:]


    def __len__(self):
        return len(self.images)

    def get_transforms(self, is_training):
        if is_training:
            return Compose([
                FixedImageResize(self.image_size),
                #PaddingImage(padding_size=20),
                ImageTransformCompose([
                    torchvision.transforms.RandomGrayscale(p=0.3),
                    torchvision.transforms.RandomApply([
                        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    ], p=0.5)
                ]),
                ToTensor()
            ])
        else:
            return Compose([
                FixedImageResize(self.image_size),
                #PaddingImage(padding_size=20),
                ToTensor()
            ])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        json_path = img_path.split(os.path.sep)
        json_path = os.path.join(os.path.sep.join(json_path[:-2]), "jsons", f"{os.path.splitext(json_path[-1])[0]}.json")

        with Image.open(img_path) as f:
            img = f.convert('RGB')

        with open(json_path) as f:
            target = json.load(f)

        target = {
            "boxes": torch.as_tensor(target["boxes"], dtype=torch.float32),
            "labels": torch.as_tensor(target["labels"], dtype=torch.int64),
            "image_id": torch.tensor(target["image_id"]),
            "image_name": target["image_name"],
            "area": torch.as_tensor(target["area"], dtype=torch.float32),
            "iscrowd": torch.as_tensor(target["iscrowd"], dtype=torch.int64),
            "image_part": "None",
            "width": img.size[0],
            "height": img.size[1]
        }

        if len(target["boxes"]) == 0:
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)

        img,target = self.transforms(img, target)
        
        return img, target



class ImagesDataset(Dataset):

    def __init__(self, dataset_path: str, image_size=1500, transforms=None):

        self.image_size = image_size
        self.dataset_path = dataset_path
        self.transforms = transforms if transforms is not None else self.get_transforms()

        self.images = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True)
        self.images.extend(glob.glob(os.path.join(dataset_path, '**', '*.JPG'), recursive=True))
        self.images.extend(glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True))

        # self.images = ["/".join(x.split(os.path.sep)[-2:]) for x in images]


    def __len__(self):
        return len(self.images)

    def get_transforms(self):
        return Compose([
            #LongRectangleCrop(),
            PaddingImage(padding_size=PADDING_BIG_IMAGE),
            FixedImageResize(self.image_size),
            ToTensor()
        ])
    
    def __getitem__(self, idx):
        image = self.images[idx]       

        image_id = torch.tensor([idx])
       
        with Image.open(image) as f:
            img = f.convert('RGB')

        target = {
            "boxes": torch.as_tensor([], dtype=torch.float32),
            "original_boxes": torch.as_tensor([], dtype=torch.float32),
            "labels": torch.as_tensor([], dtype=torch.int64),
            "image_id": image_id.detach(),
            "image_path": image,
            "image_name": os.path.basename(image),
            "area": torch.as_tensor([], dtype=torch.float32),
            "iscrowd": torch.as_tensor([], dtype=torch.int64),
            "image_part": "None",
            "width": img.size[0],
            "height": img.size[1]
        }

        # img = self.transforms(img)
        img, target = self.transforms(img, target)
        
        return img, target, {"img_path": image}





# import albumentations as A
# transform = A.Compose(
# [
#         # Trasformazioni Geometriche
#         A.HorizontalFlip(p=0.5), # Riflessione orizzontale
#         A.VerticalFlip(p=0.2),   # Riflessione verticale (meno probabile per testo standard)
#         A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT), # Rotazione leggera (max 15 gradi)
#         A.ShiftScaleRotate( # Combinazione di shift, scale e rotate
#             shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.8,
#             border_mode=cv2.BORDER_CONSTANT
#         ),
#         A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5), # Distorsioni elastiche
#         A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5), # Distorsioni a griglia

#         # Variazioni di Colore e Luminosità
#         A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7), # Variazione luminosità/contrasto
#         A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Aggiunta di rumore gaussiano
#         A.Blur(blur_limit=3, p=0.3), # Lieve sfocatura
#         A.CLAHE(clip_limit=4.0, tileGridSize=(8, 8), p=0.5), # Miglioramento contrasto adattivo

#         # Altre trasformazioni utili per testo
#         # A.Posterize(num_bits=4, p=0.3), # Riduzione della profondità di colore (può simulare stampa di bassa qualità)
#         # A.RandomGamma(gamma_limit=(80, 120), p=0.3), # Variazione gamma
#     ]