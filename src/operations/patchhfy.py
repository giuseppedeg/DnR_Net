import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from PIL import ImageDraw
import math



"""
NOTE
se l'immagine originale è troppo piccola e si usca un patch size piccolo si rischia di perdere alcuni bb 
Quelli ai bordi!

va in errore se il patch size è più grande dell'immagine originale
"""

def plot_patches(tensor):
    """
    plot image with all patches
    """
    # if len(tensor) == 1:
    #     n_patches = 1
    #     #tensor = tensor.unsqueeze(0)
    # else:   
    #     n_patches = len(tensor)

    n_patches = len(tensor)
    if n_patches > 20:
        n_patches = 20

    n_cols= math.ceil(n_patches/2/2)
    n_rows = math.ceil(n_patches / n_cols)

    fig = plt.figure(figsize=(n_rows, n_cols))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.1)

    for i, ax in enumerate(grid):
        if i < n_patches:
            patch = tensor[i].cpu().permute(1, 2, 0).numpy() 
            ax.imshow(patch)
            ax.axis('off')

    plt.show()


def plot_imgannot(imgtensor, target, title="title"):
    """
    plot an image with the relative boundingboxes
    """
    if len(imgtensor.shape) > 3:
        imgtensor = imgtensor.squeeze()
    image = ToPILImage()(imgtensor)

    if target is not None:

        img_dr = ImageDraw.Draw(image)

        for box in target['boxes']:

            img_dr.rectangle(box.tolist(), outline=(255,0,0), width=2) 
            #img_dr.text((annot['bbox'][0], annot['bbox'][1]), str(category_mapping[annot['category_id']]), font=font, fill=TEXT_BB_COLOR)

    image.show(title=title)


def depatchify(batch, orig_img_shape, padding=20):
    """
    Batch contains only patch of the same image!!
    """

    #imgs_batch = [[[],[]]] * len(orig_img_shape)
    imgs_batch = [ [[],[]] for _ in range(len(orig_img_shape))]
    ind_img_dic = {}

    images, targets = batch

    for i, t in zip(images, targets):

        if t['image_part'] != 'None':
            image_id = f"{t['image_part'].item()}_{t['image_name']}"
        else:
            image_id = f"{t['image_name']}"


        if image_id not in ind_img_dic:
            ind_img_dic[image_id] = len(ind_img_dic)

        
        imgs_batch[ind_img_dic[image_id]][0].append(i)
        imgs_batch[ind_img_dic[image_id]][1].append(t)


    rec_images = []
    rec_targets = []

    for b, image_shape in zip(imgs_batch, orig_img_shape):
        images, targets = b

        patch_size = images[0].shape[-1]-2*padding
        stride = 2 * int(patch_size/3)
        n_patches = len(images)

        c, h, w = image_shape

        # covered_w, covered_h = 0, 0
        # cols, rows = 0, 0
        # while(covered_w < w):
        #     covered_w = cols*stride + patch_size
        #     cols += 1
        # while(covered_h < h):
        #     covered_h = rows*stride + patch_size
        #     rows += 1
        
        cols = math.ceil((w-patch_size)/stride) + 1
        rows = math.ceil((h-patch_size)/stride) + 1
            
        padded_h = (rows-1) * stride + patch_size
        padded_w = (cols-1) * stride + patch_size

        fold = torch.nn.Fold(output_size=(padded_h, padded_w), kernel_size=patch_size, stride=stride )

        images = torch.stack(images)
        images = F.pad(images, (-padding, -padding, -padding, -padding), "constant", 1) 
        
        images = images.view((n_patches, -1))
        images = images.unsqueeze(0)

        images = images.permute(0, 2, 1)

        rec_tensor_img = fold(images)

        rec_tensor_img = F.pad(input=rec_tensor_img, pad=(0,w-padded_w,0,h-padded_h,0,0,0,0), mode='constant', value=1)
        rec_images.append(rec_tensor_img)
        """
        NOTE: The pixel values in overlapping zones are multiplied by the number of overlapping patches in the zone!
        """       
    
        rec_target = targets[0].copy()
        rec_target['boxes'] = rec_target['boxes'].tolist()
        rec_target['labels'] = rec_target['labels'].tolist()
        rec_target['area'] = rec_target['area'].tolist()
        rec_target['iscrowd'] = rec_target['iscrowd'].tolist()
        if "scores" in rec_target:
            rec_target['scores'] = rec_target['scores'].tolist()


        for ind, t in enumerate(targets[1:]):
            patch_ind = ind+1

            row = math.ceil((patch_ind+1)/cols) - 1

            col = patch_ind % cols
            row = math.ceil((patch_ind+1)/cols) - 1

            for b in t['boxes']:
                b[0] = b[0] + stride * col - padding
                b[1] = b[1] + stride * row - padding
                b[2] = b[2] + stride * col - padding
                b[3] = b[3] + stride * row - padding

                #b = b.unsqueeze(0)
                #rec_target['boxes'] = torch.cat([rec_target['boxes'], b], dim=0)
                b = b.tolist()
                if b not in  rec_target['boxes']:
                    rec_target['boxes'].append(b)
            
            if len(t['labels']) > 0:
                rec_target['labels'] += t['labels'].tolist()
                rec_target['area'] += t['area'].tolist()
                rec_target['iscrowd'] += t['iscrowd'].tolist()
                if 'scores' in t:
                    rec_target['scores'] += t['scores'].tolist()

        rec_target['boxes'] = torch.tensor(rec_target['boxes'])
        rec_target['labels'] = torch.tensor(rec_target['labels'])
        rec_target['area'] = torch.tensor(rec_target['area'])
        rec_target['iscrowd'] = torch.tensor(rec_target['iscrowd'])
        if "scores" in rec_target:
            rec_target['scores'] = torch.tensor(rec_target['scores'])
        
        rec_targets.append(rec_target)

    return rec_images, rec_targets


class Patchify(nn.Module):
    def __init__(self, patch_size=56, padding=20, padding_value=0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = 2*int(patch_size/3)
        self.padding = padding
        self.padding_value = padding_value
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=self.stride)

    def forward(self, x, target):
        len3 = False
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            len3 = True

        bs, c, h, w = x.shape

        n_col = math.ceil((w-self.patch_size)/self.stride) + 1
        n_row = math.ceil((h-self.patch_size)/self.stride) + 1

        # padding to match the patch size
        # if w == self.patch_size:
        #     wp == 0
        # else:
        #     wp = self.patch_size - (w - math.floor((w-self.patch_size+self.stride)/self.stride)*self.stride)
        # if h == self.patch_size:
        #     hp = 0
        # else:
        #     hp = self.patch_size - (h - math.floor((h-self.patch_size+self.stride)/self.stride)*self.stride)

        wp = self.patch_size+(n_col-1)*self.stride - w
        hp = self.patch_size+(n_row-1)*self.stride - h

        x = F.pad(input=x, pad=(0,wp,0,hp,0,0,0,0), mode='constant', value=self.padding_value)

        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        tensor_img_patched = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        #[N x,x,x ]

        tensor_img_patched = F.pad(tensor_img_patched, (self.padding, self.padding, self.padding, self.padding), "constant", self.padding_value)  # effectively zero padding

        if len3:
            tensor_img_patched = tensor_img_patched.squeeze(dim=0)
        

        new_targets = [{'boxes': None, 'labels': None, 'image_id': target['image_id'], 'image_name': target['image_name'], 'area': None, 'iscrowd': None, 'image_part': target['image_part'] } for _ in range(tensor_img_patched.shape[0])]
        #boxes = [ torch.tensor([]) for _ in range(tensor_img_patched.shape[0])]
        boxes = [ torch.empty((0, 4), dtype=torch.float32) for _ in range(tensor_img_patched.shape[0])]
        labels = [ [] for _ in range(tensor_img_patched.shape[0])]
        area = [ [] for _ in range(tensor_img_patched.shape[0])]
        iscrowd = [ [] for _ in range(tensor_img_patched.shape[0])]

        # n_col = math.ceil((w+wp-self.patch_size+self.stride)/(self.stride))
        # n_row = math.ceil((h+hp-self.patch_size+self.stride)/(self.stride))

        for ind, box in enumerate(target['boxes']):

            ind_cols, ind_rows = [], []

            for c in range(n_col):
                if c*self.stride < box[0] < c*self.stride+self.patch_size:
                    ind_cols.append(c)
            for r in range(n_row):
                if r*self.stride < box[1] < r*self.stride+self.patch_size:
                    ind_rows.append(r)

            for row_ind in ind_rows:
                for col_ind in ind_cols:
                    if (box[2] <= col_ind*self.stride+self.patch_size) and (box[3] <= row_ind*self.stride+self.patch_size):
                        ind_target = row_ind*n_col+col_ind

                        new_box = torch.empty(box.shape, dtype=box.dtype)
                        new_box[0] = box[0]-self.stride*col_ind + self.padding
                        new_box[1] = box[1]-self.stride*row_ind + self.padding
                        new_box[2] = box[2]-self.stride*col_ind + self.padding
                        new_box[3] = box[3]-self.stride*row_ind + self.padding

                        boxes[ind_target] = torch.cat([boxes[ind_target], new_box.unsqueeze(0)], dim=0)
                        labels[ind_target].append(target['labels'][ind].item())
                        area[ind_target].append(target['area'][ind].item())
                        iscrowd[ind_target].append(target['iscrowd'][ind].item())


        for ind, (box, lab, ar, iscr) in enumerate(zip(boxes, labels, area, iscrowd)):
            box = torch.reshape(box, (-1,4))
            lab = torch.tensor(lab)
            ar = torch.tensor(ar)
            iscr = torch.tensor(iscr)

            new_targets[ind]['boxes'] = box
            new_targets[ind]['labels'] = lab
            new_targets[ind]['area'] = ar
            new_targets[ind]['iscrowd'] = iscr
            
        return tensor_img_patched, new_targets
    

class Patchifier(nn.Module):
    def __init__(self, patchify, device) -> None:
        super().__init__()
        self.patchify = patchify
        self.device = device

    
    def patchify_batch(self, batch):
        images, targets = batch

        images = [x.to(self.device, non_blocking=True) for x in images]
        batch_patches = []
        batch_images = []
        batch_targets = []


        for img, tar in zip(images, targets):
            curr_img_patches = []

            patched_tensor, patched_targets = self.patchify(img, tar)
            for i, (p, t) in enumerate(zip(patched_tensor, patched_targets)):
                curr_img_patches.append(p)
                batch_images.append(p)
                batch_targets.append(t)
            
            batch_patches.append((curr_img_patches, patched_targets))
    
        return [batch_images, batch_targets]
    

    def forward(self, x):
        batch = x['data']
        patchified_batch = self.patchify_batch(batch)

        x['data'] = patchified_batch

        return x
                