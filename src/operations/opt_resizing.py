import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import statistics
from PIL import Image




class Resizer:

    def __init__(self, model, device, ref_box_height=50) -> None:
        """
        params:
           model -> is the D&R model to estimate the size of the BB
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.ref_box_height = ref_box_height


    def load_model_state(self, checkpoint):
        checkpoint = torch.load(checkpoint, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.modelmame = checkpoint['name']
        

    def predict_shapes(self, batch):
        """ 
        Computes the avarage size of BB in the images in the given batch
        """
        predicted_shapes = {}

        images, targets = batch

        images = [x.to(self.device, non_blocking=True) for x in images]

        region_predictions = self.model(images, None)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in region_predictions]

        for target, output in zip(targets, outputs):
            # for box in output['boxes']:
            #     bb_width = box[2]-box[0]
            #     bb_height = box[3]-box[1]

            #     current_widths.append(bb_width.item())
            #     current_heights.append(bb_height.item())
            
            avg_box_height = (output['boxes'][:, 3] - output['boxes'][:, 1]).mean()
            var_box_height = (output['boxes'][:, 3] - output['boxes'][:, 1]).var()
            avg_box_width = (output['boxes'][:, 2] - output['boxes'][:, 0]).mean()
            var_box_width = (output['boxes'][:, 2] - output['boxes'][:, 0]).var()

            predicted_shapes[target["image_id"].item()] = {"mean": {"height": avg_box_height.item(), 
                                                                    "width": avg_box_width.item()},
                                                            "var":{"height": var_box_height.item(), 
                                                                    "width": var_box_width.item()}
                                                          }
        return predicted_shapes
    
    
    def image_resizer(self, batch):
        images, targets = batch

        predicted_shapes = self.predict_shapes(batch)

        res_images, res_targets, scald_sizes = [], [], []
        transform_toTensor = transforms.ToTensor()

        for image, target, (batchID, bb_stats) in zip(images, targets, predicted_shapes.items()):
            mean_h = bb_stats["mean"]["height"]
           
            # 20 is the padding
            mean_h_or = mean_h * (target["height"]+2*20) / image.shape[1]

            

            #### Resize current tensor
            scale_factor = self.ref_box_height / mean_h_or
            pil_img = Image.open(target["image_path"]).convert("RGB")
            original_width, original_height = pil_img.size
            new_width = int(original_width * (self.ref_box_height / mean_h_or))
            new_height = int(original_height * (self.ref_box_height / mean_h_or))
            resized_img = pil_img.resize((new_width, new_height))
            image = transform_toTensor(resized_img)
            ref_boxes = target['original_boxes']


            #### Resize current tensor
            # scale_factor = self.ref_box_height / mean_h
            # image = image.unsqueeze(dim=0)
            # image = F.interpolate(image, scale_factor=(scale_factor, scale_factor), mode='nearest')
            # image = image.squeeze()
            # ref_boxes = target['boxes']

            res_images.append(image)
            scald_sizes.append(image.shape)

            #scale bbs
            new_target = {'boxes': None, 
                          'labels': target['labels'], 
                          'image_id': target['image_id'], 
                          'image_name': target['image_name'], 
                          'area': None, 
                          'iscrowd': target['iscrowd'], 
                          'image_part': target['image_part'] }


            boxes = torch.empty(ref_boxes.shape, dtype=torch.float32)
            area = []

            for ind, box in enumerate(ref_boxes):
                new_box = box * scale_factor
                new_area = (new_box[2]-new_box[0]) * (new_box[3]-new_box[1])

                boxes[ind] = new_box
                area.append(new_area.item())

            new_target['boxes'] = boxes
            new_target['area'] = torch.tensor(area)


            res_targets.append(new_target)

        return [res_images, res_targets], scald_sizes
    
    
    def forward(self, x):
        batch = x["data"]

        resized_batch, scald_sizes = self.image_resizer(batch)

        x['data'] = resized_batch
        x['scald_sizes'] = scald_sizes

        return x
