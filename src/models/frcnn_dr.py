import torch
import gc
from torch import nn
from torchvision.models import  resnet101, ResNet101_Weights, resnet34, ResNet34_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, \
    fasterrcnn_mobilenet_v3_large_fpn, FastRCNNConvFCHead, FasterRCNN
from torchvision.models.detection.rpn import RPNHead
from src.coco.cocoeval import COCOeval
from src.coco.coco_eval import summarizeCustom
from src.utils.coco_utils import convert_img_to_coco_api
from datetime import datetime
from src.utils.box_drawer import draw_boxes
import torchvision.transforms as T
import os
import shutil
import contextlib
from tqdm import tqdm
import time
import _config as C

import matplotlib.pyplot as plt


class FasterRCNN_DR(nn.Module):
    def __init__(self, model_type, n_classes, img_size, dropout=0.5):
        self.dropout = dropout
        self.img_size = img_size
        super().__init__()
        if model_type == 'resnet50':
            # model = fasterrcnn_resnet50_fpn_v2(weights="FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT", min_size=img_size, trainable_backbone_layers=5,
            #                                    max_size=img_size, rpn_batch_size_per_image=256,
            #                                    box_batch_size_per_image=512,
            #                                    box_nms_thresh=0.5, box_score_thresh=0.2,
            #                                    box_fg_iou_thresh=0.65, box_bg_iou_thresh=0.5,
            #                                    box_positive_fraction=0.3,
            #                                    box_detections_per_img=320)
            
            model = fasterrcnn_resnet50_fpn_v2(weights="FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
                                               min_size=img_size, trainable_backbone_layers=5,
                                               max_size=img_size, rpn_batch_size_per_image=256,
                                               box_batch_size_per_image=512,
                                               box_nms_thresh=0.5, box_score_thresh=0.2,
                                               box_fg_iou_thresh=0.65, box_bg_iou_thresh=0.5,
                                               box_positive_fraction=0.3,
                                               box_detections_per_img=320)
            
        elif model_type == 'resnet101':
            backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, progress=True)
            backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=nn.BatchNorm2d)
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
            box_head = FastRCNNConvFCHead(
                (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
            )
            model = FasterRCNN(
                backbone,
                num_classes=n_classes + 1,
                rpn_anchor_generator=rpn_anchor_generator,
                rpn_head=rpn_head,
                box_head=box_head,
                min_size=img_size,
                max_size=img_size, rpn_batch_size_per_image=256,
                box_batch_size_per_image=512,
                box_nms_thresh=0.5, box_score_thresh=0.2,
                box_positive_fraction=0.4,
                box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                box_detections_per_img=320
            )
        elif model_type == 'resnet34':
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1, progress=True)
            backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=nn.BatchNorm2d)
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
            box_head = FastRCNNConvFCHead(
                (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
            )
            model = FasterRCNN(
                backbone,
                num_classes=n_classes + 1,
                rpn_anchor_generator=rpn_anchor_generator,
                rpn_head=rpn_head,
                box_head=box_head,
                min_size=img_size,
                max_size=img_size, rpn_batch_size_per_image=256,
                box_batch_size_per_image=512,
                box_nms_thresh=0.5, box_score_thresh=0.2,
                box_positive_fraction=0.4,
                box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                box_detections_per_img=320
            )
        elif model_type == 'mobinet':
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=img_size, trainable_backbone_layers=6,
                                                      max_size=img_size, rpn_batch_size_per_image=256,
                                                      box_batch_size_per_image=512,
                                                      box_nms_thresh=0.5, box_score_thresh=0.2,
                                                      box_positive_fraction=0.4,
                                                      box_fg_iou_thresh=0.75, box_bg_iou_thresh=0.5,
                                                      box_detections_per_img=320)
        else:
            raise Exception(f'Arch {model_type} is not implemented')

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        all_classes = n_classes + 1  # +1 class for background
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, all_classes)
        model.roi_heads.box_predictor.cls_score = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, all_classes)
        )

        self.network = model


    def forward(self, x, y=None):
        return self.network(x, y)




class Trainer:
    def __init__(self, model, device) -> None:
        self.model = model
        self.model.to(device)
        self.device = device
        self.modelmame = datetime.today().strftime('%Y-%m-%d_%H-%M')
        self.best_val_loss = 100
        self.best_val_epoch = 0

        COCOeval.summarizeCustom = summarizeCustom


    def set_train(self):
        self.model.train()


    def set_eval(self):
        self.model.eval()


    def load_state(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model'])
        self.modelmame = checkpoint['name']
        self.best_val_loss = checkpoint['best_val_loss']
        if 'best_val_epoch' in checkpoint:
            self.best_val_epoch = checkpoint['best_val_epoch']


    def validate(self, epoch, validation_loader):
        """
        it compute the performance metrices on the validation set.
        The metrics are defined in the ICDAR2023 competition on detection and recognition on papyri paper

        returns:
           - labelscore    -> [0,1] is the score of the detection and recognition
           - nolabelscore  -> [0,1] is the score of only the recognition
        """
        self.model.eval()

        with torch.no_grad():
            res = []
            gt_boxes = []
            for batch in tqdm(validation_loader, desc=f"Valid {epoch}", leave=False, disable=C.DISABLE_TQDM):
                images, targets = batch

                images = [x.to(self.device, non_blocking=True) for x in images]

                region_predictions = self.model(images, targets)
                
                outputs = [{k: v.to("cpu") for k, v in t.items()} for t in region_predictions]
                
                current_res = []
                current_gt_boxes = []
                for target, output in zip(targets, outputs):
                    for box, label, score in zip(output['boxes'], output['labels'],output['scores']):
                        current_res.append({
                            "image_id": target["image_id"].item(),
                            "category_id": label.item(), 
                            'bbox': box.tolist(),
                            'score':score.item()
                        })
                    
                    for box, label in zip(target['boxes'], target['labels']):
                        current_gt_boxes.append({
                            "image_id": target["image_id"].item(),
                            'category_id': label.item(),
                            'bbox': box.tolist(),
                            'score': 1
                            })
                gt_boxes += current_gt_boxes
                res += current_res

        if len(res) > 0:
            with contextlib.redirect_stdout(None):
                cocoGT = convert_img_to_coco_api(gt_boxes)
                s = set()
                for r in res:
                    s.add(r['category_id'])
                print(s)
                cocoDt = cocoGT.loadRes(res) # pass only the annotations

                cocoEval = COCOeval(cocoGT, cocoDt, iouType="bbox")
                cocoEval.params.maxDets = [10000]

                # Detection and Recognition eval
                cocoEval.evaluate(verbose=False)
                cocoEval.accumulate(verbose=False)
                cocoEval.summarizeCustom(verbose=False)
                labelscore = cocoEval.stats[0]

                # Only Detection eval
                cocoEval.params.useCats = False
                cocoEval.evaluate(verbose=False)
                cocoEval.accumulate(verbose=False)
                cocoEval.summarizeCustom(verbose=False)
                nolabelscore = cocoEval.stats[0]

                val_score = (labelscore, nolabelscore)
        
        else:
            val_score = (float('inf'), float('inf'))

        return val_score


    def train(self, checkpoint_folder, optimizer, train_loader, validation_loader=None, checkpoint=None, epochs=5, iters_verb=50):
        """
        Train the model.

        params:
            checkpoint: if not None, the training will continue from the checkpoint
        """

        epoch_start = 0

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, weights_only=False)
            epoch_start = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            curr_lr = optimizer.param_groups[0]['lr']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if optimizer.param_groups[0]['lr'] != curr_lr:
                for g in optimizer.param_groups:
                    g['lr'] = curr_lr
            self.modelmame = checkpoint['name']
            self.best_val_loss = checkpoint['best_val_loss']
            if 'best_val_epoch' in checkpoint:
                self.best_val_epoch = checkpoint['best_val_epoch']

        os.makedirs(os.path.join(checkpoint_folder, self.modelmame, "models"), exist_ok=True)

        with open(os.path.join(checkpoint_folder, self.modelmame, f"params_{datetime.today().strftime('%Y-%m-%d_%H-%M')}.txt" ), "w") as f:
            f.write(f"modelmame: {self.modelmame}\n")
            f.write(f"modelType: {C.FRCNN_mod}\n")
            f.write(f"epoch_start: {epoch_start}\n")
            f.write(f"epochs: {epochs}\n")
            f.write(f"Dataset: {C.PREPROCESSED_DATASET_PATH}\n")
            f.write(f"model image size: {self.model.img_size}\n")
            f.write(f"Referenxe box height: {C.REF_BOX_HEIGHT}\n")
            f.write(f"dropout: {self.model.dropout}\n")
            f.write(f"optimizer: {optimizer}\n")
            f.write(f"LR: {optimizer.param_groups[0]['lr']}\n")
            f.write(f"Batch size: {train_loader.batch_size}\n")
            f.write(f"len training: {len(train_loader)}\n")
            f.write(f"len validation: {len(validation_loader)}\n\n")
            f.write(f"Categories: {C.CATEGORIES}\n")



        with open(os.path.join(checkpoint_folder, self.modelmame, f"{self.modelmame}.log" ), "a") as lof_f:
            for epoch in tqdm(range(epoch_start, epochs), desc="Training", disable=C.DISABLE_TQDM):
                start_time = time.time()
                self.model.train()
                
                total_train_loss = 0
                for iter_tr, (images, targets) in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False, disable=C.DISABLE_TQDM)):
                    images = [img.to(self.device) for img in images]
                    targets = [{
                        "boxes": t["boxes"].to(self.device),
                        "labels": t["labels"].to(self.device),
                        "image_id": t["image_id"].to(self.device),
                        "image_name": t["image_name"],
                        "area": t["area"].to(self.device),
                        "iscrowd": t["iscrowd"].to(self.device),
                        "image_part": t["image_part"]
                    } for t in targets]


                    # for img in images:
                    #     plt.imshow(img.detach().cpu().permute(1, 2, 0))
                    #     plt.show()

                    loss_dict = self.model(images, targets)
                    if not isinstance(loss_dict, dict) or len(loss_dict) == 0:
                        tqdm.write("⚠️  Warning: Empty loss_dict, skipping batch")
                        continue
                    losses = sum(loss for loss in loss_dict.values()) / len(loss_dict)

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    total_train_loss += losses.item()

                    # allocated_memory_bytes = torch.cuda.memory_allocated(self.device)
                    # print(f"Memoria allocata (Byte): {allocated_memory_bytes}")
                    # print(f"Memoria allocata (MB): {allocated_memory_bytes / (1024**2):.2f} MB")
                    # print(f"Memoria allocata (GB): {allocated_memory_bytes / (1024**3):.2f} GB")

                    # # Memoria totale riservata (caching allocator)
                    # reserved_memory_bytes = torch.cuda.memory_reserved(self.device)
                    # print(f"Memoria riservata (MB): {reserved_memory_bytes / (1024**2):.2f} MB")
                    # print(f"Memoria riservata (GB): {reserved_memory_bytes / (1024**3):.2f} GB")

                    # # Memoria massima allocata durante l'esecuzione corrente (picco)
                    # max_allocated_memory_bytes = torch.cuda.max_memory_allocated(self.device)
                    # print(f"Memoria massima allocata (MB): {max_allocated_memory_bytes / (1024**2):.2f} MB")

                    # # Memoria massima riservata (picco)
                    # max_reserved_memory_bytes = torch.cuda.max_memory_reserved(self.device)
                    # print(f"Memoria massima riservata (MB): {max_reserved_memory_bytes / (1024**2):.2f} MB")

                    # # Puoi resettare le statistiche di picco
                    # torch.cuda.reset_peak_memory_stats(self.device)
                    # print("\nStatistiche di picco resettate.")
                    # print(f"Memoria massima allocata dopo reset (MB): {torch.cuda.max_memory_allocated(self.device) / (1024**2):.2f} MB\n\n")

                    # print("\n\n--- Riepilogo dettagliato della memoria GPU ---")
                    # print(torch.cuda.memory_summary(device=self.device, abbreviated=False))


                    if iter_tr % iters_verb == 0:
                        lof_f.write(f"      Epoch: {epoch}, Iteration: {iter_tr}, Train Loss: {losses.item():.4f}, Classifier: {loss_dict['loss_classifier']}, Box Reg: {loss_dict['loss_box_reg']:.4f}, Objectness: {loss_dict['loss_objectness']:.4f}, Rpn Box Reg: {loss_dict['loss_rpn_box_reg']:.4f}\n")

                    del images, targets, loss_dict, losses # Cancella le variabili che non ti servono più
                    torch.cuda.empty_cache() # Svuota la cache della GPU
                    gc.collect() # Raccoglie la spazzatura Python (può aiutare indirettamente)
                    # reserved_memory_bytes = torch.cuda.memory_reserved(self.device)
                    # print(f"Memoria riservata (MB): {reserved_memory_bytes / (1024**2):.2f} MB")
                    # print(f"Memoria riservata (GB): {reserved_memory_bytes / (1024**3):.2f} GB")

                val_loss = -1
                if validation_loader:
                    (labelscore, nolabelscore) = self.validate(epoch, validation_loader)
                    if (labelscore == float("inf")) or (nolabelscore == float("inf")):
                        val_loss = float("inf")
                    else:
                        val_loss = ((1-labelscore)+(1-nolabelscore)) / 2

                save_best = False
                if (val_loss != -1) and (val_loss < self.best_val_loss):
                    lof_f.write(f"\nBest Model epoch {epoch}, val_loss= {val_loss:.4f}, train_loss: {total_train_loss:.4f}\n\n")
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch
                    save_best = True


                lof_f.write(f"Epoch {epoch}, time: {time.time() - start_time} sec,  Train Loss: {total_train_loss:.4f}, Validation Loss: {val_loss:.4f}\n\n")
                lof_f.flush()

                # save model weights
                checkpoint = {
                    "name": self.modelmame,
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'best_val_epoch': self.best_val_epoch,
                    #'lr_sched': lr_sched
                }
                # torch.save(checkpoint, os.path.join(checkpoint_folder, self.modelmame, "models", f"{str(epoch).zfill(4)}.pth"))
                torch.save(checkpoint, os.path.join(checkpoint_folder, self.modelmame, "models", f"last.pth"))
                if save_best:
                    torch.save(checkpoint, os.path.join(checkpoint_folder, self.modelmame, "models", f"best.pth"))
        
        return  {
                "name": self.modelmame,
                'best_val_loss': self.best_val_loss,
                'best_val_epoch': self.best_val_epoch,
                }   

    def test(self, test_dataloader, checkpoint=None, plt_imgs=False, save_imgs=None):
        if plt_imgs:
            if os.path.exists(os.path.join(save_imgs, "imgs")):
                shutil.rmtree(os.path.join(save_imgs, "imgs"))
            os.makedirs(os.path.join(save_imgs, "imgs"))

        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.modelmame = checkpoint['name']
            self.best_val_loss = checkpoint['best_val_loss']
    
        self.model.eval()

        with torch.no_grad():
            res = []
            gt_boxes = []
            for id, batch in enumerate(tqdm(test_dataloader, desc=f"Test", leave=False, disable=C.DISABLE_TQDM)):
                images, targets = batch

                images = [x.to(self.device, non_blocking=True) for x in images]

                region_predictions = self.model(images, targets)
                
                outputs = [{k: v.to("cpu") for k, v in t.items()} for t in region_predictions]

                current_res = []
                for target, output in zip(targets, outputs):
                    for box, label, score in zip(output['boxes'], output['labels'],output['scores']):
                        current_res.append({
                            "image_id": target["image_id"].item(),
                            "category_id": label.item(), 
                            'bbox': box.tolist(),
                            'score':score.item()
                        })
                res += current_res

                current_gt_boxes = []
                for box, label in zip(targets[0]['boxes'], targets[0]['labels']):
                    current_gt_boxes.append({
                        "image_id": targets[0]["image_id"].item(),
                        'category_id': label.item(),
                        'bbox': box.tolist(),
                        'score': 1
                        })
                gt_boxes += current_gt_boxes

                if plt_imgs:
                    transf = T.ToPILImage()

                    or_img = transf(images[0]) #images[0].cpu().permute(1, 2, 0)

                    CURR_img = draw_boxes(or_img, current_res, n_categories=24)
                    GT_img = draw_boxes(or_img, current_gt_boxes, n_categories=24)

                    or_img.save(os.path.join(save_imgs, "imgs", f"{str(id).zfill(4)}_original.jpg"))
                    CURR_img.save(os.path.join(save_imgs, "imgs", f"{str(id).zfill(4)}_predicted.jpg"))
                    GT_img.save(os.path.join(save_imgs, "imgs", f"{str(id).zfill(4)}_gt.jpg"))

                    # or_img.show(title="Original")
                    # CURR_img.show(title="Predicted")
                    # GT_img.show(title="GT")
                    # input("ENTER to continue...")
                
        if len(res) > 0:
            with contextlib.redirect_stdout(None):
                cocoGT = convert_img_to_coco_api(gt_boxes)
                cocoDt = cocoGT.loadRes(res) # pass only the annotations
                cocoEval = COCOeval(cocoGT, cocoDt, iouType="bbox")
                cocoEval.params.maxDets = [10000]

                # Detection and Recognition eval
                cocoEval.evaluate(verbose=False)
                cocoEval.accumulate(verbose=False)
                cocoEval.summarizeCustom(verbose=False)
                labelscore = cocoEval.stats[0]

                # Only Detection eval
                cocoEval.params.useCats = False
                cocoEval.evaluate(verbose=False)
                cocoEval.accumulate(verbose=False)
                cocoEval.summarizeCustom(verbose=False)
                nolabelscore = cocoEval.stats[0]

                test_score = (labelscore, nolabelscore)
        else:
            test_score = (float('inf'), float('inf'))


        return test_score
        

        

if __name__ == "__main__":
    print("D&R test")