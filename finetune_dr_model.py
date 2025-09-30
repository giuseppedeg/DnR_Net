# Load existing model (previously fine-tuned or from COCO)
model = fasterrcnn_resnet50_fpn_v2(weights=None)  # or load from checkpoint

model = FasterRCNN_DR("resnet50", C.N_CLASSES, image_size, dropout=dropout)
# load from checkpoint


# Replace classification head for new alphabet
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, new_num_classes)

# Optionally freeze the backbone if desired
for param in model.backbone.parameters():
    param.requires_grad = False