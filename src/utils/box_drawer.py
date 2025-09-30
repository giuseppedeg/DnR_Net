from PIL import ImageDraw, ImageFont
import distinctipy

OUT_FOLDER = "data/_OUT"
OUT_IMAGE_FOLDER = "data/_OUT/imgs"
EXCLUDE = ["imgs", "predictions", "TM062931_TimotheusPersae_crop3"]

PREDICTION_FOLDER = "TM065797_The_Curse_of_Artemisia_Fragment_WDL4310"

FONT = "assets/Anonymous_Pro.ttf"
WIDHT_BB = 4
CHARACTER_LABEL_SIZE = 30
TEXT_BB_COLOR = (10,10,10)


def draw_boxes(img, boxes, n_categories=100):

    boxes_image = img.copy() 
     
    font = ImageFont.truetype(font=FONT,size=CHARACTER_LABEL_SIZE)

    color_categories = {}
    colors = distinctipy.get_colors(n_categories)

    for cat in range(1, n_categories+1):
        col = colors[cat-1]
        color_categories[cat] = (int(255*col[0]), int(255*col[1]), int(255*col[2]))

    if boxes_image.mode != "RGB":  
        boxes_image = boxes_image.convert('RGB')
    
    img_dr = ImageDraw.Draw(boxes_image)   

    for box in boxes:
        
        shape = [box['bbox'][0], box['bbox'][1], box['bbox'][2], box['bbox'][3]]
        color = color_categories[box['category_id']]

        img_dr.rectangle(shape, outline=color, width=WIDHT_BB) 
        img_dr.text((box['bbox'][0], box['bbox'][1]), str(box['category_id']), font=font, fill=TEXT_BB_COLOR)

    
    return boxes_image
    


# if __name__ == "__main__":
#     print("Making bouding box images from characters detection...")
#     if os.path.exists(OUT_IMAGE_FOLDER):
#         shutil.rmtree(OUT_IMAGE_FOLDER)
#     os.makedirs(OUT_IMAGE_FOLDER)

#     font = ImageFont.truetype(font=FONT,size=CHARACTER_LABEL_SIZE)
    
#     for coco_prediction in os.listdir(os.path.join(OUT_FOLDER, PREDICTION_FOLDER)):
#          if coco_prediction not in EXCLUDE:
#             os.mkdir(os.path.join(OUT_IMAGE_FOLDER,coco_prediction.split(".")[0]))
#             with open(os.path.join(OUT_FOLDER, PREDICTION_FOLDER, coco_prediction), encoding="utf-8") as f:
#                 cocoj = json.load(f)


#             color_categories = {}
#             n_categories = len(cocoj['categories'])

#             colors = distinctipy.get_colors(n_categories)
#             # display the colours
#             #distinctipy.color_swatch(colors)
#             for idx, cat in enumerate(cocoj['categories']):
#                 col = colors[idx]
#                 color_categories[cat['id']] = (int(255*col[0]), int(255*col[1]), int(255*col[2]))
            
#             category_mapping = {}
#             for cat in cocoj["categories"]:
#                 category_mapping[cat['id']] = cat['name']


#             for image in tqdm(cocoj["images"]):
#                 id_image = image['id']
#                 image_name = image['file_name']
#                 image_url = image['img_url']
                
#                 img = Image.open(image_url)
#                 if img.mode != "RGB":  
#                     img = img.convert('RGB')
#                 img_dr = ImageDraw.Draw(img)   

#                 for annot in cocoj["annotations"]:
#                     if annot['image_id'] == id_image:
#                         shape = [annot['bbox'][0], annot['bbox'][1], annot['bbox'][0]+annot['bbox'][2], annot['bbox'][1]+annot['bbox'][3]]
#                         color = color_categories[annot['category_id']]

#                         img_dr.rectangle(shape, outline=color, width=WIDHT_BB) 
#                         img_dr.text((annot['bbox'][0], annot['bbox'][1]), str(category_mapping[annot['category_id']]), font=font, fill=TEXT_BB_COLOR)

#                 #img.show() 
#                 img.save(os.path.join(OUT_IMAGE_FOLDER,coco_prediction.split(".")[0], image_name))
#                 img.close()

#     print("Done")