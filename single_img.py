import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam,min_mask_region_area=100)

def split_single_img(path,outpath):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont,ImageFilter
    def segment_image(image, segmentation_mask):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
        segmented_image = Image.fromarray(segmented_image_array)
        black_image = Image.new("RGBA", image.size, (0, 0, 0,0))
        transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        transparency_mask[segmentation_mask] = 255
        transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
        black_image.paste(segmented_image, mask=transparency_mask_image)
        return black_image
    def convert_box_xywh_to_xyxy(box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        return [x1, y1, x2, y2]

    def smooth_edge(img):
        #smooth edge of image
        img = img.filter(ImageFilter.SMOOTH_MORE)
        return img

    def remove_small_dots_in_image(img,size):
        # convert to binary by thresholding
        ret, binary_map = cv2.threshold(img,127,255,0)

        # do connected components processing
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

        #get CC_STAT_AREA component as stats[label, COLUMN] 
        areas = stats[1:,cv2.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= size:   #keep
                result[labels == i + 1] = 255        
        
        return result


    import os
    # cropped_boxes = []
    image_path = path
    # save_path = "./crop/"
    
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    

    def fix_img(image):
        # 找到像素有效区域
        min_point = [image.width,image.height]
        max_point = [0,0]
        for x in range (0,image.width):
            for y in range (0,image.height):
                #------------------
                pixel = image.getpixel((x,y))
                if pixel[3] != 0:
                    if x < min_point[0]:
                        min_point[0] = x
                    elif x > max_point[0]:
                        max_point[0] = x
                    if y < min_point[1]:
                        min_point[1] = y
                    elif y > max_point[1]:
                        max_point[1] = y
                #------------------
        # print("min",min_point)
        # print("max",max_point)
        if max_point[0] - min_point[0] < 150:
            return None
        if max_point[1] - min_point[1] < 150:
            return None
        # 剔除无效区域
        image = image.crop((min_point[0],min_point[1],max_point[0],max_point[1]))
        bgsz = 600
        uppersz = 544 
        # 1创建一个空图像 600*600
        black_image = Image.new("RGBA", (bgsz,bgsz), (0, 0, 0,0))
        h = image.height
        w = image.width
        scale = 1
        if h>w:
            scale = uppersz/h
        else:
            scale = uppersz/w
        image = image.resize((round(w*scale),round(h*scale)))
        # cv2.resize(image,(h*scale,w*scale))
        result = black_image.copy()
        yoff = round((bgsz - h*scale)/2)
        xoff = round((bgsz - w*scale)/2)
        # result[yoff:yoff+h,xoff:xoff+w] = image
        black_image.paste(image,(xoff,yoff))
        return black_image




    image = Image.open(image_path)
    for i, mask in enumerate(masks):
        sub_img = segment_image(image, mask["segmentation"])
        # cropped_boxes.append(sub_img)
        # sub_img = smooth_edge(sub_img)
        # sub_img = remove_small_dots_in_image(sub_img,128)
        sub_img = fix_img(sub_img)
        if sub_img:
            fname = os.path.splitext( os.path.basename(image_path))[0]
           
            outsub = outpath+"/"+os.path.basename(os.path.dirname(image_path))
            if not os.path.exists(outsub):
                os.mkdir(outsub)
            sub_img.save(f'{outsub}/{fname}_{i}.png')