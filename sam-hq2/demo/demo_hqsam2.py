import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()


if __name__ == "__main__":
    checkpoint = "./checkpoints/sam2.1_hq_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    for i in range(1,5):
        print("image:   ",i)
        # hq_token_only: False means use hq output to correct SAM output. 
        #                True means use hq output only. 
        #                Default: False
        hq_token_only = False 
        # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
        # For images contain single object, we suggest to set hq_token_only = True
        # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

        image = cv2.imread('./demo/input_images/example'+str(i)+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        if i==1:
            input_box = np.array([[306, 132, 925, 893]])
            input_point, input_label = None, None
        elif i==2:
            input_point = np.array([[495,518],[217,140]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==3:
            input_box = np.array([[64,76,940,919]])
            input_point, input_label = None, None
        elif i==4:
            # multi box input
            input_box = torch.tensor([[45,260,515,470], [310,228,424,296]],device=predictor.device)
            # transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            input_point, input_label = None, None

        batch_box = False if input_box is None else len(input_box)>1 
        result_path = 'demo/hq_sam_result_vis/'
        os.makedirs(result_path, exist_ok=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(point_coords=input_point,
                                            point_labels=input_label,
                                            box=input_box,
                                            multimask_output=False, hq_token_only=hq_token_only)

            if not batch_box:
                show_res(masks,scores,input_point, input_label, input_box, result_path + 'example'+str(i), image)
            else:
                masks = masks.squeeze(1)
                scores = scores.squeeze(1)
                input_box = input_box.cpu().numpy()
                show_res_multi(masks, scores, input_point, input_label, input_box, result_path + 'example'+str(i), image)

        

    



