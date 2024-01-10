from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from dataloaders import CocoDetection, MyCocoDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import requests
from transformers import SamModel, pipeline, SamProcessor, SamVisionConfig
from utility import calculate_metrics, point_grid
from raffm import RaFFM



# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

LIMIT = 5000
N = 8
MODEL = 'MODEL'

# path to your own data and coco file
data_dir = 'datasets/coco/val2017'
ann_file = 'datasets/coco/annotations/instances_val2017.json'


img_transform = transforms.Compose([transforms.ToTensor()])

coco_dataset  = CocoDetection(data_dir,ann_file,transform=get_transform(),target_transform=None)

print(coco_dataset)

#Initialize COCO object
coco = COCO(ann_file)

# Create a DataLoader with batch_size=1 since images have varying sizes
batch_size = 1
data_loader = DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=True)


#Setup SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)

## Subnetwork space
elastic_config = {
    "atten_out_space": [1280], # keep attention layer fixed
    "inter_hidden_space": [2048],
    "residual_hidden_space": [2048],
}
raffm_model = RaFFM(model.to("cpu"),elastic_config=elastic_config)
print("Original FM number of parameters:",raffm_model.total_params)

#Random sample a scaled FM
submodel, params, config = raffm_model.random_resource_aware_model()
print("subnetwork params",params)

# SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# generator = pipeline('mask-generation', model=submodel, image_processor=processor)

generator = pipeline("mask-generation", model=submodel if MODEL == 'SUBMODEL' else model, device=device, image_processor=processor.image_processor)


final_ious, final_precs, final_recs, final_f1s = [], [], [], []

# Iterate till (LIMIT) through the DataLoader
for idx, (images, targets) in enumerate(data_loader):
    if not targets:
        print(f'Empty targets: {targets}')
        continue
    #Print image and annotaiton info
    img_area = images.shape[2] * images.shape[3]
    
    #Predict
    to_pil = transforms.transforms.ToPILImage()
    image_pil = to_pil(images[0].squeeze(0))
    
    img_dim = tuple([images.shape[3],images.shape[2]])

    outputs = generator(image_pil, points_per_batch=64)
    masks = outputs['masks']
    
    print(f'image: {targets[0]["image_id"].item()}, shape:{images.shape}, anns:{len(targets)}, masks:{len(masks)}')
    
    #Grab annotation ids associated with this image
    annIds = []
    for target in targets:
        annIds.append(target['id'].item()) 
    anns = coco.loadAnns(annIds)
    
    gt_areas, avg_iou, avg_pre, avg_rec, avg_f1 = 0, 0, 0, 0, 0

    #For each annotation, compute metrics against every mask
    for ann in anns:
        area_gt = ann['area']
        gt_areas += area_gt
        mask_gt = coco.annToMask(ann)
        mask_gt[mask_gt >= 0.5] = 1
        mask_gt[mask_gt < 0.5] = 0
        ious, recs, precs, f1s = [], [], [], []
        for mask in masks:
            # mask_pred = mask[0, 0, :, :]
            # mask_pred = mask_pred.numpy()
            iou, rec, prec, f1 = calculate_metrics(mask,mask_gt)
            #Only print values with iou>50%
            if iou > 0:
                ious.append(iou)
                recs.append(rec)
                precs.append(prec)
                f1s.append(f1)
        mean_iou = sum(ious) / len(ious) if ious else 0
        mean_rec = sum(recs) / len(recs) if recs else 0
        mean_pre = sum(precs) / len(precs) if precs else 0
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0
        
        print(f'\tAnn_id: {ann["id"]:<20} Ann_area={round(area_gt, 4):<20} m_iou={round(mean_iou, 4):<20} m_rec={round(mean_rec, 4):<20} m_pre={round(mean_pre, 4):<20} m_f1={round(mean_f1,4):<20}')         

        avg_iou += (area_gt / gt_areas) * mean_iou * 100
        avg_pre += (area_gt / gt_areas) * mean_pre * 100
        avg_rec += (area_gt / gt_areas) * mean_rec * 100
        avg_f1 += (area_gt / gt_areas) * mean_f1 * 100
        
    print(f'\tSummary: Img_area={round(img_area, 4):<20} m_iou={round(avg_iou, 2):<20} m_rec={round(avg_rec, 2):<20} m_pre={round(avg_pre, 2):<20} m_f1={round(avg_f1,2):<20}')         

    final_ious.append(avg_iou)
    final_recs.append(avg_rec)
    final_precs.append(avg_pre)
    final_f1s.append(avg_f1)
    
    #Save random image and break loop
    if idx == LIMIT:
        image_np = transforms.ToPILImage()(images[0].squeeze(0))
        plt.axis('off')
        plt.imshow(image_np)
        plt.savefig("test.jpg")
        break

final_iou = sum(final_ious) / len(final_ious) if final_ious else 0
final_rec = sum(final_recs) / len(final_recs) if final_recs else 0
final_pre = sum(final_precs) / len(final_precs) if final_precs else 0
final_f1 = sum(final_f1s) / len(final_f1s) if final_f1s else 0

print(f'\n\tFinal: model:{MODEL} Img_area={round(img_area, 4):<20} m_iou={round(final_iou, 2):<20} m_rec={round(final_rec, 2):<20} m_pre={round(final_pre, 2):<20} m_f1={round(final_f1,2):<20}')         
