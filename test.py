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
from utility import calculate_metrics
from raffm import RaFFM



# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)


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
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)

## Subnetwork space
elastic_config = {
    "atten_out_space": [512, 768, 1280], # keep attention layer fixed
    "inter_hidden_space": [512, 1024, 1280, 2048],
    "residual_hidden_space": [1024, 1280, 2048],
}
raffm_model = RaFFM(model.to("cpu"),elastic_config=elastic_config)
print("Original FM number of parameters:",raffm_model.total_params)

#Random sample a scaled FM
submodel, params, config = raffm_model.random_resource_aware_model()
print("subnetwork params",params)

generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device=device)

LIMIT = 3

# Iterate till (LIMIT) through the DataLoader
for idx, (images, targets) in enumerate(data_loader):
    #Print image and annotaiton info
    print(f'image: {targets[0]["image_id"]}, shape:{images.shape}, anns:{len(targets)}')
    
    #Predict
    to_pil = transforms.transforms.ToPILImage()
    image_pil = to_pil(images[0].squeeze(0))

    with torch.no_grad():
        outputs = generator(image_pil, points_per_batch=16)

    masks = outputs['masks']
    print("masks",len(masks))
    
    img_dim = tuple([images.shape[3],images.shape[2]])
    
    #Grab annotation ids associated with this image
    annIds = []
    for target in targets:
        annIds.append(target['id'].item()) 
    anns = coco.loadAnns(annIds)
    
    #For each annotation, compute metrics against every mask
    for ann in anns:
        area_gt = target['area']
        mask_gt = coco.annToMask(ann)
        mask_gt[mask_gt >= 0.5] = 1
        mask_gt[mask_gt < 0.5] = 0
        for mask in masks:
            iou, rec, pre, f1 = calculate_metrics(mask,mask_gt)
            #Only print values with iou>50%
            if iou > .5:
                print(f'\tiou={round(iou, 4):<20} rec={round(rec, 4):<20} pre={round(pre, 4):<20} f1={round(f1,4):<20}')

    #Save random image and break loop
    if idx == LIMIT:
        image_np = transforms.ToPILImage()(images[0].squeeze(0))
        plt.axis('off')
        plt.imshow(image_np)
        plt.savefig("test.jpg")
        break
