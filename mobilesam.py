import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor
from utility import sa1b_collate_fn, compute_iou
from torch.utils.data import DataLoader, Subset
from dataset import SA1BDataset
from transformers import SamProcessor
from tqdm import tqdm
from pycocotools import mask as mask_utils

def eval_mobile(predictor,dataloader,disable_verbose=False, prompt='p'):
    mIoU = []
    for idx,(_, images, labels, boxes, points) in enumerate(tqdm(dataloader, disable=disable_verbose)): 

        for i, image in enumerate(images):
            pts,bxs,gts = np.array(points[i]), np.array(boxes[i]), labels[i]
            image = np.transpose(image, (1, 2, 0)) 
            predictor.set_image(image)
            with torch.no_grad():
                    for j,pt in enumerate(pts):
                        if prompt == 'p':
                            masks, scores, logits = predictor.predict(
                                point_coords=np.array([pt]),
                                point_labels=np.array([1]),
                                multimask_output=True,
                            )
                        elif prompt == 'b':
                            masks, scores, logits = predictor.predict(
                                box=bxs,
                                multimask_output=True,
                            )
                        pred_1, pred_2, pred_3 = torch.from_numpy(masks[0]), \
                                                torch.from_numpy(masks[1]), \
                                                torch.from_numpy( masks[2])

                        gt = mask_utils.decode(gts[j]['segmentation'])
                        gt = torch.from_numpy(gt)
                        _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])
                        mIoU.append(max(mious))

            
    mIoU = torch.tensor(mIoU)
    average_mIoU = torch.mean(mIoU)
    return mIoU, average_mIoU


processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

test_dataset = SA1BDataset(f'datasets/SA1B', processor=processor, do_crop=False,label='all')

# Apply subset for shorter training

subset_dataset = Subset(test_dataset, indices=range(10000,10100,1))
test_dataloader = DataLoader(subset_dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn = sa1b_collate_fn) #collate_fn = sa1b_collate_fn


sam_checkpoint = "mobile_sam.pt"
model_type = "vit_t"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

ious, miou = eval_mobile(predictor,test_dataloader,disable_verbose=True,prompt='p')


print(f'miou : {round(miou.item(),4)}')

