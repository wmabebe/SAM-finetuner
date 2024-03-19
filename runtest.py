import os
import random
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from transformers import SamModel, SamProcessor
from raffm import RaFFM
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Subset

# Set the PyTorch CUDA memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

# Initialize the original SAM model and processor
original_model = SamModel.from_pretrained("facebook/sam-vit-large").to("cuda")
processor = SamProcessor.from_pretrained("facebook/sam-vit-large")

# RaFFM configuration and submodel initialization
elastic_config = {
    "atten_out_space": [1280],
    "inter_hidden_space": [2048],
    "residual_hidden_space": [2048],
}
raffm_model = RaFFM(original_model, elastic_config=elastic_config)
submodel, params, config = raffm_model.random_resource_aware_model()
submodel = submodel.to("cuda")  # Move submodel to GPU

# Freeze vision_encoder and prompt_encoder in the submodel
for name, param in submodel.named_parameters():
    if name.startswith("mask_decoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

def get_image_info(dataset_directory, num_images=1):
    image_mask_pairs = []
    for filename in os.listdir(dataset_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(dataset_directory, filename)
            mask_filename = filename.replace(".jpg", ".json")
            mask_path = os.path.join(dataset_directory, mask_filename)
            if os.path.exists(mask_path):
                image_mask_pairs.append((image_path, mask_path))
    selected_pairs = random.sample(image_mask_pairs, min(num_images, len(image_mask_pairs)))
    return selected_pairs

def get_ground_truth_masks(mask_path):
    binary_masks = []
    with open(mask_path, 'r') as json_file:
        mask_data = json.load(json_file)
    for annotation in mask_data['annotations']:
        rle_mask = annotation['segmentation']
        binary_mask = coco_mask.decode(rle_mask)
        binary_masks.append(binary_mask)
    return binary_masks

def calculate_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    return F_loss.mean() if reduction == 'mean' else F_loss.sum()

def dice_loss(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice

def valid_points_from_masks(gt_masks):
    points = []
    for mask in gt_masks:
        ys, xs = np.where(mask > 0)
        points += [(x, y) for x, y in zip(xs, ys)]
    return points

def inference(model, dataloader):
    model.eval()
    image_ious = []
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to("cuda")
            masks = masks.to("cuda")

            outputs = model(images)['pred_masks']
            pred_masks = outputs.squeeze(1)
            pred_masks = pred_masks[:, 0, :, :].unsqueeze(1)
            pred_masks_resized = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear', align_corners=False)

            for i in range(images.shape[0]):
                gt_mask = masks[i].cpu().numpy().squeeze()
                pred_mask = pred_masks_resized[i].cpu().numpy().squeeze()

                valid_points = valid_points_from_masks([gt_mask])
                random_points = random.sample(valid_points, min(5, len(valid_points)))
                point_ious = []
                for point in random_points:
                    x, y = point
                    iou = calculate_metrics(pred_mask[y:y+256, x:x+256], gt_mask[y:y+256, x:x+256])
                    point_ious.append(iou)
                image_ious.append(sum(point_ious) / len(point_ious) if point_ious else 0)

    average_iou = sum(image_ious) / len(image_ious) if image_ious else 0
    return image_ious, average_iou

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory):
        self.image_mask_pairs = get_image_info(dataset_directory, num_images= 320 )  # Adjust num_images as needed
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize images to 1024x1024
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet
        ])

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = get_ground_truth_masks(mask_path)[0]  # Assuming there is at least one mask per image

        image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask = transforms.functional.resize(mask, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)  # Resize mask

        return image, mask

dataset = SA1BDataset("SA1B")
train_dataset = Subset(dataset, indices=range(0, 256, 1))
test_dataset = Subset(dataset, indices=range(256, 320, 1))
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Calculate IoUs and average IoU before training
image_ious_original, average_iou_original = inference(original_model, test_dataloader)
print('IoUs for original:')
for iou in image_ious_original:
    print(iou)
print(f'Average IoU original: {average_iou_original:.4f}')

image_ious_before, average_iou_before = inference(submodel, test_dataloader)
print('IoUs for each image before training submodel:')
for iou in image_ious_before:
    print(iou)
print(f'Average IoU before training submodel: {average_iou_before:.4f}')

# Training loop
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, submodel.parameters()), lr=1e-8)
# num_epochs = 100

# for epoch in range(num_epochs):
#     torch.cuda.empty_cache()
#     submodel.train()
#     running_loss = 0.0

#     for images, masks in train_dataloader:
#         images = images.to("cuda")
#         masks = masks.to("cuda")

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = submodel(images)['pred_masks']
#         pred_masks = outputs.squeeze(1)

#         # Select the appropriate channel and squeeze the channel dimension
#         pred_masks = pred_masks[:, 0, :, :].unsqueeze(1)

#         # Resize predicted masks to match target masks
#         pred_masks_resized = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear', align_corners=False)

#         # Compute the loss
#         loss_focal = focal_loss(pred_masks_resized, masks)
#         loss_dice = dice_loss(pred_masks_resized, masks)
#         loss = loss_focal + loss_dice

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     epoch_loss = running_loss / len(train_dataloader)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# # Calculate IoUs and average IoU after training
# image_ious_after, average_iou_after = inference(submodel, test_dataloader)
# print('IoUs for each image after training:')
# for iou in image_ious_after:
#     print(iou)
# print(f'Average IoU after training: {average_iou_after:.4f}')
