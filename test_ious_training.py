import os
import random
import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from transformers import SamModel, SamProcessor, pipeline
from raffm import RaFFM
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# Initialize the original SAM model and processor
original_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Initialize the mask generation pipeline for the original SAM model
original_generator = pipeline("mask-generation", model=original_model, device="cuda", image_processor=processor.image_processor)

# RaFFM configuration and submodel initialization
elastic_config = {
    "atten_out_space": [1280],
    "inter_hidden_space": [2048],
    "residual_hidden_space": [2048],
}
raffm_model = RaFFM(original_model, elastic_config=elastic_config)
submodel, params, config = raffm_model.random_resource_aware_model()
submodel_generator = pipeline("mask-generation", model=submodel, device="cuda", image_processor=processor.image_processor)

models = {"submodel": submodel, "original_model": original_model}

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

def point_in_mask(point, mask):
    x, y = point
    return mask[y, x] > 0

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean() if reduction == 'mean' else F_loss.sum()

def dice_loss(inputs, targets, smooth=1e-6):
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice


def calculate_average_iou(model, dataloader, processor):
    total_iou = 0
    iou_count = 0
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to("cuda")
            masks = masks.to("cuda")
            for i in range(images.shape[0]):
                input_image = images[i].unsqueeze(0)
                input_mask = masks[i].unsqueeze(0)
                # Resize the image to the expected input size of the model
                resized_image = torch.nn.functional.interpolate(input_image, size=(1024, 1024), mode='bilinear', align_corners=False)
                # Process the image through the model
                outputs = model(resized_image)
                # Create reshaped_input_sizes as a list of tuples
                reshaped_input_sizes = [(resized_image.shape[2], resized_image.shape[3])] * len(outputs.pred_masks)
                pred_masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(), input_image.shape[-2:], reshaped_input_sizes
                )
                pred_mask = pred_masks[0].squeeze(0).squeeze(0).cpu().numpy()
                gt_mask = input_mask.squeeze(0).cpu().numpy()
                iou = calculate_metrics(pred_mask, gt_mask)
                total_iou += iou
                iou_count += 1
    average_iou = total_iou / iou_count if iou_count > 0 else 0
    return average_iou


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))  # Change from (C, H, W) to (H, W, C)
    tensor = tensor * std + mean  # Denormalize
    tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)  # Convert to [0, 255]
    return tensor

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory):
        self.image_mask_pairs = get_image_info(dataset_directory, num_images=320)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = get_ground_truth_masks(mask_path)[0]
        image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = transforms.functional.resize(mask, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask

# Define the test_dataset
dataset = SA1BDataset("SA1B")
train_dataset = Subset(dataset, indices=range(0, 256))
test_dataset = Subset(dataset, indices=range(256, 320))
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Use all test_dataset images for initial IoU calculations
selected_images = [(test_dataset.dataset.image_mask_pairs[i][0], test_dataset.dataset.image_mask_pairs[i][1]) for i in range(len(test_dataset))]

# Initialize variables to store the total IoUs and counts for averaging
total_iou_submodel = 0
total_iou_original = 0
iou_count = 0

# Process each selected image and its corresponding mask
# Initialize variables to store the total IoUs and counts for averaging
total_iou_submodel = 0
total_iou_original = 0
iou_count = 0

# Process each selected image and its corresponding mask
# Process each selected image and its corresponding mask
for image_path, mask_path in selected_images:
    original_image = Image.open(image_path).convert("RGB")  # Open the image directly from the path
    image_tensor = transforms.ToTensor()(original_image)  # Convert the PIL image to a tensor
    denorm_image = denormalize(image_tensor)  # Denormalize the image tensor
    original_image = Image.fromarray(denorm_image).convert("RGB")  # Convert back to PIL image for further processing
    ground_truth_masks = get_ground_truth_masks(mask_path)

    for _ in range(5):  # Process 5 random points
        raw_image = np.array(original_image)
        input_point = [random.randint(0, raw_image.shape[1]-1), random.randint(0, raw_image.shape[0]-1)]

        relevant_gt_mask = next((mask for mask in ground_truth_masks if point_in_mask(input_point, mask)), None)
        if relevant_gt_mask is None:
            continue

        input_points = [[input_point]]
        inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to("cuda")

        with torch.no_grad():
            submodel_outputs = models["submodel"](**inputs)
            submodel_masks = processor.image_processor.post_process_masks(
                submodel_outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            original_outputs = models["original_model"](**inputs)
            original_masks = processor.image_processor.post_process_masks(
                original_outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )

        submodel_predicted_mask = submodel_masks[0].squeeze(0).squeeze(0).cpu()[1] if submodel_masks else None
        original_predicted_mask = original_masks[0].squeeze(0).squeeze(0).cpu()[1] if original_masks else None

        iou_submodel = calculate_metrics(submodel_predicted_mask, relevant_gt_mask) if relevant_gt_mask is not None and submodel_predicted_mask is not None else 0
        iou_original = calculate_metrics(original_predicted_mask, relevant_gt_mask) if relevant_gt_mask is not None and original_predicted_mask is not None else 0

        # Accumulate the IoUs for averaging
        total_iou_submodel += iou_submodel
        total_iou_original += iou_original
        iou_count += 1



# Calculate the average IoUs for the submodel and original model
average_iou_submodel = total_iou_submodel / iou_count if iou_count > 0 else 0
average_iou_original = total_iou_original / iou_count if iou_count > 0 else 0

print(f"Average IoU for Submodel: {average_iou_submodel:.4f}")
print(f"Average IoU for Original Model: {average_iou_original:.4f}")


def train_model(model, train_dataloader, num_epochs=20, learning_rate=1e-8):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_dataloader:
            images = images.to("cuda")
            masks = masks.to("cuda")
            optimizer.zero_grad()
            outputs = model(images)['pred_masks']
            pred_masks = outputs.squeeze(1)
            pred_masks = pred_masks[:, 0, :, :].unsqueeze(1)
            pred_masks_resized = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear', align_corners=False)
            loss_focal = focal_loss(pred_masks_resized, masks)
            loss_dice = dice_loss(pred_masks_resized, masks)
            loss = loss_focal + loss_dice
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory):
        self.image_mask_pairs = get_image_info(dataset_directory, num_images=320)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = get_ground_truth_masks(mask_path)[0]
        image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = transforms.functional.resize(mask, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)
        return image, mask

    
    
    
dataset_directory = "SA1B"
# Define the test_dataset
dataset = SA1BDataset("SA1B")
train_dataset = Subset(dataset, indices=range(0, 256))
test_dataset = Subset(dataset, indices=range(256, 320))
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Combine test_dataset images with selected_images
# Combine test_dataset images with selected_images
test_image_paths = [test_dataset.dataset.image_mask_pairs[i][0] for i in range(len(test_dataset))]
test_mask_paths = [test_dataset.dataset.image_mask_pairs[i][1] for i in range(len(test_dataset))]

selected_images = get_image_info(dataset_directory, num_images=5)
selected_images.extend(list(zip(test_image_paths, test_mask_paths)))


# Continue with the rest of your code...

# Train the model
train_model(submodel, train_dataloader, num_epochs=1, learning_rate=1e-8)

# Calculate average IoUs before training
average_iou_submodel_before_training = calculate_average_iou(submodel, test_dataloader, processor)
average_iou_original_before_training = calculate_average_iou(original_model, test_dataloader, processor)

print(f"Average IoU for Submodel before training: {average_iou_submodel_before_training:.4f}")
print(f"Average IoU for Original Model before training: {average_iou_original_before_training:.4f}")

# Calculate average IoU for the submodel after training
average_iou_submodel_after_training = calculate_average_iou(submodel, test_dataloader, processor)

print(f"Average IoU for Submodel after training: {average_iou_submodel_after_training:.4f}")
