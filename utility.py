import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms
import os
import json
import torch
import torch.nn.functional as F
import argparse
import random
import json
from pycocotools import mask as coco_mask
import tifffile
from patchify import patchify  #Only to handle large images
from datasets import Dataset as DatasetX
from tqdm import tqdm


def get_logger():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def init_logs(log_file_name, log_dir=None):
    
    #mkdirs(log_dir)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_path = log_file_name + '.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    logging.basicConfig(
        filename=os.path.join(log_dir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='res18', help='Neural network used for encoding')
    parser.add_argument('--embed', type=int, default=1, help='Embed chips to a latent space vector')
    parser.add_argument('--post', type=int, default=0, help='Apply post processing to remove bad masks')
    args = parser.parse_args()
    return args

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = ann["color"] if "color" in ann.keys() else np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


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
    
def show_box(box, ax, label="",linecolor=[]):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='black', facecolor=(0,0,0,0), lw=5))
    if label:
        # Add label to the box
        label_x = x0 + 0.5 * w  # x-coordinate of the label position
        label_y = y0 + 0.5 * h  # y-coordinate of the label position
        ax.text(label_x, label_y, label, fontsize=20, color='red',
                ha='center', va='center')



def mutual_overlap(parent_mask, child_mask):
    intersection_mask = np.logical_and(parent_mask, child_mask)
    inter_inter_child = np.logical_and(intersection_mask, child_mask)
    inter_inter_parent = np.logical_and(intersection_mask, parent_mask)

    intersection_pixels = np.sum(intersection_mask)

    if not intersection_pixels:
        return 0, 0, 0

    child_pixels = np.sum(child_mask)
    parent_pixels = np.sum(parent_mask)
    
    inter_inter_child_pixels = np.sum(inter_inter_child)
    inter_inter_parent_pixels = np.sum(inter_inter_parent)

    child_overlap = inter_inter_child_pixels / child_pixels * 100
    parent_overlap = inter_inter_parent_pixels / parent_pixels * 100
    intersection_overlap = intersection_pixels / parent_pixels * 100
    
    # False, False -> Unrelated masks
    # True, False -> component mask
    # False, True -> reverse component (i.e. parent is the component of child)
    # True, True -> redundant masks

    return child_overlap, parent_overlap, intersection_overlap

def compute_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt)
    union = np.logical_or(mask_pred, mask_gt)
    
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_metrics(predicted_mask, ground_truth_mask):    
    intersection = np.logical_and(predicted_mask, ground_truth_mask)
    union = np.logical_or(predicted_mask, ground_truth_mask)
    true_positives = np.sum(np.logical_and(predicted_mask, ground_truth_mask))
    #true_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), np.logical_not(ground_truth_mask)))
    false_positives = np.sum(np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predicted_mask), ground_truth_mask))
    
    iou = np.sum(intersection) / np.sum(union)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    #fpr = false_positives / (false_positives + true_negatives)
    f1 = 2 * (precision * recall) / (precision + recall) if precision * recall != 0 else 0
    
    return iou, recall, precision, f1

def compute_fpr(predicted_masks, ground_truth_mask):
    fpr = 0
    for predicted_mask in predicted_masks:
        intersection = np.logical_and(predicted_mask, ground_truth_mask)
        union = np.logical_or(predicted_mask, ground_truth_mask)
        iou = np.sum(intersection) / np.sum(union)
        fpr += iou
    return fpr

def toBinaryMask(coco, annIds, input_image_size):
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        print(f'train_mask: {train_mask.shape}, new_mask: {new_mask.shape}')
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

def point_grid(image_size, n):
    """
    Generate a list of coordinates representing an nxn grid within the image's dimensions.

    Args:
    - image_size (tuple): Tuple representing the dimensions of the image (x, y).
    - n (int): Number of points along each dimension for the grid.

    Returns:
    - List of coordinates [(x1, y1), (x2, y2), ..., (xn, yn)].
    """
    x_max, y_max = image_size

    # Calculate the step size between points
    step_x = x_max / (n + 1)
    step_y = y_max / (n + 1)

    # Generate the grid of coordinates
    grid = [[int(i * step_x), int(j * step_y)] for i in range(1, n + 1) for j in range(1, n + 1)]

    return grid

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

def inference(model, dataloader, device='cuda'):
    model.eval()
    image_ious = []
    with torch.no_grad():
        for images, masks in dataloader:

            print(f'images : {images.shape}')
            print(f'masks : {masks.shape}')

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['pred_masks']
            print(f'outputs : {outputs}')
            break 
            # pred_masks = outputs.squeeze(1)
            # pred_masks = pred_masks[:, 0, :, :].unsqueeze(1)
            # pred_masks_resized = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear', align_corners=False)

            # for i in range(images.shape[0]):
            #     gt_mask = masks[i].cpu().numpy().squeeze()
            #     pred_mask = pred_masks_resized[i].cpu().numpy().squeeze()

            #     valid_points = valid_points_from_masks([gt_mask])
            #     random_points = random.sample(valid_points, min(5, len(valid_points)))
            #     point_ious = []
            #     for point in random_points:
            #         x, y = point
            #         iou = calculate_metrics(pred_mask[y:y+256, x:x+256], gt_mask[y:y+256, x:x+256])
            #         point_ious.append(iou)
            #     image_ious.append(sum(point_ious) / len(point_ious) if point_ious else 0)

    # average_iou = sum(image_ious) / len(image_ious) if image_ious else 0
    # return image_ious, average_iou
    return 0, 0

# Load tiff stack images and masks
def load_dataset(data_path, label_path, patch_size=256, step=256):
    
    #165 large images as tiff image stack
    large_images = tifffile.imread(data_path)
    large_masks = tifffile.imread(label_path)

    large_images.shape

    """Now. let us divide these large images into smaller patches for training. We can use patchify or write custom code."""
    all_img_patches = []
    for img in range(large_images.shape[0]):
        large_image = large_images[img]
        patches_img = patchify(large_image, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):

                single_patch_img = patches_img[i,j,:,:]
                all_img_patches.append(single_patch_img)

    images = np.array(all_img_patches)

    #Let us do the same for masks
    all_mask_patches = []
    for img in range(large_masks.shape[0]):
        large_mask = large_masks[img]
        patches_mask = patchify(large_mask, (patch_size, patch_size), step=step)  #Step=256 for 256 patches means no overlap

        for i in range(patches_mask.shape[0]):
            for j in range(patches_mask.shape[1]):

                single_patch_mask = patches_mask[i,j,:,:]
                single_patch_mask = (single_patch_mask / 255.).astype(np.uint8)
                all_mask_patches.append(single_patch_mask)

    masks = np.array(all_mask_patches)

    print("images.shape:",images.shape)
    print("masks.shape:",masks.shape)

    """Now, let us delete empty masks as they may cause issues later on during training. If a batch contains empty masks then the loss function will throw an error as it may not know how to handle empty tensors."""

    # Create a list to store the indices of non-empty masks
    valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
    # Filter the image and mask arrays to keep only the non-empty pairs
    filtered_images = images[valid_indices]
    filtered_masks = masks[valid_indices]
    print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
    print("Mask shape:", filtered_masks.shape)

    """Let us create a 'dataset' that serves us input images and masks for the rest of our journey."""



    # Convert the NumPy arrays to Pillow images and store them in a dictionary
    dataset_dict = {
        "image": [Image.fromarray(img) for img in filtered_images],
        "label": [Image.fromarray(mask) for mask in filtered_masks],
    }

    # Create the dataset using the datasets.Dataset class
    dataset = DatasetX.from_dict(dataset_dict)

    return dataset


"""Get bounding boxes from masks. You can get here directly if you are working with coco style annotations where bounding boxes are captured in a JSON file."""

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

def compute_iou(predicted_masks, gt_masks):

    # Convert predicted and ground truth masks to boolean tensors
    predicted_masks_bool = predicted_masks.bool()
    gt_masks_bool = gt_masks.bool()

    # Compute intersection and union
    intersection = (predicted_masks_bool & gt_masks_bool).sum(dim=(1, 2)).float()
    union = (predicted_masks_bool | gt_masks_bool).sum(dim=(1, 2)).float()

    # Compute IoU
    iou = intersection / (union + 1e-10)  # Adding epsilon to avoid division by zero

    iou = torch.mean(iou)

    return round(iou.item(), 4)
  
def get_prompt_grid(size=10, batch_size=1):
    # Define the size of your array
   array_size = 256

    # Define the size of your grid
   grid_size = size

    # Generate the grid points
   x = np.linspace(0, array_size-1, grid_size)
   y = np.linspace(0, array_size-1, grid_size)

    # Generate a grid of coordinates
   xv, yv = np.meshgrid(x, y)

    # Convert the numpy arrays to lists
   xv_list = xv.tolist()
   yv_list = yv.tolist()

    # Combine the x and y coordinates into a list of list of lists
   input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]
   input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)

   input_points = input_points.repeat(batch_size, 1, 1, 1)
   
   return input_points

#Test model performance on given dataset
def test(model,dataloader,size=10000,grid_size=10,batch_size=2,device='cuda',disable_verbose=False):
    mIoU = []
    count = 0
    input_points = get_prompt_grid(grid_size,batch_size)
    print(f'input_points :  {input_points.shape}')
    model.eval()
    model = model.to(device)
    for batch in tqdm(dataloader, disable=disable_verbose):

        # inputs = processor(batch["pixel_values"], input_points=input_points, return_tensors="pt")
        # inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward pass
        with torch.no_grad():
          outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_points=input_points.to(device),
                        multimask_output=False)
          #outputs = my_mito_model(**inputs, multimask_output=False)

        # compute iou
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)


        # apply sigmoid
        single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        single_patch_prob = single_patch_prob.squeeze()
        predicted_masks_binary = (single_patch_prob > 0.5)


        mIoU.append(compute_iou(predicted_masks_binary,ground_truth_masks))

        if count == size:
          break

        count += 1
    return round(sum(mIoU) / len(mIoU), 4) 