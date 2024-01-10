import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import torchvision.transforms as transforms
import pandas as pd
import os
import shutil
import json
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import argparse
import pickle

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