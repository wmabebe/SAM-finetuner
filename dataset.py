import numpy as np
import torch
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
import json
from pycocotools import mask as mask_utils


class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory, processor):
        self.datadir = dataset_directory
        self.processor = processor
        self.imgs, self.labels = SA1BDataset.get_image_json_pairs(self.datadir)
        self.target_transform = transforms.Compose([
                                 transforms.Resize((1024, 1024)),  # Resize to 1024x1024
                                transforms.ToTensor()  # Convert to tensor
                                ])

    def __len__(self):
        return len(self.imgs)
    
    def loader(self,file_path):
        image = Image.open(file_path)
        image = np.array(image)
        image = np.moveaxis(image, -1, 0)
        return image
    
    @staticmethod
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
    
    @staticmethod
    def get_image_json_pairs(directory):
        jpg_files = []
        json_files = []

        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                jpg_files.append(filename)
                json_file = filename[:-4] + '.json'
                json_files.append(json_file)

        return jpg_files, json_files
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path =  os.path.join(self.datadir, self.imgs[index]) # discard automatic subfolder labels 
        label_path = os.path.join(self.datadir, self.labels[index])
        image = self.loader(img_path)
        masks = json.load(open(label_path))['annotations'] # load json masks

        #Just pick the first mask
        mask = masks[0]
        bin_ground_truth_mask = mask_utils.decode(mask['segmentation'])
        bbox_prompt = SA1BDataset.get_bounding_box(bin_ground_truth_mask)
        bin_ground_truth_mask = torch.tensor(bin_ground_truth_mask)
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[bbox_prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        return inputs, bin_ground_truth_mask
    
class MitoDataset(torch.utils.data.Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)
  
    @staticmethod
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

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])

        ground_truth_mask = torch.tensor(ground_truth_mask)

        # get bounding box prompt
        prompt = MitoDataset.get_bounding_box(ground_truth_mask)

        # Convert the image to grayscale (if it's not already in grayscale)
        image = image.convert("L")

        # Add a channel dimension
        image = image.convert("RGB")

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        return inputs, ground_truth_mask